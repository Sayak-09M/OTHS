from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from pii_id.augmentation.degradations import DegradationAugmenter
from pii_id.evaluation.access import AccessGrant
from pii_id.geometry import letterbox_document
from pii_id.models.document import QUALITY_LABELS

from .loader import DatasetManifest

_IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
_IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)


@dataclass(frozen=True)
class DocumentLabelSpace:
    classes: tuple[str, ...]

    @classmethod
    def from_records(cls, records: Iterable[dict[str, object]]) -> DocumentLabelSpace:
        document_types = sorted(
            {
                str(record["document_type"])
                for record in records
                if record["document_type"] != "NON_ID"
            }
        )
        return cls(classes=("UNKNOWN", *document_types))

    def index(self, document_type: str) -> int:
        try:
            return self.classes.index(document_type)
        except ValueError:
            return 0


class TorchDocumentDataset(Dataset[dict[str, object]]):
    """PII-minimized PyTorch view over an authorized canonical split."""

    def __init__(
        self,
        *,
        manifest: DatasetManifest,
        grant: AccessGrant,
        label_space: DocumentLabelSpace,
        input_size: int = 224,
        augmenter: DegradationAugmenter | None = None,
    ) -> None:
        if input_size < 32:
            raise ValueError("input_size must be at least 32")
        self.manifest = manifest
        self.grant = grant
        self.label_space = label_space
        self.input_size = input_size
        self.augmenter = augmenter
        self.epoch = 0
        self.records = list(manifest.iter_records())
        if not self.records:
            raise ValueError("dataset manifest is empty")
        if any(
            not grant.allows(str(record["split"]), grant.purpose)
            for record in self.records
        ):
            raise ValueError("dataset contains a split not allowed by the access grant")

    def __len__(self) -> int:
        return len(self.records)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index]
        loaded = self.manifest.load_sample(record)
        source_image = loaded.image
        source_polygon = record["annotations"]["document_polygon"]
        quality = torch.zeros(len(QUALITY_LABELS), dtype=torch.float32)
        if self.augmenter is not None:
            degraded = self.augmenter.apply_random(
                source_image,
                source_polygon,
                record_key=str(record["record_id"]),
                epoch=self.epoch,
            )
            source_image = degraded.image
            source_polygon = degraded.polygon
            quality = torch.tensor(degraded.labels, dtype=torch.float32)
        image, corners, transform = letterbox_document(
            source_image,
            source_polygon,
            size=self.input_size,
        )
        normalized_image = (image - _IMAGENET_MEAN) / _IMAGENET_STD
        is_identity_document = record["document_type"] != "NON_ID"
        return {
            "image": normalized_image,
            "presence": torch.tensor(float(is_identity_document), dtype=torch.float32),
            "corners": torch.tensor(corners, dtype=torch.float32),
            "document_type": torch.tensor(
                self.label_space.index(str(record["document_type"])),
                dtype=torch.long,
            ),
            "quality": quality,
            "metadata": {
                "record_id": record["record_id"],
                "dataset_id": record["dataset_id"],
                "country": record["country"],
                "script": record["script"],
                "modality": record["modality"],
                "source_width": transform.source_width,
                "source_height": transform.source_height,
                "input_size": transform.input_size,
                "scale": transform.scale,
                "pad_x": transform.pad_x,
                "pad_y": transform.pad_y,
            },
        }
