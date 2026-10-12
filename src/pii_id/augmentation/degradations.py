from __future__ import annotations

import hashlib
import io
import random
from collections.abc import Sequence
from dataclasses import dataclass

from PIL import Image, ImageDraw, ImageFilter

from pii_id.models.document import QUALITY_LABELS


@dataclass(frozen=True)
class DegradationResult:
    image: Image.Image
    polygon: list[list[float]]
    labels: tuple[float, ...]
    severities: dict[str, float]


class DegradationAugmenter:
    def __init__(self, *, seed: int) -> None:
        self.seed = seed

    def apply_named(
        self,
        image: Image.Image,
        polygon: Sequence[Sequence[float]],
        *,
        name: str,
        severity: float,
    ) -> DegradationResult:
        if name not in QUALITY_LABELS:
            raise ValueError("unknown degradation")
        if not 0.0 < severity <= 1.0:
            raise ValueError("severity must be in (0, 1]")
        source = image.convert("RGB")
        points = [[float(x), float(y)] for x, y in polygon]
        operation = getattr(self, f"_{name}")
        degraded, transformed = operation(source, points, severity)
        labels = tuple(float(label == name) for label in QUALITY_LABELS)
        return DegradationResult(
            image=degraded,
            polygon=transformed,
            labels=labels,
            severities={name: severity},
        )

    def apply_random(
        self,
        image: Image.Image,
        polygon: Sequence[Sequence[float]],
        *,
        record_key: str,
        epoch: int,
    ) -> DegradationResult:
        digest = hashlib.sha256(
            f"{self.seed}:{epoch}:{record_key}".encode()
        ).digest()
        generator = random.Random(int.from_bytes(digest[:8], "big"))
        count = generator.choices((0, 1, 2), weights=(2, 6, 2), k=1)[0]
        names = generator.sample(QUALITY_LABELS, k=count)
        current_image = image.convert("RGB")
        current_polygon = [[float(x), float(y)] for x, y in polygon]
        labels = [0.0] * len(QUALITY_LABELS)
        severities: dict[str, float] = {}
        for name in names:
            severity = generator.uniform(0.25, 1.0)
            result = self.apply_named(
                current_image, current_polygon, name=name, severity=severity
            )
            current_image = result.image
            current_polygon = result.polygon
            labels[QUALITY_LABELS.index(name)] = 1.0
            severities[name] = severity
        return DegradationResult(
            image=current_image,
            polygon=current_polygon,
            labels=tuple(labels),
            severities=severities,
        )

    @staticmethod
    def _blur(
        image: Image.Image, polygon: list[list[float]], severity: float
    ) -> tuple[Image.Image, list[list[float]]]:
        blurred = image.filter(ImageFilter.GaussianBlur(radius=0.5 + 3 * severity))
        return blurred, polygon

    @staticmethod
    def _glare(
        image: Image.Image, polygon: list[list[float]], severity: float
    ) -> tuple[Image.Image, list[list[float]]]:
        overlay = Image.new("RGBA", image.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)
        width, height = image.size
        radius_x = int(width * (0.12 + 0.18 * severity))
        radius_y = int(height * (0.15 + 0.20 * severity))
        center_x, center_y = int(width * 0.68), int(height * 0.35)
        draw.ellipse(
            (
                center_x - radius_x,
                center_y - radius_y,
                center_x + radius_x,
                center_y + radius_y,
            ),
            fill=(255, 255, 255, int(90 + 140 * severity)),
        )
        composited = Image.alpha_composite(image.convert("RGBA"), overlay)
        return composited.convert("RGB"), polygon

    @staticmethod
    def _shadow(
        image: Image.Image, polygon: list[list[float]], severity: float
    ) -> tuple[Image.Image, list[list[float]]]:
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        width, height = image.size
        draw.polygon(
            ((0, 0), (int(width * 0.65), 0), (int(width * 0.35), height), (0, height)),
            fill=(0, 0, 0, int(45 + 120 * severity)),
        )
        composited = Image.alpha_composite(image.convert("RGBA"), overlay)
        return composited.convert("RGB"), polygon

    @staticmethod
    def _low_resolution(
        image: Image.Image, polygon: list[list[float]], severity: float
    ) -> tuple[Image.Image, list[list[float]]]:
        factor = max(0.12, 1.0 - 0.82 * severity)
        reduced = image.resize(
            (max(1, round(image.width * factor)), max(1, round(image.height * factor))),
            resample=Image.Resampling.BILINEAR,
        )
        return reduced.resize(image.size, resample=Image.Resampling.BILINEAR), polygon

    @staticmethod
    def _partial_crop(
        image: Image.Image, polygon: list[list[float]], severity: float
    ) -> tuple[Image.Image, list[list[float]]]:
        margin_x = max(1, round(image.width * 0.22 * severity))
        margin_y = max(1, round(image.height * 0.12 * severity))
        right = image.width - max(1, margin_x // 3)
        bottom = image.height - max(1, margin_y // 3)
        cropped = image.crop((margin_x, margin_y, right, bottom))
        scale_x = image.width / cropped.width
        scale_y = image.height / cropped.height
        transformed = [
            [(x - margin_x) * scale_x, (y - margin_y) * scale_y]
            for x, y in polygon
        ]
        return (
            cropped.resize(image.size, resample=Image.Resampling.BILINEAR),
            transformed,
        )

    @staticmethod
    def _compression(
        image: Image.Image, polygon: list[list[float]], severity: float
    ) -> tuple[Image.Image, list[list[float]]]:
        encoded = io.BytesIO()
        image.save(encoded, format="JPEG", quality=max(10, round(95 - 80 * severity)))
        encoded.seek(0)
        with Image.open(encoded) as decoded:
            return decoded.convert("RGB").copy(), polygon

    @staticmethod
    def _perspective(
        image: Image.Image, polygon: list[list[float]], severity: float
    ) -> tuple[Image.Image, list[list[float]]]:
        shear = 0.22 * severity
        offset = shear * image.height / 2
        transformed_image = image.transform(
            image.size,
            Image.Transform.AFFINE,
            (1, -shear, offset, 0, 1, 0),
            resample=Image.Resampling.BILINEAR,
            fillcolor=(114, 114, 114),
        )
        transformed_polygon = [
            [x + shear * y - offset, y] for x, y in polygon
        ]
        return transformed_image, transformed_polygon
