from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from PIL import Image
from torch.nn import functional as F
from torchvision.transforms.functional import pil_to_tensor, to_pil_image

from pii_id.geometry import canonicalize_quadrilateral

Point = Sequence[float]


def _homography(source: Sequence[Point], destination: Sequence[Point]) -> torch.Tensor:
    rows: list[list[float]] = []
    values: list[float] = []
    for (x, y), (u, v) in zip(source, destination, strict=True):
        rows.extend(
            (
                [x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y],
                [0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y],
            )
        )
        values.extend((u, v))
    matrix = torch.tensor(rows, dtype=torch.float64)
    target = torch.tensor(values, dtype=torch.float64)
    try:
        solution = torch.linalg.solve(matrix, target)
    except torch.linalg.LinAlgError as error:
        raise ValueError("document corners form a degenerate quadrilateral") from error
    return torch.cat((solution, torch.ones(1, dtype=torch.float64))).reshape(3, 3)


def _map_points(matrix: torch.Tensor, points: Sequence[Point]) -> list[list[float]]:
    homogeneous = torch.tensor(
        [[float(x), float(y), 1.0] for x, y in points], dtype=torch.float64
    )
    mapped = homogeneous @ matrix.T
    mapped = mapped[:, :2] / mapped[:, 2:].clamp(min=1e-12)
    return mapped.tolist()


@dataclass(frozen=True)
class RectificationTransform:
    source_to_rectified_matrix: torch.Tensor

    def source_to_rectified(self, points: Sequence[Point]) -> list[list[float]]:
        return _map_points(self.source_to_rectified_matrix, points)

    def rectified_to_source(self, points: Sequence[Point]) -> list[list[float]]:
        return _map_points(torch.linalg.inv(self.source_to_rectified_matrix), points)


def rectify_document(
    image: Image.Image,
    corners: Sequence[Point],
    *,
    output_size: tuple[int, int],
) -> tuple[Image.Image, RectificationTransform]:
    output_width, output_height = output_size
    if output_width < 2 or output_height < 2:
        raise ValueError("rectified output must be at least 2x2")
    source = canonicalize_quadrilateral(corners)
    destination = [
        [0.0, 0.0],
        [output_width - 1.0, 0.0],
        [output_width - 1.0, output_height - 1.0],
        [0.0, output_height - 1.0],
    ]
    source_to_destination = _homography(source, destination)
    destination_to_source = torch.linalg.inv(source_to_destination)
    y_coordinates, x_coordinates = torch.meshgrid(
        torch.arange(output_height, dtype=torch.float64),
        torch.arange(output_width, dtype=torch.float64),
        indexing="ij",
    )
    destination_points = torch.stack(
        (
            x_coordinates.reshape(-1),
            y_coordinates.reshape(-1),
            torch.ones(output_width * output_height, dtype=torch.float64),
        ),
        dim=1,
    )
    source_points = destination_points @ destination_to_source.T
    source_points = source_points[:, :2] / source_points[:, 2:].clamp(min=1e-12)
    source_width, source_height = image.size
    normalized_x = source_points[:, 0] / max(1, source_width - 1) * 2 - 1
    normalized_y = source_points[:, 1] / max(1, source_height - 1) * 2 - 1
    grid = torch.stack((normalized_x, normalized_y), dim=1).reshape(
        1, output_height, output_width, 2
    )
    image_tensor = pil_to_tensor(image.convert("RGB")).to(torch.float32).div_(255.0)
    rectified = F.grid_sample(
        image_tensor.unsqueeze(0),
        grid.to(dtype=torch.float32),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ).squeeze(0)
    return (
        to_pil_image(rectified.clamp(0.0, 1.0)),
        RectificationTransform(source_to_destination),
    )
