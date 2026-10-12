from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

Point = Sequence[float]


def canonicalize_quadrilateral(polygon: Sequence[Point]) -> list[list[float]]:
    """Order four corners as top-left, top-right, bottom-right, bottom-left."""
    if len(polygon) != 4 or any(len(point) != 2 for point in polygon):
        raise ValueError("a document quadrilateral must contain four 2D points")
    points = [(float(point[0]), float(point[1])) for point in polygon]
    center_x = sum(point[0] for point in points) / 4
    center_y = sum(point[1] for point in points) / 4
    points.sort(key=lambda point: math.atan2(point[1] - center_y, point[0] - center_x))
    start = min(range(4), key=lambda index: sum(points[index]))
    ordered = points[start:] + points[:start]
    cross = (ordered[1][0] - ordered[0][0]) * (
        ordered[2][1] - ordered[1][1]
    ) - (ordered[1][1] - ordered[0][1]) * (
        ordered[2][0] - ordered[1][0]
    )
    if cross < 0:
        ordered = [ordered[0], ordered[3], ordered[2], ordered[1]]
    return [[x, y] for x, y in ordered]


@dataclass(frozen=True)
class LetterboxTransform:
    source_width: int
    source_height: int
    input_size: int
    scale: float
    pad_x: int
    pad_y: int

    def source_to_input(
        self, points: Sequence[Point], *, normalized: bool = False
    ) -> list[list[float]]:
        transformed = [
            [point[0] * self.scale + self.pad_x, point[1] * self.scale + self.pad_y]
            for point in points
        ]
        if normalized:
            return [
                [point[0] / self.input_size, point[1] / self.input_size]
                for point in transformed
            ]
        return transformed

    def input_to_source(
        self, points: Sequence[Point], *, normalized: bool = False
    ) -> list[list[float]]:
        restored: list[list[float]] = []
        for point in points:
            x = point[0] * self.input_size if normalized else point[0]
            y = point[1] * self.input_size if normalized else point[1]
            restored.append(
                [(x - self.pad_x) / self.scale, (y - self.pad_y) / self.scale]
            )
        return restored


def letterbox_document(
    image: Image.Image,
    polygon: Sequence[Point],
    *,
    size: int,
    fill: tuple[int, int, int] = (114, 114, 114),
) -> tuple[torch.Tensor, list[list[float]], LetterboxTransform]:
    if size <= 0:
        raise ValueError("size must be positive")
    source = image.convert("RGB")
    scale = min(size / source.width, size / source.height)
    resized_width = max(1, round(source.width * scale))
    resized_height = max(1, round(source.height * scale))
    pad_x = (size - resized_width) // 2
    pad_y = (size - resized_height) // 2
    canvas = Image.new("RGB", (size, size), color=fill)
    resized = source.resize(
        (resized_width, resized_height), resample=Image.Resampling.BILINEAR
    )
    canvas.paste(resized, (pad_x, pad_y))
    transform = LetterboxTransform(
        source_width=source.width,
        source_height=source.height,
        input_size=size,
        scale=scale,
        pad_x=pad_x,
        pad_y=pad_y,
    )
    ordered = canonicalize_quadrilateral(polygon)
    normalized = transform.source_to_input(ordered, normalized=True)
    tensor = pil_to_tensor(canvas).to(dtype=torch.float32).div_(255.0)
    return tensor, normalized, transform
