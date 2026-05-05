"""Generate synthetic calibration-board images for local testing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT_DIR / "demo_output" / "calibration_synthetic"


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _make_canvas(width: int = 760, height: int = 520, background: int = 20) -> np.ndarray:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:] = background
    return image


def _apply_perspective(
    image: np.ndarray,
    source_quad: np.ndarray,
    target_quad: np.ndarray,
    border_value: int = 10,
) -> np.ndarray:
    matrix = cv2.getPerspectiveTransform(source_quad, target_quad)
    return cv2.warpPerspective(
        image,
        matrix,
        (image.shape[1], image.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderValue=(border_value, border_value, border_value),
    )


def _save_image(path: Path, image: np.ndarray) -> None:
    _ensure_dir(path.parent)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Failed to write image: {path}")


def _draw_symmetric_circles(
    image: np.ndarray,
    pattern_size: Tuple[int, int],
    origin: Tuple[int, int],
    spacing: Tuple[int, int],
    radius: int,
) -> None:
    width, height = pattern_size
    ox, oy = origin
    sx, sy = spacing
    for y in range(height):
        for x in range(width):
            center = (ox + x * sx, oy + y * sy)
            cv2.circle(image, center, radius, (255, 255, 255), -1)


def _draw_asymmetric_circles(
    image: np.ndarray,
    pattern_size: Tuple[int, int],
    origin: Tuple[int, int],
    spacing: int,
    radius: int,
) -> None:
    width, height = pattern_size
    ox, oy = origin
    for y in range(height):
        for x in range(width):
            px = ox + int((2 * x + (y % 2)) * spacing / 2)
            py = oy + y * spacing
            cv2.circle(image, (px, py), radius, (255, 255, 255), -1)


def _draw_volume_dots(
    image: np.ndarray,
    pattern_size: Tuple[int, int],
    origin: Tuple[int, int],
    spacing: Tuple[int, int],
    radius: int,
    missing: Sequence[Tuple[int, int]],
) -> None:
    width, height = pattern_size
    ox, oy = origin
    sx, sy = spacing
    missing_set = set(missing)
    for y in range(height):
        for x in range(width):
            if (x, y) in missing_set:
                continue
            center = (ox + x * sx, oy + y * sy)
            cv2.circle(image, center, radius, (255, 255, 255), -1)


def _make_volume_dots_images() -> List[Dict[str, object]]:
    pattern_size = (11, 8)
    origin = (110, 80)
    spacing = (48, 48)
    radius = 9
    missing = [(5, 3), (5, 4), (6, 3), (6, 4)]

    source_quad = np.float32([[80, 60], [680, 70], [670, 460], [90, 450]])
    target_quads = [
        np.float32([[120, 80], [640, 50], [700, 470], [80, 430]]),
        np.float32([[90, 100], [660, 80], [690, 450], [100, 440]]),
        np.float32([[130, 70], [620, 60], [710, 460], [60, 420]]),
    ]

    outputs = []
    out_dir = OUTPUT_DIR / "volume_dots"
    _ensure_dir(out_dir)

    for index, quad in enumerate(target_quads, start=1):
        image = _make_canvas()
        _draw_volume_dots(image, pattern_size, origin, spacing, radius, missing)
        warped = _apply_perspective(image, source_quad, quad)
        file_path = out_dir / f"volume_dots_{index:02d}.png"
        _save_image(file_path, warped)
        outputs.append(
            {
                "path": str(file_path.relative_to(ROOT_DIR)),
                "pattern_type": "volume_dots",
                "pattern_size": list(pattern_size),
                "missing_points": [list(item) for item in missing],
            }
        )

    return outputs


def _make_symmetric_circle_images() -> List[Dict[str, object]]:
    pattern_size = (9, 6)
    origin = (110, 80)
    spacing = (48, 48)
    radius = 9
    source_quad = np.float32([[80, 60], [640, 60], [650, 410], [90, 420]])
    target_quads = [
        np.float32([[90, 70], [650, 50], [680, 430], [80, 410]]),
        np.float32([[110, 60], [620, 80], [670, 420], [100, 440]]),
        np.float32([[85, 90], [655, 65], [690, 410], [95, 435]]),
    ]

    outputs = []
    out_dir = OUTPUT_DIR / "circles"
    _ensure_dir(out_dir)

    for index, quad in enumerate(target_quads, start=1):
        image = _make_canvas()
        _draw_symmetric_circles(image, pattern_size, origin, spacing, radius)
        warped = _apply_perspective(image, source_quad, quad)
        file_path = out_dir / f"circles_{index:02d}.png"
        _save_image(file_path, warped)
        outputs.append(
            {
                "path": str(file_path.relative_to(ROOT_DIR)),
                "pattern_type": "circles",
                "pattern_size": list(pattern_size),
            }
        )

    return outputs


def _make_asymmetric_circle_images() -> List[Dict[str, object]]:
    pattern_size = (9, 6)
    origin = (90, 70)
    spacing = 44
    radius = 8
    source_quad = np.float32([[70, 50], [470, 55], [500, 330], [60, 340]])
    target_quads = [
        np.float32([[90, 70], [480, 50], [510, 330], [70, 340]]),
        np.float32([[80, 85], [470, 60], [520, 335], [85, 350]]),
        np.float32([[95, 60], [455, 78], [525, 340], [65, 320]]),
    ]

    outputs = []
    out_dir = OUTPUT_DIR / "acircles"
    _ensure_dir(out_dir)

    for index, quad in enumerate(target_quads, start=1):
        image = _make_canvas(width=560, height=400)
        _draw_asymmetric_circles(image, pattern_size, origin, spacing, radius)
        warped = _apply_perspective(image, source_quad, quad)
        file_path = out_dir / f"acircles_{index:02d}.png"
        _save_image(file_path, warped)
        outputs.append(
            {
                "path": str(file_path.relative_to(ROOT_DIR)),
                "pattern_type": "acircles",
                "pattern_size": list(pattern_size),
            }
        )

    return outputs


def generate_all() -> Dict[str, object]:
    _ensure_dir(OUTPUT_DIR)
    generated = {
        "volume_dots": _make_volume_dots_images(),
        "circles": _make_symmetric_circle_images(),
        "acircles": _make_asymmetric_circle_images(),
    }

    manifest = {
        "output_dir": str(OUTPUT_DIR.relative_to(ROOT_DIR)),
        "generated_sets": generated,
    }
    manifest_path = OUTPUT_DIR / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return manifest


if __name__ == "__main__":
    result = generate_all()
    print(json.dumps(result, indent=2, ensure_ascii=False))
