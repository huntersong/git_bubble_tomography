"""Generate synchronized checkerboard images for all GUI calibration modes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Sequence, Tuple

import cv2
import numpy as np


PATTERN_SIZE = (11, 8)  # inner corners: width, height
SQUARE_SIZE_MM = 5.0
IMAGE_SIZE = (1280, 960)
FRAME_COUNT = 12
SUPERSAMPLE = 2

CAMERA_MATRIX = np.array(
    [
        [1150.0, 0.0, IMAGE_SIZE[0] / 2.0],
        [0.0, 1140.0, IMAGE_SIZE[1] / 2.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
DIST_COEFFS = np.zeros(5, dtype=np.float64)


def _board_poses() -> Sequence[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:
    """Return deterministic board-to-world poses as degrees and millimetres."""
    return [
        ((-8, -12, -5), (-12.5, -7.5, 153.333)),
        ((7, -9, 4), (10.833, -9.167, 163.333)),
        ((-10, 8, 6), (-7.5, 10.0, 173.333)),
        ((9, 11, -6), (11.667, 7.5, 150.0)),
        ((-5, -15, 2), (0.0, -11.667, 183.333)),
        ((12, 5, -3), (-13.333, 3.333, 160.0)),
        ((-12, 3, 7), (13.333, -0.833, 170.0)),
        ((5, 14, -8), (3.333, 10.833, 156.667)),
        ((-7, -6, 9), (-4.167, -2.5, 143.333)),
        ((10, -2, -9), (7.5, 2.5, 180.0)),
        ((-3, 10, 3), (-11.667, 9.167, 165.0)),
        ((4, -11, -1), (12.5, 10.0, 176.667)),
    ]


def _rotation_matrix_xyz(angles_deg: Iterable[float]) -> np.ndarray:
    rx, ry, rz = np.radians(tuple(angles_deg))
    rotation_x = np.array(
        [[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]],
        dtype=np.float64,
    )
    rotation_y = np.array(
        [[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]],
        dtype=np.float64,
    )
    rotation_z = np.array(
        [[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]],
        dtype=np.float64,
    )
    return rotation_z @ rotation_y @ rotation_x


def _project(
    points_board: np.ndarray,
    board_rotation: np.ndarray,
    board_translation: np.ndarray,
    camera_position: np.ndarray,
    scaled_camera_matrix: np.ndarray,
) -> np.ndarray:
    rotation_camera_board = board_rotation
    translation_camera_board = board_translation - camera_position
    rotation_vector, _ = cv2.Rodrigues(rotation_camera_board)
    image_points, _ = cv2.projectPoints(
        points_board,
        rotation_vector,
        translation_camera_board,
        scaled_camera_matrix,
        DIST_COEFFS,
    )
    return np.round(image_points.reshape(-1, 2)).astype(np.int32)


def _render_checkerboard(
    angles_deg: Tuple[float, float, float],
    translation_mm: Tuple[float, float, float],
    camera_position_mm: Tuple[float, float, float],
) -> np.ndarray:
    width, height = IMAGE_SIZE
    scale = SUPERSAMPLE
    canvas = np.full((height * scale, width * scale, 3), 205, dtype=np.uint8)
    camera_matrix = CAMERA_MATRIX.copy()
    camera_matrix[:2, :] *= scale

    inner_w, inner_h = PATTERN_SIZE
    square_cols, square_rows = inner_w + 1, inner_h + 1
    board_width = square_cols * SQUARE_SIZE_MM
    board_height = square_rows * SQUARE_SIZE_MM
    x_origin = -board_width / 2.0
    y_origin = -board_height / 2.0

    rotation = _rotation_matrix_xyz(angles_deg)
    translation = np.asarray(translation_mm, dtype=np.float64).reshape(3, 1)
    camera_position = np.asarray(camera_position_mm, dtype=np.float64).reshape(3, 1)

    outer = np.array(
        [
            [x_origin, y_origin, 0.0],
            [x_origin + board_width, y_origin, 0.0],
            [x_origin + board_width, y_origin + board_height, 0.0],
            [x_origin, y_origin + board_height, 0.0],
        ],
        dtype=np.float64,
    )
    outer_image = _project(
        outer, rotation, translation, camera_position, camera_matrix
    )
    cv2.fillConvexPoly(canvas, outer_image, (245, 245, 245), cv2.LINE_AA)

    for row in range(square_rows):
        for col in range(square_cols):
            if (row + col) % 2 == 0:
                continue
            x0 = x_origin + col * SQUARE_SIZE_MM
            y0 = y_origin + row * SQUARE_SIZE_MM
            square = np.array(
                [
                    [x0, y0, 0.0],
                    [x0 + SQUARE_SIZE_MM, y0, 0.0],
                    [
                        x0 + SQUARE_SIZE_MM,
                        y0 + SQUARE_SIZE_MM,
                        0.0,
                    ],
                    [x0, y0 + SQUARE_SIZE_MM, 0.0],
                ],
                dtype=np.float64,
            )
            polygon = _project(
                square, rotation, translation, camera_position, camera_matrix
            )
            cv2.fillConvexPoly(canvas, polygon, (12, 12, 12), cv2.LINE_AA)

    canvas = cv2.GaussianBlur(canvas, (3, 3), 0.55)
    return cv2.resize(canvas, IMAGE_SIZE, interpolation=cv2.INTER_AREA)


def _write_set(
    output_dir: Path,
    cameras: Dict[str, Tuple[float, float, float]],
) -> Dict[str, object]:
    poses = _board_poses()
    camera_entries: Dict[str, object] = {}
    for camera_name, camera_position in cameras.items():
        camera_dir = output_dir / camera_name
        camera_dir.mkdir(parents=True, exist_ok=True)
        image_paths = []
        for frame_index, (angles, translation) in enumerate(poses, start=1):
            image = _render_checkerboard(angles, translation, camera_position)
            image_path = camera_dir / f"frame_{frame_index:02d}.png"
            if not cv2.imwrite(str(image_path), image):
                raise OSError(f"Unable to write {image_path}")
            image_paths.append(str(image_path))
        camera_entries[camera_name] = {
            "position_mm": list(camera_position),
            "images": image_paths,
        }
    return {
        "frame_count": len(poses),
        "cameras": camera_entries,
    }


def generate(output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    generated = {
        "single_camera": _write_set(
            output_root / "single_camera",
            {"camera_1": (0.0, 0.0, 0.0)},
        ),
        "stereo": _write_set(
            output_root / "stereo",
            {
                "left": (-13.333, 0.0, 0.0),
                "right": (13.333, 0.0, 0.0),
            },
        ),
        "multi_camera": _write_set(
            output_root / "multi_camera",
            {
                "cam1": (-15.0, -9.167, 0.0),
                "cam2": (15.0, -9.167, 0.0),
                "cam3": (-15.0, 9.167, 0.0),
                "cam4": (15.0, 9.167, 0.0),
            },
        ),
    }
    manifest = {
        "pattern_type": "checkerboard",
        "pattern_size_inner_corners": list(PATTERN_SIZE),
        "gui_square_size_mm": SQUARE_SIZE_MM,
        "image_size_px": list(IMAGE_SIZE),
        "synchronized_by_filename": True,
        "datasets": generated,
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "bubble1"
        / "calibration_images",
    )
    args = parser.parse_args()
    print(generate(args.output.resolve()))


if __name__ == "__main__":
    main()
