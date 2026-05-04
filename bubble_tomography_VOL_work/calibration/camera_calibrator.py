"""
Multi-camera calibration utilities.

Supported target types:
- checkerboard
- circles
- acircles
- volume_dots
"""

from __future__ import annotations

import glob
import json
import logging
import os
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

PointId = Tuple[float, ...]


@dataclass
class CameraParams:
    """Calibration parameters for one camera."""

    camera_id: str
    image_size: Tuple[int, int]
    camera_matrix: List[List[float]]
    dist_coeffs: List[float]
    rvec: List[float]
    tvec: List[float]
    rms_error: float

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "CameraParams":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)


@dataclass
class PatternObservation:
    """Matched 2D-3D point observation on one image."""

    image_points: np.ndarray
    object_points: np.ndarray
    point_ids: List[PointId]


class MultiCameraCalibrator:
    """
    Calibrator for multi-camera tomography workflows.

    `volume_dots` is intended for bright circular dot targets similar to a
    volume-calibration plate. It supports incomplete regular grids by:
    1. detecting circular blobs;
    2. recovering the lattice topology;
    3. calibrating from the visible subset of indexed dots.
    """

    SUPPORTED_PATTERN_TYPES = {
        "checkerboard",
        "circles",
        "acircles",
        "volume_dots",
    }
    PATTERN_PRIORITY = {
        "checkerboard": 4,
        "acircles": 3,
        "circles": 2,
        "volume_dots": 1,
    }

    def __init__(
        self,
        pattern_type: str = "checkerboard",
        pattern_size: Tuple[int, int] = (11, 8),
        square_size: float = 1.0,
        circle_radius: float = 0.5,
        level_separation: Optional[float] = None,
    ):
        if pattern_type not in self.SUPPORTED_PATTERN_TYPES:
            raise ValueError(f"Unsupported pattern type: {pattern_type}")

        self.pattern_type = pattern_type
        self.pattern_size = tuple(pattern_size)
        self.square_size = float(square_size)
        self.circle_radius = float(circle_radius)
        self.level_separation = (
            float(level_separation)
            if level_separation is not None
            else max(1.0, 0.2 * float(square_size))
        )

        self.obj_points = self._generate_object_points()
        self.camera_params: Dict[str, CameraParams] = {}
        self._calib_data: Dict[str, dict] = {}

    def _generate_grid_point_ids(self) -> List[PointId]:
        w, h = self.pattern_size
        return [(x, y) for y in range(h) for x in range(w)]

    def _point_sort_key(self, point_id: PointId) -> Tuple[float, ...]:
        if len(point_id) == 3:
            return (point_id[0], point_id[2], point_id[1])
        return (point_id[1], point_id[0])

    @classmethod
    def infer_pattern_spec_from_paths(
        cls,
        image_paths: Sequence[str],
        candidate_types: Optional[Sequence[str]] = None,
        size_min: int = 3,
        size_max: int = 20,
        max_images: int = 3,
    ) -> Optional[Dict[str, object]]:
        aggregated: Dict[Tuple[str, Tuple[int, int]], Dict[str, object]] = {}

        for image_path in image_paths[:max_images]:
            image = cv2.imread(image_path)
            if image is None:
                continue

            result = cls.infer_pattern_spec(
                image,
                candidate_types=candidate_types,
                size_min=size_min,
                size_max=size_max,
            )
            if result is None:
                continue

            key = (str(result["pattern_type"]), tuple(result["pattern_size"]))
            bucket = aggregated.setdefault(
                key,
                {
                    "pattern_type": result["pattern_type"],
                    "pattern_size": tuple(result["pattern_size"]),
                    "score_sum": 0.0,
                    "votes": 0,
                },
            )
            bucket["score_sum"] += float(result["score"])
            bucket["votes"] += 1

        if not aggregated:
            return None

        best = max(
            aggregated.values(),
            key=lambda item: (
                item["votes"],
                item["score_sum"],
                cls.PATTERN_PRIORITY.get(str(item["pattern_type"]), 0),
            ),
        )
        return {
            "pattern_type": best["pattern_type"],
            "pattern_size": best["pattern_size"],
            "score": float(best["score_sum"]),
            "votes": int(best["votes"]),
        }

    @classmethod
    def infer_pattern_spec(
        cls,
        image: np.ndarray,
        candidate_types: Optional[Sequence[str]] = None,
        size_min: int = 3,
        size_max: int = 20,
    ) -> Optional[Dict[str, object]]:
        if candidate_types is None:
            pattern_types = ["checkerboard", "circles", "acircles", "volume_dots"]
        else:
            pattern_types = [
                pattern_type
                for pattern_type in candidate_types
                if pattern_type in cls.SUPPORTED_PATTERN_TYPES
            ]

        if not pattern_types:
            return None

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        best_result = None

        for pattern_type in pattern_types:
            if pattern_type == "volume_dots":
                result = cls._infer_volume_dot_spec(gray, size_min, size_max)
            else:
                result = cls._infer_standard_pattern_spec(
                    gray, pattern_type, size_min, size_max
                )

            if result is None:
                continue

            if (
                best_result is None
                or result["score"] > best_result["score"]
                or (
                    result["score"] == best_result["score"]
                    and cls.PATTERN_PRIORITY.get(result["pattern_type"], 0)
                    > cls.PATTERN_PRIORITY.get(best_result["pattern_type"], 0)
                )
            ):
                best_result = result

        if best_result is None:
            return None

        pattern_w, pattern_h = best_result["pattern_size"]
        image_h, image_w = gray.shape[:2]
        if image_w >= image_h and pattern_w < pattern_h:
            best_result["pattern_size"] = (pattern_h, pattern_w)
        elif image_h > image_w and pattern_h < pattern_w:
            best_result["pattern_size"] = (pattern_h, pattern_w)

        return best_result

    @classmethod
    def _infer_standard_pattern_spec(
        cls,
        gray: np.ndarray,
        pattern_type: str,
        size_min: int,
        size_max: int,
    ) -> Optional[Dict[str, object]]:
        best_result = None

        for width in range(size_min, size_max + 1):
            for height in range(size_min, size_max + 1):
                calibrator = cls(
                    pattern_type=pattern_type,
                    pattern_size=(width, height),
                    square_size=1.0,
                )
                points = calibrator._detect_standard_pattern(gray)
                if points is None:
                    continue

                point_count = width * height
                score = float(point_count * 10 + cls.PATTERN_PRIORITY.get(pattern_type, 0))
                if best_result is None or score > best_result["score"]:
                    best_result = {
                        "pattern_type": pattern_type,
                        "pattern_size": (width, height),
                        "score": score,
                    }

        if best_result is None and pattern_type in {"circles", "acircles"}:
            best_result = cls._infer_blob_grid_spec(
                gray,
                pattern_type=pattern_type,
                size_min=size_min,
                size_max=size_max,
            )

        return best_result

    @classmethod
    def _infer_blob_grid_spec(
        cls,
        gray: np.ndarray,
        pattern_type: str,
        size_min: int,
        size_max: int,
    ) -> Optional[Dict[str, object]]:
        helper = cls(pattern_type="volume_dots", pattern_size=(11, 8), square_size=1.0)
        centers = helper._extract_round_blob_centers(gray)
        if len(centers) < 8:
            return None

        centered = centers - centers.mean(axis=0)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        axis_candidates = [
            (vh[0], vh[1]),
            (np.array([1.0, 0.0], dtype=np.float32), np.array([0.0, 1.0], dtype=np.float32)),
            (np.array([0.0, 1.0], dtype=np.float32), np.array([1.0, 0.0], dtype=np.float32)),
        ]

        best_result = None
        rough_expected = int(np.clip(round(np.sqrt(len(centers))), size_min, size_max))

        for axis_u, axis_v in axis_candidates:
            proj_u = centers @ axis_u
            proj_v = centers @ axis_v

            spacing_u = helper._estimate_axis_spacing(proj_u, rough_expected)
            spacing_v = helper._estimate_axis_spacing(proj_v, rough_expected)
            if spacing_u <= 0 or spacing_v <= 0:
                continue

            row_idx = np.round((proj_v - np.min(proj_v)) / spacing_v).astype(int)
            unique_rows = np.unique(row_idx)
            if len(unique_rows) < size_min:
                continue

            row_counts = []
            row_offsets = []
            for row_value in unique_rows:
                row_points = proj_u[row_idx == row_value]
                row_counts.append(len(row_points))
                row_offsets.append(float(np.median(row_points)))

            width = int(np.clip(round(float(np.median(row_counts))), size_min, size_max))
            height = int(np.clip(len(unique_rows), size_min, size_max))
            if width < size_min or height < size_min:
                continue

            estimated_height = int(np.clip(round(len(centers) / max(width, 1)), size_min, size_max))
            if abs(estimated_height - height) <= 1:
                height = estimated_height

            normalized_offsets = (
                (np.array(row_offsets, dtype=np.float32) - np.min(row_offsets))
                / max(spacing_u, 1e-6)
            )
            fractional = np.mod(normalized_offsets, 1.0)
            half_phase = np.mod(np.round(fractional * 2.0), 2.0)

            if pattern_type == "circles":
                consistency = float(np.std(np.minimum(fractional, 1.0 - fractional)))
                if consistency > 0.22:
                    continue
                quality = 1.0 - consistency
            else:
                if len(half_phase) < 2:
                    continue
                alternating = float(np.mean(np.abs(np.diff(half_phase)) > 0.5))
                if alternating < 0.3:
                    continue
                quality = alternating

            point_count = width * height
            score = float(point_count * 8 + quality + cls.PATTERN_PRIORITY.get(pattern_type, 0))
            result = {
                "pattern_type": pattern_type,
                "pattern_size": (width, height),
                "score": score,
            }
            if best_result is None or result["score"] > best_result["score"]:
                best_result = result

        return best_result

    @classmethod
    def _infer_volume_dot_spec(
        cls,
        gray: np.ndarray,
        size_min: int,
        size_max: int,
    ) -> Optional[Dict[str, object]]:
        helper = cls(pattern_type="volume_dots", pattern_size=(11, 8), square_size=1.0)
        centers = helper._extract_round_blob_centers(gray)
        if len(centers) < 8:
            return None

        centered = centers - centers.mean(axis=0)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        axis_u = vh[0]
        axis_v = vh[1]
        proj_u = centers @ axis_u
        proj_v = centers @ axis_v

        rough_expected = int(np.clip(round(np.sqrt(len(centers))), size_min, size_max))
        spacing_u = helper._estimate_axis_spacing(proj_u, rough_expected)
        spacing_v = helper._estimate_axis_spacing(proj_v, rough_expected)
        if spacing_u <= 0 or spacing_v <= 0:
            return None

        estimate_w = int(
            np.clip(round((float(np.max(proj_u)) - float(np.min(proj_u))) / spacing_u) + 1, size_min, size_max)
        )
        estimate_h = int(
            np.clip(round((float(np.max(proj_v)) - float(np.min(proj_v))) / spacing_v) + 1, size_min, size_max)
        )

        width_candidates = sorted(
            {
                int(np.clip(estimate_w + delta, size_min, size_max))
                for delta in range(-3, 4)
            }
        )
        height_candidates = sorted(
            {
                int(np.clip(estimate_h + delta, size_min, size_max))
                for delta in range(-3, 4)
            }
        )

        best_result = None
        for width in width_candidates:
            for height in height_candidates:
                calibrator = cls(
                    pattern_type="volume_dots",
                    pattern_size=(width, height),
                    square_size=1.0,
                )
                observation = calibrator._detect_volume_dot_target(gray)
                if observation is None:
                    continue

                detected = len(observation.point_ids)
                expected = width * height
                coverage = detected / max(expected, 1)
                score = detected * 10 * coverage + 0.1 * cls.PATTERN_PRIORITY["volume_dots"]

                if best_result is None or score > best_result["score"]:
                    best_result = {
                        "pattern_type": "volume_dots",
                        "pattern_size": (width, height),
                        "score": float(score),
                        "coverage": float(coverage),
                    }

        if best_result is not None and best_result.get("coverage", 0.0) >= 0.98:
            best_result["pattern_type"] = "circles"

        return best_result

    def _grid_id_to_object_point(self, point_id: PointId) -> np.ndarray:
        if self.pattern_type == "checkerboard":
            x_idx, y_idx = point_id
            return np.array(
                [x_idx * self.square_size, y_idx * self.square_size, 0.0],
                dtype=np.float32,
            )

        if self.pattern_type == "circles":
            x_idx, y_idx = point_id
            spacing = 2.0 * self.circle_radius + 1.0
            return np.array([x_idx * spacing, y_idx * spacing, 0.0], dtype=np.float32)

        if self.pattern_type == "acircles":
            x_idx, y_idx = point_id
            spacing = 2.0 * self.circle_radius + 1.0
            x = (2.0 * x_idx + (y_idx % 2)) * spacing / 2.0
            y = y_idx * spacing
            return np.array([x, y, 0.0], dtype=np.float32)

        if self.pattern_type == "volume_dots":
            if len(point_id) == 3:
                layer_idx, x_sign, y_sign = point_id
                return np.array(
                    [
                        0.5 * float(x_sign) * self.square_size,
                        0.5 * float(y_sign) * self.square_size,
                        float(layer_idx) * self.level_separation,
                    ],
                    dtype=np.float32,
                )
            x_idx, y_idx = point_id
            return np.array(
                [x_idx * self.square_size, y_idx * self.square_size, 0.0],
                dtype=np.float32,
            )

        raise ValueError(f"Unsupported pattern type: {self.pattern_type}")

    def _object_points_from_ids(self, point_ids: Sequence[PointId]) -> np.ndarray:
        return np.array(
            [self._grid_id_to_object_point(point_id) for point_id in point_ids],
            dtype=np.float32,
        )

    def _generate_object_points(self) -> np.ndarray:
        return self._object_points_from_ids(self._generate_grid_point_ids())

    def detect_pattern(self, image: np.ndarray) -> Optional[np.ndarray]:
        observation = self.detect_pattern_observation(image)
        return None if observation is None else observation.image_points

    def detect_pattern_observation(
        self, image: np.ndarray
    ) -> Optional[PatternObservation]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if self.pattern_type in {"checkerboard", "circles", "acircles"}:
            image_points = self._detect_standard_pattern(gray)
            if image_points is None:
                return None
            point_ids = self._generate_grid_point_ids()
            return PatternObservation(
                image_points=image_points,
                object_points=self.obj_points.copy(),
                point_ids=point_ids,
            )

        if self.pattern_type == "volume_dots":
            return self._detect_volume_dot_target(gray)

        return None

    def _detect_standard_pattern(self, gray: np.ndarray) -> Optional[np.ndarray]:
        w, h = self.pattern_size

        if self.pattern_type == "checkerboard":
            flags = (
                cv2.CALIB_CB_ADAPTIVE_THRESH
                | cv2.CALIB_CB_NORMALIZE_IMAGE
                | cv2.CALIB_CB_FAST_CHECK
            )
            ret, corners = cv2.findChessboardCorners(gray, (w, h), flags)
            if not ret:
                return None

            criteria = (
                cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                100,
                1e-6,
            )
            return cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        if self.pattern_type == "circles":
            flags = cv2.CALIB_CB_SYMMETRIC_GRID | cv2.CALIB_CB_CLUSTERING
            ret, centers = cv2.findCirclesGrid(gray, (w, h), flags=flags)
            return centers if ret else None

        if self.pattern_type == "acircles":
            flags = cv2.CALIB_CB_ASYMMETRIC_GRID | cv2.CALIB_CB_CLUSTERING
            ret, centers = cv2.findCirclesGrid(gray, (w, h), flags=flags)
            return centers if ret else None

        return None

    def _detect_volume_dot_target(
        self, gray: np.ndarray
    ) -> Optional[PatternObservation]:
        # Fast path: if OpenCV can solve the grid directly, reuse it.
        w, h = self.pattern_size
        flags = cv2.CALIB_CB_SYMMETRIC_GRID | cv2.CALIB_CB_CLUSTERING
        ret, centers = cv2.findCirclesGrid(gray, (w, h), flags=flags)
        if ret:
            point_ids = self._generate_grid_point_ids()
            return PatternObservation(
                image_points=centers,
                object_points=self.obj_points.copy(),
                point_ids=point_ids,
            )

        blob_centers = self._extract_round_blob_centers(gray)
        if len(blob_centers) < 8:
            return None

        layered_observation = self._detect_two_level_fiducial_observation(
            gray, blob_centers
        )
        if layered_observation is not None:
            return layered_observation

        if len(blob_centers) < max(8, min(w * h // 3, 20)):
            return None

        grid_mapping = self._assign_volume_dot_indices(blob_centers)
        if not grid_mapping:
            return None

        point_ids = sorted(grid_mapping.values(), key=self._point_sort_key)
        image_points = np.array(
            [blob_centers[idx] for idx, point_id in grid_mapping.items()],
            dtype=np.float32,
        )
        point_ids = [grid_mapping[idx] for idx in grid_mapping]

        order = sorted(range(len(point_ids)), key=lambda idx: self._point_sort_key(point_ids[idx]))
        point_ids = [point_ids[i] for i in order]
        image_points = image_points[order].reshape(-1, 1, 2)

        object_points = self._object_points_from_ids(point_ids)

        min_required = max(8, min(20, self.pattern_size[0] + self.pattern_size[1]))
        if len(point_ids) < min_required:
            return None

        return PatternObservation(
            image_points=image_points,
            object_points=object_points,
            point_ids=point_ids,
        )

    def _extract_round_blob_centers(self, gray: np.ndarray) -> np.ndarray:
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        expected_count = self.pattern_size[0] * self.pattern_size[1]
        candidates = []

        for thresh_flag in (cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV):
            _, mask = cv2.threshold(
                blurred, 0, 255, thresh_flag | cv2.THRESH_OTSU
            )
            mask = cv2.morphologyEx(
                mask,
                cv2.MORPH_OPEN,
                np.ones((3, 3), np.uint8),
                iterations=1,
            )
            centers, stats = self._collect_round_blobs(mask)
            if len(centers) == 0:
                continue

            score = abs(len(centers) - expected_count)
            score -= 0.1 * min(len(centers), expected_count)
            candidates.append((score, centers, stats))

        if not candidates:
            return np.empty((0, 2), dtype=np.float32)

        _, centers, _ = min(candidates, key=lambda item: item[0])
        return centers

    def _collect_round_blobs(
        self, mask: np.ndarray
    ) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        height, width = mask.shape[:2]
        blob_rows = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 10:
                continue

            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 0:
                continue

            circularity = 4.0 * np.pi * area / (perimeter * perimeter)
            if circularity < 0.45:
                continue

            (x, y), radius = cv2.minEnclosingCircle(contour)
            if radius < 1.5:
                continue
            if x <= radius or y <= radius or x >= width - radius or y >= height - radius:
                continue

            moments = cv2.moments(contour)
            if abs(moments["m00"]) < 1e-6:
                continue

            cx = moments["m10"] / moments["m00"]
            cy = moments["m01"] / moments["m00"]
            blob_rows.append((cx, cy, area, circularity))

        if not blob_rows:
            return np.empty((0, 2), dtype=np.float32), []

        areas = np.array([row[2] for row in blob_rows], dtype=np.float32)
        circularities = np.array([row[3] for row in blob_rows], dtype=np.float32)
        median_area = float(np.median(areas))

        filtered = []
        for cx, cy, area, circularity in blob_rows:
            if area < 0.35 * median_area or area > 2.8 * median_area:
                continue
            if circularity < max(0.55, float(np.median(circularities)) * 0.8):
                continue
            filtered.append((cx, cy, area, circularity))

        if not filtered:
            filtered = blob_rows

        centers = np.array([[row[0], row[1]] for row in filtered], dtype=np.float32)
        stats = [(row[2], row[3]) for row in filtered]
        return centers, stats

    def _detect_two_level_fiducial_observation(
        self, gray: np.ndarray, blob_centers: np.ndarray
    ) -> Optional[PatternObservation]:
        markers = self._detect_square_triangle_markers(gray, blob_centers)
        if not {"square", "triangle"}.issubset({marker["shape"] for marker in markers}):
            return None

        axis_u, axis_v, spacing = self._estimate_volume_axes(blob_centers)
        if spacing <= 0:
            return None

        layer_by_shape = {"square": 0.0, "triangle": 1.0}
        point_rows = []
        used_shapes = set()

        for marker in sorted(markers, key=lambda item: layer_by_shape.get(item["shape"], 99)):
            shape = marker["shape"]
            if shape in used_shapes or shape not in layer_by_shape:
                continue

            surrounding = self._find_marker_surrounding_dots(
                marker["center"], blob_centers, axis_u, axis_v, spacing
            )
            if surrounding is None:
                continue

            used_shapes.add(shape)
            layer_idx = layer_by_shape[shape]
            for x_sign, y_sign, center_index in surrounding:
                point_rows.append(
                    (
                        (layer_idx, float(x_sign), float(y_sign)),
                        blob_centers[center_index],
                    )
                )

        if len(point_rows) < 8 or not {row[0][0] for row in point_rows}.issuperset({0.0, 1.0}):
            return None

        point_rows.sort(key=lambda row: (row[0][0], row[0][2], row[0][1]))
        point_ids = [row[0] for row in point_rows]
        image_points = np.array([row[1] for row in point_rows], dtype=np.float32).reshape(-1, 1, 2)
        object_points = self._object_points_from_ids(point_ids)

        return PatternObservation(
            image_points=image_points,
            object_points=object_points,
            point_ids=point_ids,
        )

    def _detect_square_triangle_markers(
        self, gray: np.ndarray, blob_centers: np.ndarray
    ) -> List[dict]:
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        nearest_dist = self._median_nearest_distance(blob_centers)
        if nearest_dist <= 0:
            nearest_dist = max(gray.shape[:2]) / 30.0

        markers = []
        for thresh_flag in (cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV):
            _, mask = cv2.threshold(
                blurred, 0, 255, thresh_flag | cv2.THRESH_OTSU
            )
            mask = cv2.morphologyEx(
                mask,
                cv2.MORPH_OPEN,
                np.ones((3, 3), np.uint8),
                iterations=1,
            )
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 12:
                    continue

                perimeter = cv2.arcLength(contour, True)
                if perimeter <= 0:
                    continue

                circularity = 4.0 * np.pi * area / (perimeter * perimeter)
                if circularity > 0.9:
                    continue

                approx = cv2.approxPolyDP(contour, 0.045 * perimeter, True)
                vertices = len(approx)
                marker_shape = None

                if vertices == 3:
                    marker_shape = "triangle"
                elif vertices == 4 and cv2.isContourConvex(approx):
                    rect = cv2.minAreaRect(contour)
                    side_a, side_b = rect[1]
                    if min(side_a, side_b) <= 0:
                        continue
                    aspect = max(side_a, side_b) / min(side_a, side_b)
                    if aspect <= 1.45:
                        marker_shape = "square"

                if marker_shape is None:
                    continue

                moments = cv2.moments(contour)
                if abs(moments["m00"]) < 1e-6:
                    continue

                center = np.array(
                    [moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]],
                    dtype=np.float32,
                )

                markers.append(
                    {
                        "shape": marker_shape,
                        "center": center,
                        "area": float(area),
                    }
                )

        deduped = []
        for marker in sorted(markers, key=lambda item: item["area"], reverse=True):
            duplicate = False
            for existing in deduped:
                if (
                    existing["shape"] == marker["shape"]
                    and np.linalg.norm(existing["center"] - marker["center"]) < 0.3 * nearest_dist
                ):
                    duplicate = True
                    break
            if not duplicate:
                deduped.append(marker)

        return deduped

    def _estimate_volume_axes(
        self, centers: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        centered = centers - centers.mean(axis=0)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        axis_u = vh[0].astype(np.float32)
        axis_v = vh[1].astype(np.float32)
        spacing = self._median_nearest_distance(centers)
        return axis_u, axis_v, spacing

    def _median_nearest_distance(self, centers: np.ndarray) -> float:
        if len(centers) < 2:
            return 0.0
        pairwise = centers[:, None, :] - centers[None, :, :]
        distances = np.linalg.norm(pairwise, axis=2)
        np.fill_diagonal(distances, np.inf)
        nearest = np.min(distances, axis=1)
        nearest = nearest[np.isfinite(nearest)]
        if len(nearest) == 0:
            return 0.0
        return float(np.median(nearest))

    def _find_marker_surrounding_dots(
        self,
        marker_center: np.ndarray,
        blob_centers: np.ndarray,
        axis_u: np.ndarray,
        axis_v: np.ndarray,
        spacing: float,
    ) -> Optional[List[Tuple[int, int, int]]]:
        axis_u = axis_u / max(np.linalg.norm(axis_u), 1e-6)
        axis_v = axis_v / max(np.linalg.norm(axis_v), 1e-6)

        offsets = blob_centers - marker_center
        proj_u = offsets @ axis_u
        proj_v = offsets @ axis_v
        distances = np.linalg.norm(offsets, axis=1)

        lower = 0.35 * spacing
        upper = 1.35 * spacing
        quadrant_best: Dict[Tuple[int, int], Tuple[int, float]] = {}

        for index, distance in enumerate(distances):
            if distance < lower or distance > upper:
                continue
            if abs(proj_u[index]) < 0.15 * spacing or abs(proj_v[index]) < 0.15 * spacing:
                continue

            x_sign = 1 if proj_u[index] >= 0 else -1
            y_sign = 1 if proj_v[index] >= 0 else -1
            key = (x_sign, y_sign)
            target = spacing / np.sqrt(2.0)
            score = abs(float(distance) - target)

            current = quadrant_best.get(key)
            if current is None or score < current[1]:
                quadrant_best[key] = (int(index), score)

        expected_quadrants = [(-1, -1), (1, -1), (-1, 1), (1, 1)]
        if not all(key in quadrant_best for key in expected_quadrants):
            return None

        selected_distances = np.array(
            [distances[quadrant_best[key][0]] for key in expected_quadrants],
            dtype=np.float32,
        )
        if float(np.std(selected_distances)) > 0.35 * float(np.mean(selected_distances)):
            return None

        return [
            (key[0], key[1], quadrant_best[key][0])
            for key in expected_quadrants
        ]

    def _assign_volume_dot_indices(
        self, centers: np.ndarray
    ) -> Dict[int, PointId]:
        if len(centers) < 6:
            return {}

        mean_center = centers.mean(axis=0)
        centered = centers - mean_center
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        base_axes = [vh[0], vh[1]]

        pairwise = centers[:, None, :] - centers[None, :, :]
        distances = np.linalg.norm(pairwise, axis=2)
        np.fill_diagonal(distances, np.inf)
        nearest_dist = float(np.median(np.min(distances, axis=1)))
        if not np.isfinite(nearest_dist) or nearest_dist <= 0:
            return {}

        best_mapping: Dict[int, PointId] = {}
        best_score = -1e9

        axis_orders = [
            (base_axes[0], base_axes[1]),
            (base_axes[1], base_axes[0]),
        ]

        for axis_u, axis_v in axis_orders:
            for sign_u in (-1.0, 1.0):
                for sign_v in (-1.0, 1.0):
                    mapping, conflicts = self._trace_grid_graph(
                        centers,
                        axis_u * sign_u,
                        axis_v * sign_v,
                        nearest_dist,
                    )
                    projection_mapping = self._assign_by_axis_projection(
                        centers,
                        axis_u * sign_u,
                        axis_v * sign_v,
                    )

                    for candidate_mapping in (mapping, projection_mapping):
                        if len(candidate_mapping) < 6:
                            continue

                        normalized = self._normalize_and_filter_grid_mapping(candidate_mapping)
                        if len(normalized) < 6:
                            continue

                        refined = self._refine_grid_mapping(centers, normalized)
                        if len(refined) < 6:
                            continue

                        coords = np.array(list(refined.values()), dtype=np.int32)
                        span_x = int(coords[:, 0].max() - coords[:, 0].min() + 1)
                        span_y = int(coords[:, 1].max() - coords[:, 1].min() + 1)

                        score = (
                            len(refined) * 10
                            - conflicts * 2
                            - abs(span_x - self.pattern_size[0]) * 2
                            - abs(span_y - self.pattern_size[1]) * 2
                        )

                        if score > best_score:
                            best_score = score
                            best_mapping = refined

        return best_mapping

    def _assign_by_axis_projection(
        self,
        centers: np.ndarray,
        axis_u: np.ndarray,
        axis_v: np.ndarray,
    ) -> Dict[int, PointId]:
        axis_u = axis_u / np.linalg.norm(axis_u)
        axis_v = axis_v / np.linalg.norm(axis_v)

        proj_u = centers @ axis_u
        proj_v = centers @ axis_v
        spacing_u = self._estimate_axis_spacing(proj_u, self.pattern_size[0])
        spacing_v = self._estimate_axis_spacing(proj_v, self.pattern_size[1])
        if spacing_u <= 0 or spacing_v <= 0:
            return {}

        origin_u = float(np.min(proj_u))
        origin_v = float(np.min(proj_v))
        mapping = self._projective_quantize(
            centers,
            proj_u,
            proj_v,
            origin_u,
            origin_v,
            spacing_u,
            spacing_v,
        )
        if len(mapping) < 6:
            return mapping

        used_proj_u = []
        used_proj_v = []
        for point_index, (grid_x, grid_y) in mapping.items():
            used_proj_u.append(proj_u[point_index] - grid_x * spacing_u)
            used_proj_v.append(proj_v[point_index] - grid_y * spacing_v)

        refined_origin_u = float(np.median(used_proj_u))
        refined_origin_v = float(np.median(used_proj_v))
        refined_mapping = self._projective_quantize(
            centers,
            proj_u,
            proj_v,
            refined_origin_u,
            refined_origin_v,
            spacing_u,
            spacing_v,
        )

        return refined_mapping if len(refined_mapping) >= len(mapping) else mapping

    def _estimate_axis_spacing(self, values: np.ndarray, expected_count: int) -> float:
        if expected_count <= 1:
            return 1.0

        value_range = float(np.max(values) - np.min(values))
        rough_spacing = value_range / max(expected_count - 1, 1)
        if rough_spacing <= 0:
            return 0.0

        sorted_values = np.sort(values)
        diffs = np.diff(sorted_values)
        diffs = diffs[diffs > 1e-6]
        if len(diffs) == 0:
            return rough_spacing

        candidate = diffs[
            (diffs > 0.35 * rough_spacing) & (diffs < 1.8 * rough_spacing)
        ]
        if len(candidate) < max(3, expected_count // 3):
            candidate = diffs[(diffs > 0.2 * rough_spacing) & (diffs < 2.5 * rough_spacing)]

        if len(candidate) == 0:
            return rough_spacing

        return float(np.median(candidate))

    def _projective_quantize(
        self,
        centers: np.ndarray,
        proj_u: np.ndarray,
        proj_v: np.ndarray,
        origin_u: float,
        origin_v: float,
        spacing_u: float,
        spacing_v: float,
    ) -> Dict[int, PointId]:
        max_x, max_y = self.pattern_size[0] - 1, self.pattern_size[1] - 1
        candidates: Dict[PointId, Tuple[int, float]] = {}

        for point_index in range(len(centers)):
            grid_x = int(np.round((proj_u[point_index] - origin_u) / spacing_u))
            grid_y = int(np.round((proj_v[point_index] - origin_v) / spacing_v))
            if grid_x < 0 or grid_y < 0 or grid_x > max_x or grid_y > max_y:
                continue

            err_u = abs(proj_u[point_index] - (origin_u + grid_x * spacing_u)) / spacing_u
            err_v = abs(proj_v[point_index] - (origin_v + grid_y * spacing_v)) / spacing_v
            total_error = float(err_u + err_v)
            if total_error > 1.1:
                continue

            key = (grid_x, grid_y)
            current = candidates.get(key)
            if current is None or total_error < current[1]:
                candidates[key] = (point_index, total_error)

        return {point_index: grid_id for grid_id, (point_index, _) in candidates.items()}

    def _trace_grid_graph(
        self,
        centers: np.ndarray,
        axis_u: np.ndarray,
        axis_v: np.ndarray,
        nearest_dist: float,
    ) -> Tuple[Dict[int, PointId], int]:
        axis_u = axis_u / np.linalg.norm(axis_u)
        axis_v = axis_v / np.linalg.norm(axis_v)

        adjacency: Dict[int, List[Tuple[int, PointId]]] = {}
        for idx in range(len(centers)):
            adjacency[idx] = []
            for direction, delta in (
                (axis_u, (1, 0)),
                (-axis_u, (-1, 0)),
                (axis_v, (0, 1)),
                (-axis_v, (0, -1)),
            ):
                neighbor = self._find_directed_neighbor(
                    idx, centers, direction, nearest_dist
                )
                if neighbor is not None:
                    adjacency[idx].append((neighbor, delta))

        projections_u = centers @ axis_u
        projections_v = centers @ axis_v
        origin = int(np.argmin(projections_u + projections_v))

        coords: Dict[int, PointId] = {origin: (0, 0)}
        queue: deque[int] = deque([origin])
        conflicts = 0

        while queue:
            current = queue.popleft()
            base_x, base_y = coords[current]

            for neighbor, delta in adjacency[current]:
                candidate = (base_x + delta[0], base_y + delta[1])
                if neighbor not in coords:
                    coords[neighbor] = candidate
                    queue.append(neighbor)
                elif coords[neighbor] != candidate:
                    conflicts += 1

        return coords, conflicts

    def _find_directed_neighbor(
        self,
        index: int,
        centers: np.ndarray,
        direction: np.ndarray,
        nearest_dist: float,
    ) -> Optional[int]:
        base = centers[index]
        diffs = centers - base
        dist = np.linalg.norm(diffs, axis=1)
        valid = (dist > 0.35 * nearest_dist) & (dist < 2.2 * nearest_dist)
        valid[index] = False
        if not np.any(valid):
            return None

        direction = direction / np.linalg.norm(direction)
        normal = np.array([-direction[1], direction[0]], dtype=np.float32)

        best_neighbor = None
        best_score = None
        for neighbor_idx in np.where(valid)[0]:
            vec = diffs[neighbor_idx]
            along = float(np.dot(vec, direction))
            if along <= 0:
                continue

            perp = abs(float(np.dot(vec, normal)))
            if perp > max(0.8 * along, 0.6 * nearest_dist):
                continue

            score = perp / max(along, 1e-6) + 0.15 * abs(dist[neighbor_idx] - nearest_dist)
            if best_score is None or score < best_score:
                best_score = score
                best_neighbor = int(neighbor_idx)

        return best_neighbor

    def _normalize_and_filter_grid_mapping(
        self, mapping: Dict[int, PointId]
    ) -> Dict[int, PointId]:
        if not mapping:
            return {}

        coords = np.array(list(mapping.values()), dtype=np.int32)
        min_x = int(coords[:, 0].min())
        min_y = int(coords[:, 1].min())

        normalized: Dict[int, PointId] = {}
        used = set()
        max_x, max_y = self.pattern_size[0] - 1, self.pattern_size[1] - 1

        for point_index, (x, y) in mapping.items():
            coord = (x - min_x, y - min_y)
            if coord[0] < 0 or coord[1] < 0 or coord[0] > max_x or coord[1] > max_y:
                continue
            if coord in used:
                continue
            normalized[point_index] = coord
            used.add(coord)

        return normalized

    def _refine_grid_mapping(
        self, centers: np.ndarray, mapping: Dict[int, PointId]
    ) -> Dict[int, PointId]:
        if len(mapping) < 6:
            return mapping

        ordered_indices = list(mapping.keys())
        grid = np.array([mapping[idx] for idx in ordered_indices], dtype=np.float32)
        image = centers[ordered_indices].astype(np.float32)

        design = np.hstack([grid, np.ones((len(grid), 1), dtype=np.float32)])
        affine, _, _, _ = np.linalg.lstsq(design, image, rcond=None)
        predicted = design @ affine
        residual = np.linalg.norm(predicted - image, axis=1)

        keep = residual <= max(3.0, 0.45 * float(np.median(residual) + 1.0))
        refined = {
            ordered_indices[i]: mapping[ordered_indices[i]]
            for i in range(len(ordered_indices))
            if keep[i]
        }

        return refined if len(refined) >= 6 else mapping

    def _draw_detected_points(
        self, image: np.ndarray, observation: PatternObservation
    ) -> np.ndarray:
        vis = image.copy()
        points = observation.image_points.reshape(-1, 2)

        for point, point_id in zip(points, observation.point_ids):
            px = tuple(np.round(point).astype(int))
            cv2.circle(vis, px, 5, (0, 255, 0), 1)
            cv2.putText(
                vis,
                ",".join(f"{value:g}" for value in point_id),
                (px[0] + 4, px[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return vis

    def calibrate_camera(
        self,
        camera_id: str,
        image_paths: List[str],
        show_detection: bool = False,
    ) -> CameraParams:
        obj_points_list = []
        img_points_list = []
        detected_count = 0
        image_size = None

        for img_path in image_paths:
            img = cv2.imread(img_path)
            if img is None:
                logger.warning("Failed to read image: %s", img_path)
                continue

            if image_size is None:
                image_size = (img.shape[1], img.shape[0])

            observation = self.detect_pattern_observation(img)
            if observation is None:
                continue

            obj_points_list.append(observation.object_points.copy())
            img_points_list.append(observation.image_points.copy())
            detected_count += 1

            if show_detection:
                vis = self._draw_detected_points(img, observation)
                cv2.imshow(f"Camera {camera_id} - {Path(img_path).name}", vis)
                cv2.waitKey(500)

        if show_detection:
            cv2.destroyAllWindows()

        if image_size is None:
            raise ValueError(f"Camera {camera_id}: no readable calibration images")

        if detected_count < 3:
            raise ValueError(
                f"Camera {camera_id}: only {detected_count}/{len(image_paths)} "
                "images produced valid calibration points"
            )

        logger.info(
            "Camera %s: detected calibration target in %d/%d images",
            camera_id,
            detected_count,
            len(image_paths),
        )

        rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            obj_points_list, img_points_list, image_size, None, None
        )

        rvec_avg = np.mean(np.array([r.flatten() for r in rvecs]), axis=0)
        tvec_avg = np.mean(np.array([t.flatten() for t in tvecs]), axis=0)

        params = CameraParams(
            camera_id=camera_id,
            image_size=list(image_size),
            camera_matrix=camera_matrix.tolist(),
            dist_coeffs=dist_coeffs.flatten().tolist(),
            rvec=rvec_avg.tolist(),
            tvec=tvec_avg.tolist(),
            rms_error=float(rms),
        )

        self.camera_params[camera_id] = params
        self._calib_data[camera_id] = {
            "obj_points": obj_points_list,
            "img_points": img_points_list,
            "rvecs": rvecs,
            "tvecs": tvecs,
        }

        logger.info(
            "Camera %s calibration finished, RMS reprojection error %.4f px",
            camera_id,
            rms,
        )
        return params

    def calibrate_multi_camera(
        self,
        camera_images: Dict[str, List[str]],
        show_detection: bool = False,
    ) -> Dict[str, CameraParams]:
        if len(camera_images) < 3:
            raise ValueError("At least 3 cameras are required for multi-camera calibration")

        results = {}
        for cam_id, img_paths in camera_images.items():
            logger.info("Calibrating camera %s with %d images", cam_id, len(img_paths))
            results[cam_id] = self.calibrate_camera(cam_id, img_paths, show_detection)

        return results

    def stereo_calibrate_pair(
        self,
        cam1_id: str,
        cam2_id: str,
        image_pairs: List[Tuple[str, str]],
        stereo_criteria: float = 1e-6,
    ) -> Dict:
        cam1 = self.camera_params.get(cam1_id)
        cam2 = self.camera_params.get(cam2_id)
        if cam1 is None or cam2 is None:
            raise ValueError("Please calibrate both cameras before stereo calibration")

        obj_pts = []
        img1_pts = []
        img2_pts = []

        for img1_path, img2_path in image_pairs:
            img1 = cv2.imread(img1_path)
            img2 = cv2.imread(img2_path)
            if img1 is None or img2 is None:
                continue

            obs1 = self.detect_pattern_observation(img1)
            obs2 = self.detect_pattern_observation(img2)
            if obs1 is None or obs2 is None:
                continue

            common_ids = sorted(
                set(obs1.point_ids) & set(obs2.point_ids),
                key=self._point_sort_key,
            )
            if len(common_ids) < 5:
                continue

            obs1_map = {
                point_id: point
                for point_id, point in zip(obs1.point_ids, obs1.image_points.reshape(-1, 2))
            }
            obs2_map = {
                point_id: point
                for point_id, point in zip(obs2.point_ids, obs2.image_points.reshape(-1, 2))
            }

            obj_pts.append(self._object_points_from_ids(common_ids))
            img1_pts.append(
                np.array([obs1_map[point_id] for point_id in common_ids], dtype=np.float32).reshape(-1, 1, 2)
            )
            img2_pts.append(
                np.array([obs2_map[point_id] for point_id in common_ids], dtype=np.float32).reshape(-1, 1, 2)
            )

        if len(obj_pts) < 5:
            raise ValueError(
                f"Stereo calibration needs at least 5 valid image pairs, got {len(obj_pts)}"
            )

        k1 = np.array(cam1.camera_matrix)
        d1 = np.array(cam1.dist_coeffs)
        k2 = np.array(cam2.camera_matrix)
        d2 = np.array(cam2.dist_coeffs)
        image_size = tuple(cam1.image_size)

        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            100,
            stereo_criteria,
        )
        flags = cv2.CALIB_FIX_INTRINSIC

        rms, _, _, _, _, rmat, tvec, emat, fmat = cv2.stereoCalibrate(
            obj_pts,
            img1_pts,
            img2_pts,
            k1,
            d1,
            k2,
            d2,
            image_size,
            criteria=criteria,
            flags=flags,
        )

        result = {
            "camera_pair": f"{cam1_id}-{cam2_id}",
            "rotation_matrix": rmat.tolist(),
            "translation_vector": tvec.flatten().tolist(),
            "essential_matrix": emat.tolist(),
            "fundamental_matrix": fmat.tolist(),
            "rms_error": float(rms),
            "baseline_mm": float(np.linalg.norm(tvec)),
        }

        logger.info(
            "Stereo calibration %s-%s finished, RMS %.4f, baseline %.2f mm",
            cam1_id,
            cam2_id,
            rms,
            np.linalg.norm(tvec),
        )
        return result

    def compute_projection_matrix(
        self,
        camera_id: str,
        rvec: Optional[np.ndarray] = None,
        tvec: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        params = self.camera_params[camera_id]
        k_matrix = np.array(params.camera_matrix)

        if rvec is None:
            rvec = np.array(params.rvec)
        if tvec is None:
            tvec = np.array(params.tvec)

        rmat, _ = cv2.Rodrigues(rvec)
        rt = np.hstack([rmat, np.asarray(tvec).reshape(3, 1)])
        return k_matrix @ rt

    def save_results(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)

        for cam_id, params in self.camera_params.items():
            params.save(os.path.join(output_dir, f"{cam_id}_params.json"))

        summary = {
            "pattern_type": self.pattern_type,
            "pattern_size": list(self.pattern_size),
            "square_size": self.square_size,
            "circle_radius": self.circle_radius,
            "level_separation": self.level_separation,
            "num_cameras": len(self.camera_params),
            "cameras": {},
        }
        for cam_id, params in self.camera_params.items():
            summary["cameras"][cam_id] = {
                "image_size": params.image_size,
                "rms_error": params.rms_error,
                "focal_length": [
                    params.camera_matrix[0][0],
                    params.camera_matrix[1][1],
                ],
                "principal_point": [
                    params.camera_matrix[0][2],
                    params.camera_matrix[1][2],
                ],
            }

        with open(
            os.path.join(output_dir, "calibration_summary.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        logger.info("Calibration results saved to: %s", output_dir)

    @classmethod
    def load_results(cls, params_dir: str) -> "MultiCameraCalibrator":
        calib = cls()

        for path in glob.glob(os.path.join(params_dir, "*_params.json")):
            params = CameraParams.load(path)
            calib.camera_params[params.camera_id] = params

        summary_path = os.path.join(params_dir, "calibration_summary.json")
        if os.path.exists(summary_path):
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)

            calib.pattern_type = summary["pattern_type"]
            calib.pattern_size = tuple(summary["pattern_size"])
            calib.square_size = summary.get("square_size", 1.0)
            calib.circle_radius = summary.get("circle_radius", 0.5)
            calib.level_separation = summary.get(
                "level_separation",
                max(1.0, 0.2 * float(calib.square_size)),
            )
            calib.obj_points = calib._generate_object_points()

        logger.info("Loaded calibration results for %d cameras", len(calib.camera_params))
        return calib

    def get_calibration_report(self) -> str:
        lines = ["=" * 60, "Multi-camera calibration report", "=" * 60]
        lines.append(f"Pattern type: {self.pattern_type}")
        lines.append(f"Pattern size: {self.pattern_size}")
        lines.append(f"Camera count: {len(self.camera_params)}")
        lines.append("")

        for cam_id, params in self.camera_params.items():
            lines.append(f"--- Camera: {cam_id} ---")
            lines.append(f"  Image size: {params.image_size}")
            lines.append(
                f"  Focal length: fx={params.camera_matrix[0][0]:.2f}, "
                f"fy={params.camera_matrix[1][1]:.2f}"
            )
            lines.append(
                f"  Principal point: cx={params.camera_matrix[0][2]:.2f}, "
                f"cy={params.camera_matrix[1][2]:.2f}"
            )
            lines.append(f"  Distortion: {params.dist_coeffs}")
            lines.append(f"  RMS reprojection error: {params.rms_error:.4f} px")
            lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)
