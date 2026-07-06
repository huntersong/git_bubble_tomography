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
        origin_point_id: Optional[Sequence[float]] = None,
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
        self.origin_point_id = (
            tuple(origin_point_id) if origin_point_id is not None else None
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
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
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
    def detect_pattern_automatically(
        cls,
        image: np.ndarray,
        pattern_size_hint: Tuple[int, int] = (11, 8),
        pattern_type_hint: str = "checkerboard",
        square_size: float = 1.0,
        circle_radius: float = 0.5,
        level_separation: Optional[float] = None,
        size_min: int = 3,
        size_max: int = 15,
    ) -> Optional[Dict[str, object]]:
        """Detect any supported target, preferring the current GUI dimensions.

        The inexpensive first pass tries every target type with the configured
        grid size.  Only when that fails do we infer both type and dimensions.
        """
        if image.ndim == 2:
            source = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.ndim == 3 and image.shape[2] == 4:
            source = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        else:
            source = image

        # A LaVision target is substantially more expensive than the planar
        # detectors, so honor an explicit volume-target hint before probing
        # every 2-D family. If it fails, the generic fallback below remains.
        if pattern_type_hint == "volume_dots":
            volume_detector = cls(
                pattern_type="volume_dots",
                pattern_size=pattern_size_hint,
                square_size=square_size,
                circle_radius=circle_radius,
                level_separation=level_separation,
            )
            volume_observation = volume_detector.detect_pattern_observation(source)
            if volume_observation is not None:
                return {
                    "pattern_type": "volume_dots",
                    "pattern_size": tuple(pattern_size_hint),
                    "observation": volume_observation,
                    "inferred_size": False,
                }

        type_order = [pattern_type_hint] + [
            pattern_type
            for pattern_type in (
                "checkerboard",
                "circles",
                "acircles",
                "volume_dots",
            )
            if pattern_type != pattern_type_hint
        ]
        standard_types = [
            pattern_type
            for pattern_type in type_order
            if pattern_type in {"checkerboard", "circles", "acircles"}
        ]
        for pattern_type in standard_types:
            if pattern_type not in cls.SUPPORTED_PATTERN_TYPES:
                continue
            detector = cls(
                pattern_type=pattern_type,
                pattern_size=pattern_size_hint,
                square_size=square_size,
                circle_radius=circle_radius,
                level_separation=level_separation,
            )
            observation = detector.detect_pattern_observation(source)
            if observation is not None:
                return {
                    "pattern_type": pattern_type,
                    "pattern_size": tuple(pattern_size_hint),
                    "observation": observation,
                    "inferred_size": False,
                }

        gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
        blob_helper = cls(
            pattern_type="volume_dots",
            pattern_size=pattern_size_hint,
            square_size=square_size,
        )
        blob_count = len(blob_helper._extract_round_blob_centers(gray))
        factor_sizes = []
        for width in range(size_min, size_max + 1):
            if blob_count % width:
                continue
            height = blob_count // width
            if size_min <= height <= size_max:
                factor_sizes.append((width, height))

        # Test asymmetric grids first: a staggered grid can occasionally be
        # accepted as a rotated symmetric grid, while the reverse is rare.
        dot_type_order = ["acircles", "circles"]
        for pattern_type in dot_type_order:
            for pattern_size in factor_sizes:
                detector = cls(
                    pattern_type=pattern_type,
                    pattern_size=pattern_size,
                    square_size=square_size,
                    circle_radius=circle_radius,
                    level_separation=level_separation,
                )
                observation = detector.detect_pattern_observation(source)
                if observation is not None:
                    return {
                        "pattern_type": detector.pattern_type,
                        "pattern_size": detector.pattern_size,
                        "observation": observation,
                        "inferred_size": True,
                    }

        # A volume target may intentionally contain missing points, so its
        # detected count need not factor into the configured grid dimensions.
        volume_detector = cls(
            pattern_type="volume_dots",
            pattern_size=pattern_size_hint,
            square_size=square_size,
            circle_radius=circle_radius,
            level_separation=level_separation,
        )
        volume_observation = volume_detector.detect_pattern_observation(source)
        if volume_observation is not None:
            return {
                "pattern_type": volume_detector.pattern_type,
                "pattern_size": volume_detector.pattern_size,
                "observation": volume_observation,
                "inferred_size": False,
            }

        inferred = cls.infer_pattern_spec(
            source,
            candidate_types=["checkerboard"],
            size_min=size_min,
            size_max=size_max,
        )
        if inferred is not None:
            detector = cls(
                pattern_type="checkerboard",
                pattern_size=tuple(inferred["pattern_size"]),
                square_size=square_size,
                circle_radius=circle_radius,
                level_separation=level_separation,
            )
            observation = detector.detect_pattern_observation(source)
            if observation is not None:
                return {
                    "pattern_type": detector.pattern_type,
                    "pattern_size": detector.pattern_size,
                    "observation": observation,
                    "inferred_size": True,
                }

        volume_spec = cls.infer_pattern_spec(
            source,
            candidate_types=["volume_dots"],
            size_min=size_min,
            size_max=size_max,
        )
        if volume_spec is not None:
            detector = cls(
                pattern_type=str(volume_spec["pattern_type"]),
                pattern_size=tuple(volume_spec["pattern_size"]),
                square_size=square_size,
                circle_radius=circle_radius,
                level_separation=level_separation,
            )
            observation = detector.detect_pattern_observation(source)
            if observation is not None:
                return {
                    "pattern_type": detector.pattern_type,
                    "pattern_size": detector.pattern_size,
                    "observation": observation,
                    "inferred_size": True,
                }

        return None

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
            x = (2.0 * x_idx + (y_idx % 2)) * spacing
            y = y_idx * spacing
            return np.array([x, y, 0.0], dtype=np.float32)

        if self.pattern_type == "volume_dots":
            if len(point_id) == 3:
                layer_idx, x_idx, y_idx = point_id
                return np.array(
                    [
                        float(x_idx) * self.square_size,
                        float(y_idx) * self.square_size,
                        -float(layer_idx) * self.level_separation,
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
        points = np.array(
            [self._grid_id_to_object_point(point_id) for point_id in point_ids],
            dtype=np.float32,
        )
        if self.origin_point_id is not None:
            origin = self._grid_id_to_object_point(self.origin_point_id)
            points = points - origin.reshape(1, 3)
        return points

    def _generate_object_points(self) -> np.ndarray:
        return self._object_points_from_ids(self._generate_grid_point_ids())

    def detect_pattern(self, image: np.ndarray) -> Optional[np.ndarray]:
        observation = self.detect_pattern_observation(image)
        return None if observation is None else observation.image_points

    def detect_pattern_observation(
        self, image: np.ndarray
    ) -> Optional[PatternObservation]:
        gray = self._to_gray8(image)

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

    @staticmethod
    def _to_gray8(image: np.ndarray) -> np.ndarray:
        """Preserve useful contrast when calibration images are 12/16 bit TIFFs."""
        if image.ndim == 3:
            if image.shape[2] == 4:
                gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
            else:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        if gray.dtype == np.uint8:
            return gray

        finite = gray[np.isfinite(gray)]
        if finite.size == 0:
            return np.zeros(gray.shape, dtype=np.uint8)

        low, high = np.percentile(finite, (0.5, 99.5))
        if high <= low:
            low = float(np.min(finite))
            high = float(np.max(finite))
        if high <= low:
            return np.zeros(gray.shape, dtype=np.uint8)

        normalized = (gray.astype(np.float32) - float(low)) * (255.0 / (high - low))
        return np.clip(normalized, 0, 255).astype(np.uint8)

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
            for flags in (
                cv2.CALIB_CB_SYMMETRIC_GRID | cv2.CALIB_CB_CLUSTERING,
                cv2.CALIB_CB_SYMMETRIC_GRID,
            ):
                for source in (gray, cv2.bitwise_not(gray)):
                    ret, centers = cv2.findCirclesGrid(
                        source, (w, h), flags=flags
                    )
                    if ret:
                        return centers
            return None

        if self.pattern_type == "acircles":
            for flags in (
                cv2.CALIB_CB_ASYMMETRIC_GRID | cv2.CALIB_CB_CLUSTERING,
                cv2.CALIB_CB_ASYMMETRIC_GRID,
            ):
                for source in (gray, cv2.bitwise_not(gray)):
                    ret, centers = cv2.findCirclesGrid(
                        source, (w, h), flags=flags
                    )
                    if ret:
                        return centers
            return None

        return None

    def _detect_volume_dot_target(
        self, gray: np.ndarray
    ) -> Optional[PatternObservation]:
        lavision_observation = self._detect_lavision_double_layer_target(gray)
        if lavision_observation is not None:
            return lavision_observation
        # Do not reinterpret a recognized LaVision plate as an unrelated
        # single-plane grid when a clipped/blurred frame lacks enough columns.
        if self._extract_lavision_target_features(gray) is not None:
            return None

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

    def _detect_lavision_double_layer_target(
        self, gray: np.ndarray
    ) -> Optional[PatternObservation]:
        """Detect the interleaved lattices on LaVision 025-3.3 volume targets."""
        features = self._extract_lavision_target_features(gray)
        if features is None:
            return None

        circles, square_center, triangle_center = features
        assignments = self._assign_lavision_interleaved_columns(
            circles, square_center, triangle_center
        )
        if assignments is None:
            return None

        point_rows: List[Tuple[PointId, np.ndarray, float]] = [
            ((0.0, 0.0, 0.0), square_center, 0.0),
        ]

        for center_index, fine_x, fine_y, layer, residual in assignments:
            point_rows.append(
                (
                    (layer, 0.5 * fine_x, 0.5 * fine_y),
                    circles[center_index],
                    residual,
                )
            )

        # Keep only the best image candidate if thresholding produced a duplicate ID.
        best_by_id: Dict[PointId, Tuple[np.ndarray, float]] = {}
        for point_id, center, residual in point_rows:
            current = best_by_id.get(point_id)
            if current is None or residual < current[1]:
                best_by_id[point_id] = (center, residual)

        layer_counts = {
            layer: sum(point_id[0] == layer for point_id in best_by_id)
            for layer in (0.0, 1.0)
        }
        if min(layer_counts.values()) < 20:
            return None

        point_ids = sorted(best_by_id, key=self._point_sort_key)
        image_points = np.array(
            [best_by_id[point_id][0] for point_id in point_ids],
            dtype=np.float32,
        ).reshape(-1, 1, 2)
        return PatternObservation(
            image_points=image_points,
            object_points=self._object_points_from_ids(point_ids),
            point_ids=point_ids,
        )

    def _extract_lavision_target_features(
        self, gray: np.ndarray
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        min_dim = min(gray.shape[:2])
        kernel_size = max(11, int(round(min_dim * 0.026)))
        if kernel_size % 2 == 0:
            kernel_size += 1

        top_hat = cv2.morphologyEx(
            gray,
            cv2.MORPH_TOPHAT,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
            ),
        )
        _, mask = cv2.threshold(
            top_hat, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
        )
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        rows = []
        max_area = float((min_dim * 0.035) ** 2)
        for contour in contours:
            area = float(cv2.contourArea(contour))
            perimeter = float(cv2.arcLength(contour, True))
            if area < 6.0 or area > max_area or perimeter <= 0:
                continue
            moments = cv2.moments(contour)
            if abs(moments["m00"]) < 1e-6:
                continue
            circularity = 4.0 * np.pi * area / (perimeter * perimeter)
            if circularity < 0.25:
                continue
            center = np.array(
                [
                    moments["m10"] / moments["m00"],
                    moments["m01"] / moments["m00"],
                ],
                dtype=np.float32,
            )
            rows.append(
                {
                    "center": center,
                    "area": area,
                    "circularity": circularity,
                    "vertices_005": len(
                        cv2.approxPolyDP(contour, 0.05 * perimeter, True)
                    ),
                    "vertices_009": len(
                        cv2.approxPolyDP(contour, 0.09 * perimeter, True)
                    ),
                    "convex": bool(cv2.isContourConvex(contour)),
                }
            )

        if len(rows) < 40:
            return None

        all_centers = np.array([row["center"] for row in rows], dtype=np.float32)
        distances = np.linalg.norm(
            all_centers[:, None, :] - all_centers[None, :, :], axis=2
        )
        np.fill_diagonal(distances, np.inf)
        nearest = np.min(distances, axis=1)
        nearest_distance = float(np.median(nearest[np.isfinite(nearest)]))
        if nearest_distance <= 0:
            return None

        dense = np.sum(distances < 2.2 * nearest_distance, axis=1) >= 3
        rows = [row for row, keep in zip(rows, dense) if keep]
        if len(rows) < 40:
            return None

        median_area = float(np.median([row["area"] for row in rows]))
        centers = np.array([row["center"] for row in rows], dtype=np.float32)
        center_median = np.median(centers, axis=0)
        center_span = np.ptp(centers, axis=0)

        def is_central(row: dict) -> bool:
            delta = np.abs(row["center"] - center_median)
            return bool(np.all(delta <= np.maximum(0.28 * center_span, nearest_distance)))

        square_candidates = [
            (index, row)
            for index, row in enumerate(rows)
            if is_central(row)
            and row["vertices_005"] == 4
            and row["area"] >= 1.25 * median_area
        ]
        if not square_candidates:
            return None
        square_index, square = min(
            square_candidates,
            key=lambda item: (
                np.linalg.norm(
                    (item[1]["center"] - center_median)
                    / np.maximum(center_span, nearest_distance)
                ),
                -item[1]["area"],
            ),
        )

        triangle_candidates = [
            (index, row)
            for index, row in enumerate(rows)
            if index != square_index
            and is_central(row)
            and 0.35 * nearest_distance
            <= np.linalg.norm(row["center"] - square["center"])
            <= 1.45 * nearest_distance
            and row["vertices_009"] == 3
            and 0.25 * median_area <= row["area"] <= 0.95 * median_area
        ]
        if not triangle_candidates:
            triangle_candidates = [
                (index, row)
                for index, row in enumerate(rows)
                if index != square_index
                and 0.35 * nearest_distance
                <= np.linalg.norm(row["center"] - square["center"])
                <= 1.45 * nearest_distance
                and 0.20 * median_area <= row["area"] <= 0.95 * median_area
            ]
        if not triangle_candidates:
            return None

        triangle_index, triangle = min(
            triangle_candidates,
            key=lambda item: (
                0 if item[1]["vertices_009"] == 3 else 1,
                item[1]["area"],
            ),
        )

        circle_centers = np.array(
            [
                row["center"]
                for index, row in enumerate(rows)
                if index not in {square_index, triangle_index}
                and 0.30 * median_area <= row["area"] <= 2.0 * median_area
                and row["circularity"] >= 0.42
            ],
            dtype=np.float32,
        )
        if len(circle_centers) < 40:
            return None
        return circle_centers, square["center"], triangle["center"]

    def _assign_lavision_interleaved_columns(
        self,
        centers: np.ndarray,
        square_center: np.ndarray,
        triangle_center: np.ndarray,
    ) -> Optional[List[Tuple[int, int, int, float, float]]]:
        """Index alternating depth columns without assuming one common plane."""
        column_count = max(7, 2 * int(self.pattern_size[0]) - 1)
        if len(centers) < column_count * 4:
            return None

        y_center = float(np.median(centers[:, 1]))
        best_clustering = None
        for shear in np.linspace(-0.12, 0.12, 49):
            corrected_x = centers[:, 0] - shear * (centers[:, 1] - y_center)
            sort_order = np.argsort(corrected_x)
            sorted_x = corrected_x[sort_order].astype(np.float64)
            prefix = np.concatenate(([0.0], np.cumsum(sorted_x)))
            prefix_sq = np.concatenate(([0.0], np.cumsum(sorted_x * sorted_x)))
            point_count = len(sorted_x)
            infinity = float("inf")
            dp = np.full((column_count + 1, point_count + 1), infinity)
            previous = np.full(
                (column_count + 1, point_count + 1), -1, dtype=np.int32
            )
            dp[0, 0] = 0.0

            for cluster_index in range(1, column_count + 1):
                for end in range(1, point_count + 1):
                    for size in range(5, 12):
                        start = end - size
                        if start < 0 or not np.isfinite(
                            dp[cluster_index - 1, start]
                        ):
                            continue
                        value_sum = prefix[end] - prefix[start]
                        value_sq_sum = prefix_sq[end] - prefix_sq[start]
                        sse = value_sq_sum - value_sum * value_sum / size
                        count_cost = 20.0 * (size - 7.5) ** 2
                        score = dp[cluster_index - 1, start] + sse + count_cost
                        if score < dp[cluster_index, end]:
                            dp[cluster_index, end] = score
                            previous[cluster_index, end] = start

            if not np.isfinite(dp[column_count, point_count]):
                continue
            ordered_labels = np.empty(point_count, dtype=np.int32)
            end = point_count
            for cluster_index in range(column_count, 0, -1):
                start = int(previous[cluster_index, end])
                ordered_labels[sort_order[start:end]] = cluster_index - 1
                end = start
            score = float(dp[column_count, point_count])
            if best_clustering is None or score < best_clustering[0]:
                best_clustering = (score, ordered_labels)

        if best_clustering is None:
            return None
        labels = best_clustering[1]

        # Points in one physical column are separated by two fine-grid steps.
        column_differences = []
        for column_index in range(column_count):
            indices = np.where(labels == column_index)[0]
            ordered = indices[np.argsort(centers[indices, 1])]
            if len(ordered) >= 2:
                differences = np.diff(centers[ordered], axis=0)
                column_differences.extend(
                    difference
                    for difference in differences
                    if difference[1] > 0
                )
        if not column_differences:
            return None
        fine_vertical = 0.5 * np.median(
            np.asarray(column_differences, dtype=np.float64), axis=0
        )
        nearest_distance = self._median_nearest_distance(centers)
        local_vertical_offsets = []
        for marker in (square_center, triangle_center):
            delta_y = np.abs(centers[:, 1] - marker[1])
            local_vertical_offsets.extend(
                float(value)
                for value in delta_y
                if 0.30 * nearest_distance <= value <= 1.20 * nearest_distance
            )
        if local_vertical_offsets:
            fine_vertical[1] = float(np.median(local_vertical_offsets))
            fine_vertical[0] = 0.0
        vertical_norm_sq = float(np.dot(fine_vertical, fine_vertical))
        if vertical_norm_sq < 4.0:
            return None

        center_column = (column_count - 1) // 2
        assignments = []
        used_ids = set()
        for column_index in range(column_count):
            column_indices = np.where(labels == column_index)[0]
            fine_x = int(column_index - center_column)
            layer = 0.0 if abs(fine_x) % 2 == 0 else 1.0
            layer_center = square_center if layer == 0.0 else triangle_center
            y_coordinates = np.array(
                [
                    np.dot(
                        centers[center_index].astype(np.float64)
                        - layer_center.astype(np.float64),
                        fine_vertical,
                    )
                    / vertical_norm_sq
                    for center_index in column_indices
                ],
                dtype=np.float64,
            )

            phase_results = []
            for phase in (0, 1):
                fine_rows = (
                    np.rint((y_coordinates - phase) / 2.0) * 2 + phase
                ).astype(np.int32)
                row_residuals = np.abs(y_coordinates - fine_rows)
                phase_results.append(
                    (
                        len(set(int(value) for value in fine_rows)),
                        -float(np.median(row_residuals)),
                        fine_rows,
                        row_residuals,
                    )
                )
            _, _, fine_rows, row_residuals = max(
                phase_results, key=lambda item: (item[0], item[1])
            )
            for center_index, fine_y, residual in zip(
                column_indices, fine_rows, row_residuals
            ):
                point_id = (layer, fine_x, int(fine_y))
                if (
                    residual > 0.55
                    or abs(int(fine_y)) > 7
                    or point_id in used_ids
                ):
                    continue
                used_ids.add(point_id)
                assignments.append(
                    (
                        int(center_index),
                        fine_x,
                        int(fine_y),
                        layer,
                        float(residual),
                    )
                )

        layer_counts = [
            sum(row[3] == layer for row in assignments)
            for layer in (0.0, 1.0)
        ]
        if min(layer_counts) < 40 or len(assignments) < 90:
            return None
        return assignments

    def _fit_lavision_fine_lattice(
        self, marker_center: np.ndarray, centers: np.ndarray
    ) -> Optional[Tuple[np.ndarray, np.ndarray, List[Tuple[int, np.ndarray, float]]]]:
        """Recover the 15-column interleaved checker lattice around the square."""
        offsets = centers.astype(np.float64) - marker_center.astype(np.float64)
        radii = np.linalg.norm(offsets, axis=1)
        nearby = np.argsort(radii)[: min(60, len(centers))]
        opposite_vectors = []

        for row, first in enumerate(nearby):
            for second in nearby[row + 1 :]:
                mean_radius = 0.5 * (radii[first] + radii[second])
                if mean_radius < 4.0:
                    continue
                if abs(radii[first] - radii[second]) / mean_radius > 0.40:
                    continue
                symmetry_error = (
                    np.linalg.norm(offsets[first] + offsets[second]) / mean_radius
                )
                if symmetry_error <= 0.35:
                    opposite_vectors.append(
                        (
                            float(symmetry_error),
                            0.5 * (offsets[first] - offsets[second]),
                        )
                    )

        if len(opposite_vectors) < 2:
            return None
        opposite_vectors.sort(key=lambda item: item[0])
        candidates = opposite_vectors[:60]

        horizontal_candidates = []
        for symmetry_error, raw_vector in candidates:
            for scale in (0.5, 1.0, 2.0):
                vector = raw_vector * scale
                if abs(vector[0]) < 1.5 * abs(vector[1]):
                    continue
                if vector[0] < 0:
                    vector = -vector
                if not any(
                    np.linalg.norm(vector - existing[1])
                    < 0.08 * np.linalg.norm(vector)
                    for existing in horizontal_candidates
                ):
                    horizontal_candidates.append((symmetry_error, vector))
        horizontal_candidates = horizontal_candidates[:18]

        best = None
        for error_u, basis_u in horizontal_candidates:
            for error_v, raw_v in candidates:
                for x_multiple in range(-6, 7):
                    for y_multiple in tuple(range(-6, 0)) + tuple(range(1, 7)):
                        basis_v = (
                            raw_v - x_multiple * basis_u
                        ) / float(y_multiple)
                        if abs(basis_v[1]) < 1.5 * abs(basis_v[0]):
                            continue
                        ratio = np.linalg.norm(basis_v) / np.linalg.norm(basis_u)
                        if ratio < 0.55 or ratio > 1.8:
                            continue
                        if basis_v[1] < 0:
                            basis_v = -basis_v

                        basis = np.column_stack((basis_u, basis_v))
                        if abs(np.linalg.det(basis)) < 20.0:
                            continue
                        uv = np.linalg.solve(basis, offsets.T).T
                        rounded = np.rint(uv)
                        residuals = np.linalg.norm(uv - rounded, axis=1)
                        parity = (
                            np.abs(rounded).astype(np.int32).sum(axis=1) % 2
                        ) == 1
                        valid = (
                            (residuals <= 0.24)
                            & parity
                            & (np.abs(rounded[:, 0]) <= 8)
                            & (np.abs(rounded[:, 1]) <= 9)
                        )
                        unique_count = len(
                            {tuple(value.astype(np.int32)) for value in rounded[valid]}
                        )
                        if unique_count < 70:
                            continue
                        score = (
                            unique_count * 10.0
                            - float(np.median(residuals[valid])) * 10.0
                            - error_u
                            - error_v
                        )
                        if best is None or score > best[0]:
                            best = (score, basis_u.copy(), basis_v.copy())

        if best is None:
            return None
        _, basis_u, basis_v = best
        if basis_u[0] < 0:
            basis_u = -basis_u
        if basis_v[1] < 0:
            basis_v = -basis_v

        uv = np.linalg.solve(
            np.column_stack((basis_u, basis_v)), offsets.T
        ).T
        rounded = np.rint(uv).astype(np.int32)
        residuals = np.linalg.norm(uv - rounded, axis=1)
        best_by_grid: Dict[Tuple[int, int], Tuple[int, float]] = {}
        for index, (grid_xy, residual) in enumerate(zip(rounded, residuals)):
            grid_key = (int(grid_xy[0]), int(grid_xy[1]))
            if (
                residual > 0.24
                or (abs(grid_key[0]) + abs(grid_key[1])) % 2 != 1
                or abs(grid_key[0]) > 8
                or abs(grid_key[1]) > 9
            ):
                continue
            current = best_by_grid.get(grid_key)
            if current is None or residual < current[1]:
                best_by_grid[grid_key] = (index, float(residual))

        assignments = [
            (center_index, np.array(grid_key, dtype=np.int32), residual)
            for grid_key, (center_index, residual) in best_by_grid.items()
        ]
        if len(assignments) < 70:
            return None
        return basis_u.astype(np.float32), basis_v.astype(np.float32), assignments

    def _fit_lavision_layer_lattice(
        self, marker_center: np.ndarray, centers: np.ndarray
    ) -> Optional[Tuple[np.ndarray, np.ndarray, List[Tuple[int, np.ndarray, float]]]]:
        offsets = centers.astype(np.float64) - marker_center.astype(np.float64)
        radii = np.linalg.norm(offsets, axis=1)
        nearby = np.argsort(radii)[: min(60, len(centers))]

        opposite_vectors = []
        for row, first in enumerate(nearby):
            for second in nearby[row + 1 :]:
                mean_radius = 0.5 * (radii[first] + radii[second])
                if mean_radius < 4.0:
                    continue
                if abs(radii[first] - radii[second]) / mean_radius > 0.40:
                    continue
                symmetry_error = (
                    np.linalg.norm(offsets[first] + offsets[second]) / mean_radius
                )
                if symmetry_error <= 0.35:
                    opposite_vectors.append(
                        (
                            float(symmetry_error),
                            0.5 * (offsets[first] - offsets[second]),
                        )
                    )

        if len(opposite_vectors) < 2:
            return None
        opposite_vectors.sort(key=lambda item: item[0])
        candidates = opposite_vectors[:60]

        best = None
        horizontal_candidates = []
        for symmetry_error, raw_vector in candidates:
            for scale in (1.0, 2.0):
                vector = raw_vector * scale
                if abs(vector[0]) < 1.5 * abs(vector[1]):
                    continue
                if vector[0] < 0:
                    vector = -vector
                if not any(
                    np.linalg.norm(vector - existing[1])
                    < 0.08 * np.linalg.norm(vector)
                    for existing in horizontal_candidates
                ):
                    horizontal_candidates.append((symmetry_error, vector))
        horizontal_candidates = horizontal_candidates[:16]

        half_x_values = np.arange(-3.5, 4.0, 0.5)
        y_multipliers = (-3, -2, -1, 1, 2, 3)
        max_x = max(2.0, self.pattern_size[0] / 2.0 + 0.1)
        max_y = max(2.0, (self.pattern_size[1] - 1) / 2.0 + 0.1)

        for error_u, basis_u in horizontal_candidates:
            for error_v, raw_v in candidates:
                for x_multiple in half_x_values:
                    for y_multiple in y_multipliers:
                        basis_v = (
                            raw_v - x_multiple * basis_u
                        ) / float(y_multiple)
                        if abs(basis_v[1]) < 1.5 * abs(basis_v[0]):
                            continue
                        if not 0.5 * np.linalg.norm(basis_u) <= np.linalg.norm(
                            basis_v
                        ) <= 2.5 * np.linalg.norm(basis_u):
                            continue
                        if basis_v[1] < 0:
                            basis_v = -basis_v

                        basis = np.column_stack((basis_u, basis_v))
                        if abs(np.linalg.det(basis)) < 100.0:
                            continue
                        uv = np.linalg.solve(basis, offsets.T).T
                        for phase_x, phase_y in ((0.5, 0.0), (0.0, 0.5)):
                            rounded = np.column_stack(
                                (
                                    np.rint(uv[:, 0] - phase_x) + phase_x,
                                    np.rint(uv[:, 1] - phase_y) + phase_y,
                                )
                            )
                            residuals = np.linalg.norm(uv - rounded, axis=1)
                            valid = (
                                (residuals <= 0.24)
                                & (np.abs(rounded[:, 0]) <= max_x)
                                & (np.abs(rounded[:, 1]) <= max_y)
                            )
                            unique_count = len(
                                {tuple(value) for value in rounded[valid]}
                            )
                            if unique_count < 20:
                                continue
                            score = (
                                unique_count * 10.0
                                - float(np.median(residuals[valid])) * 10.0
                                - error_u
                                - error_v
                            )
                            if best is None or score > best[0]:
                                best = (
                                    score,
                                    basis_u.copy(),
                                    basis_v.copy(),
                                    phase_x,
                                    phase_y,
                                )

        if best is None:
            return None

        _, basis_u, basis_v, phase_x, phase_y = best
        # Stable orientation gives all cameras the same point IDs for an
        # upright target, independent of contour ordering.
        if basis_u[0] < 0:
            basis_u = -basis_u
        if basis_v[1] < 0:
            basis_v = -basis_v

        basis = np.column_stack((basis_u, basis_v))
        uv = np.linalg.solve(basis, offsets.T).T
        rounded = np.column_stack(
            (
                np.rint(uv[:, 0] - phase_x) + phase_x,
                np.rint(uv[:, 1] - phase_y) + phase_y,
            )
        )
        residuals = np.linalg.norm(uv - rounded, axis=1)

        best_by_grid: Dict[Tuple[float, float], Tuple[int, float]] = {}
        for index, (grid_xy, residual) in enumerate(zip(rounded, residuals)):
            grid_key = (float(grid_xy[0]), float(grid_xy[1]))
            if (
                residual > 0.24
                or abs(grid_key[0]) > max_x
                or abs(grid_key[1]) > max_y
            ):
                continue
            current = best_by_grid.get(grid_key)
            if current is None or residual < current[1]:
                best_by_grid[grid_key] = (index, float(residual))

        assignments = [
            (center_index, np.array(grid_key, dtype=np.float32), residual)
            for grid_key, (center_index, residual) in best_by_grid.items()
        ]
        if len(assignments) < 20:
            return None
        return basis_u.astype(np.float32), basis_v.astype(np.float32), assignments

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
        gray8 = self._to_gray8(image)
        vis = cv2.cvtColor(gray8, cv2.COLOR_GRAY2BGR)
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
        observations: List[Optional[PatternObservation]] = []
        detected_count = 0
        image_size = None

        for img_path in image_paths:
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                logger.warning("Failed to read image: %s", img_path)
                continue

            if image_size is None:
                image_size = (img.shape[1], img.shape[0])

            observation = self.detect_pattern_observation(img)
            observations.append(observation)
            if observation is None:
                continue

            if self.pattern_type == "volume_dots":
                # Each detected LaVision level is internally very accurate,
                # while a wrong cross-level correspondence produces a
                # plausible-looking but physically impossible intrinsic fit.
                # Use the square-fiducial level for stable intrinsics; both
                # levels remain available in `observations` for display.
                calibration_indices = [
                    index
                    for index, point_id in enumerate(observation.point_ids)
                    if len(point_id) == 3
                    and point_id[0] == 0.0
                    and point_id[1:] != (0.0, 0.0)
                ]
            else:
                calibration_indices = list(range(len(observation.point_ids)))
            if len(calibration_indices) < 8:
                continue
            obj_points_list.append(
                observation.object_points[calibration_indices].copy()
            )
            img_points_list.append(
                observation.image_points[calibration_indices].copy()
            )
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

        camera_matrix_guess = None
        calibration_flags = 0
        if self.pattern_type == "volume_dots":
            focal_guess = float(max(image_size)) * 3.0
            camera_matrix_guess = np.array(
                [
                    [focal_guess, 0.0, image_size[0] * 0.5],
                    [0.0, focal_guess, image_size[1] * 0.5],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
            calibration_flags |= (
                cv2.CALIB_USE_INTRINSIC_GUESS
                | cv2.CALIB_FIX_ASPECT_RATIO
                | cv2.CALIB_FIX_PRINCIPAL_POINT
                | cv2.CALIB_ZERO_TANGENT_DIST
                | cv2.CALIB_FIX_K1
                | cv2.CALIB_FIX_K2
                | cv2.CALIB_FIX_K3
                | cv2.CALIB_FIX_K4
                | cv2.CALIB_FIX_K5
                | cv2.CALIB_FIX_K6
            )

        rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            obj_points_list,
            img_points_list,
            image_size,
            camera_matrix_guess,
            None,
            flags=calibration_flags,
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
            "observations": observations,
            "full_object_points": (
                observations[next(
                    index for index, value in enumerate(observations)
                    if value is not None
                )].object_points.copy()
                if any(value is not None for value in observations)
                else np.empty((0, 3), dtype=np.float32)
            ),
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

        self.finalize_multi_camera_calibration()
        return results

    def finalize_multi_camera_calibration(self) -> Dict[str, dict]:
        """Place all cameras in one physically consistent target coordinate frame.

        LaVision plates provide excellent per-level planar correspondences. For
        same-resolution camera arrays, a shared robust focal estimate prevents
        a weak single view from drifting to an extreme focal length. Relative
        poses are then solved from the square-fiducial level common to each
        camera pair and composed with the reference-camera board pose.
        """
        camera_ids = list(self.camera_params)
        if len(camera_ids) < 2:
            return {}

        if self.pattern_type == "volume_dots":
            image_sizes = {
                tuple(self.camera_params[camera_id].image_size)
                for camera_id in camera_ids
            }
            widths = [size[0] for size in image_sizes]
            heights = [size[1] for size in image_sizes]
            same_sensor_format = (
                max(widths) - min(widths) <= 2
                and max(heights) - min(heights) <= 2
            )
            if same_sensor_format:
                focal_values = [
                    0.5
                    * (
                        self.camera_params[camera_id].camera_matrix[0][0]
                        + self.camera_params[camera_id].camera_matrix[1][1]
                    )
                    for camera_id in camera_ids
                ]
                shared_focal = float(np.median(focal_values))
                fixed_flags = (
                    cv2.CALIB_USE_INTRINSIC_GUESS
                    | cv2.CALIB_FIX_FOCAL_LENGTH
                    | cv2.CALIB_FIX_PRINCIPAL_POINT
                    | cv2.CALIB_ZERO_TANGENT_DIST
                    | cv2.CALIB_FIX_K1
                    | cv2.CALIB_FIX_K2
                    | cv2.CALIB_FIX_K3
                    | cv2.CALIB_FIX_K4
                    | cv2.CALIB_FIX_K5
                    | cv2.CALIB_FIX_K6
                )
                for camera_id in camera_ids:
                    data = self._calib_data[camera_id]
                    image_size = tuple(
                        self.camera_params[camera_id].image_size
                    )
                    camera_matrix = np.array(
                        [
                            [shared_focal, 0.0, image_size[0] * 0.5],
                            [0.0, shared_focal, image_size[1] * 0.5],
                            [0.0, 0.0, 1.0],
                        ],
                        dtype=np.float64,
                    )
                    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
                        data["obj_points"],
                        data["img_points"],
                        image_size,
                        camera_matrix,
                        None,
                        flags=fixed_flags,
                    )
                    params = self.camera_params[camera_id]
                    params.camera_matrix = camera_matrix.tolist()
                    params.dist_coeffs = dist_coeffs.flatten().tolist()
                    params.rms_error = float(rms)
                    data["rvecs"] = rvecs
                    data["tvecs"] = tvecs

        reference_id = camera_ids[0]
        reference_data = self._calib_data[reference_id]
        if not reference_data.get("rvecs") or not reference_data.get("tvecs"):
            return {}

        reference_rvec = np.asarray(reference_data["rvecs"][0]).reshape(3, 1)
        reference_tvec = np.asarray(reference_data["tvecs"][0]).reshape(3, 1)
        reference_rotation, _ = cv2.Rodrigues(reference_rvec)
        joint_poses: Dict[str, dict] = {
            reference_id: {
                "rvec": reference_rvec.reshape(3),
                "tvec": reference_tvec.reshape(3),
                "stereo_rms": 0.0,
            }
        }

        for camera_id in camera_ids[1:]:
            other_data = self._calib_data[camera_id]
            obj_points = []
            reference_points = []
            other_points = []
            for reference_obs, other_obs in zip(
                reference_data.get("observations", []),
                other_data.get("observations", []),
            ):
                if reference_obs is None or other_obs is None:
                    continue
                common_ids = sorted(
                    [
                        point_id
                        for point_id in (
                            set(reference_obs.point_ids)
                            & set(other_obs.point_ids)
                        )
                        if (
                            len(point_id) != 3
                            or (
                                point_id[0] == 0.0
                                and point_id[1:] != (0.0, 0.0)
                            )
                        )
                    ],
                    key=self._point_sort_key,
                )
                if len(common_ids) < 8:
                    continue
                reference_map = dict(
                    zip(
                        reference_obs.point_ids,
                        reference_obs.image_points.reshape(-1, 2),
                    )
                )
                other_map = dict(
                    zip(
                        other_obs.point_ids,
                        other_obs.image_points.reshape(-1, 2),
                    )
                )
                obj_points.append(self._object_points_from_ids(common_ids))
                reference_points.append(
                    np.asarray(
                        [reference_map[point_id] for point_id in common_ids],
                        dtype=np.float32,
                    ).reshape(-1, 1, 2)
                )
                other_points.append(
                    np.asarray(
                        [other_map[point_id] for point_id in common_ids],
                        dtype=np.float32,
                    ).reshape(-1, 1, 2)
                )

            if len(obj_points) < 2:
                logger.warning(
                    "Camera %s has insufficient paired observations for joint alignment",
                    camera_id,
                )
                continue

            reference_params = self.camera_params[reference_id]
            other_params = self.camera_params[camera_id]
            stereo_rms, _, _, _, _, relative_rotation, relative_translation, _, _ = (
                cv2.stereoCalibrate(
                    obj_points,
                    reference_points,
                    other_points,
                    np.asarray(reference_params.camera_matrix, dtype=np.float64),
                    np.asarray(reference_params.dist_coeffs, dtype=np.float64),
                    np.asarray(other_params.camera_matrix, dtype=np.float64),
                    np.asarray(other_params.dist_coeffs, dtype=np.float64),
                    tuple(reference_params.image_size),
                    flags=cv2.CALIB_FIX_INTRINSIC,
                )
            )
            world_to_camera = relative_rotation @ reference_rotation
            world_translation = (
                relative_rotation @ reference_tvec + relative_translation
            )
            world_rvec, _ = cv2.Rodrigues(world_to_camera)
            joint_poses[camera_id] = {
                "rvec": world_rvec.reshape(3),
                "tvec": world_translation.reshape(3),
                "stereo_rms": float(stereo_rms),
            }

        for camera_id, pose in joint_poses.items():
            params = self.camera_params[camera_id]
            params.rvec = np.asarray(pose["rvec"]).reshape(3).tolist()
            params.tvec = np.asarray(pose["tvec"]).reshape(3).tolist()

        self._joint_camera_poses = joint_poses
        return joint_poses

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
            img1 = cv2.imread(img1_path, cv2.IMREAD_UNCHANGED)
            img2 = cv2.imread(img2_path, cv2.IMREAD_UNCHANGED)
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
            if self.pattern_type == "volume_dots":
                common_ids = [
                    point_id
                    for point_id in common_ids
                    if not (len(point_id) == 3 and point_id[1:] == (0.0, 0.0))
                ]
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
            "origin_point_id": (
                list(self.origin_point_id)
                if self.origin_point_id is not None
                else None
            ),
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
            origin_point_id = summary.get("origin_point_id")
            calib.origin_point_id = (
                tuple(origin_point_id) if origin_point_id is not None else None
            )
            calib.obj_points = calib._generate_object_points()
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
