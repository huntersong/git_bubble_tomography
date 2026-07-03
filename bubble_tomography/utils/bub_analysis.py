"""Bubble recognition and size statistics adapted from BubAnalysis VOL1.0.

The original MATLAB workflow was developed by the Nuclear Power Pump and
Valve Laboratory, Shanghai Jiao Tong University (WG-Chen, 2021-06-05).
This module reimplements the documented processing flow in Python so it can
be used by the application's image-processing workflow.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from scipy import ndimage


@dataclass
class BubAnalysisParams:
    enabled: bool = False
    background_path: str = ""
    scale_mm_per_px: float = 1.0
    min_area: int = 20
    min_diameter: float = 5.0
    max_diameter: float = 200.0
    min_circularity: float = 0.40
    focused_filter: bool = True
    gradient_threshold: float = 0.25
    gray_threshold: float = 0.25
    split_overlaps: bool = True
    bilateral_filter: bool = False


@dataclass
class BubbleMeasurement:
    bubble_id: int
    center_x_px: float
    center_y_px: float
    diameter_px: float
    diameter_mm: float
    area_px2: float
    perimeter_px: float
    circularity: float
    major_axis_px: float
    minor_axis_px: float
    angle_deg: float


@dataclass
class BubAnalysisResult:
    overlay: np.ndarray
    mask: np.ndarray
    bubbles: List[BubbleMeasurement] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.bubbles)

    @property
    def mean_diameter_mm(self) -> float:
        if not self.bubbles:
            return 0.0
        return float(np.mean([item.diameter_mm for item in self.bubbles]))

    @property
    def sauter_mean_diameter_mm(self) -> float:
        diameters = np.asarray(
            [item.diameter_mm for item in self.bubbles], dtype=np.float64
        )
        denominator = float(np.sum(diameters ** 2))
        if denominator <= 0:
            return 0.0
        return float(np.sum(diameters ** 3) / denominator)

    def summary(self) -> str:
        return (
            f"气泡数: {self.count}, "
            f"平均直径: {self.mean_diameter_mm:.4g} mm, "
            f"Sauter平均直径 D32: {self.sauter_mean_diameter_mm:.4g} mm"
        )

    def write_csv(self, path: str) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        columns = [
            "bubble_id",
            "center_x_px",
            "center_y_px",
            "diameter_px",
            "diameter_mm",
            "area_px2",
            "perimeter_px",
            "circularity",
            "major_axis_px",
            "minor_axis_px",
            "angle_deg",
        ]
        with output.open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            for item in self.bubbles:
                writer.writerow({name: getattr(item, name) for name in columns})

    def render_distribution_chart(
        self, width: int = 960, height: int = 540
    ) -> np.ndarray:
        """Render a shareable diameter histogram with a cumulative curve."""
        canvas = np.full((height, width, 3), 248, dtype=np.uint8)
        left, right, top, bottom = 90, width - 80, 80, height - 75
        cv2.rectangle(canvas, (left, top), (right, bottom), (70, 70, 70), 1)

        diameters = np.asarray(
            [item.diameter_mm for item in self.bubbles], dtype=np.float64
        )
        cv2.putText(
            canvas,
            "Bubble diameter distribution",
            (left, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (25, 25, 25),
            2,
            cv2.LINE_AA,
        )
        summary = (
            f"N={self.count}   Mean={self.mean_diameter_mm:.4g} mm   "
            f"D32={self.sauter_mean_diameter_mm:.4g} mm"
        )
        cv2.putText(
            canvas, summary, (left, 65), cv2.FONT_HERSHEY_SIMPLEX,
            0.48, (70, 70, 70), 1, cv2.LINE_AA
        )
        if diameters.size == 0:
            cv2.putText(
                canvas, "No accepted bubbles", (left + 220, top + 190),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2, cv2.LINE_AA
            )
            return canvas

        bin_count = min(20, max(5, int(np.ceil(np.sqrt(diameters.size)))))
        if np.allclose(diameters.min(), diameters.max()):
            padding = max(0.5, abs(float(diameters[0])) * 0.1)
            value_range = (diameters.min() - padding, diameters.max() + padding)
        else:
            value_range = (float(diameters.min()), float(diameters.max()))
        counts, edges = np.histogram(diameters, bins=bin_count, range=value_range)
        max_count = max(1, int(counts.max()))
        plot_width = right - left
        plot_height = bottom - top
        bar_width = plot_width / bin_count

        for index, count in enumerate(counts):
            x0 = round(left + index * bar_width + 2)
            x1 = round(left + (index + 1) * bar_width - 2)
            y = round(bottom - (count / max_count) * plot_height)
            cv2.rectangle(canvas, (x0, y), (x1, bottom), (219, 132, 52), -1)
            cv2.rectangle(canvas, (x0, y), (x1, bottom), (160, 90, 25), 1)

        cumulative = np.cumsum(counts) / max(1, counts.sum())
        curve_points = []
        for index, value in enumerate(cumulative):
            x = round(left + (index + 0.5) * bar_width)
            y = round(bottom - value * plot_height)
            curve_points.append((x, y))
        if curve_points:
            cv2.polylines(
                canvas, [np.asarray(curve_points, dtype=np.int32)],
                False, (55, 78, 210), 2, cv2.LINE_AA
            )
            for point in curve_points:
                cv2.circle(canvas, point, 3, (55, 78, 210), -1)

        for tick in range(6):
            fraction = tick / 5.0
            y = round(bottom - fraction * plot_height)
            cv2.line(canvas, (left - 5, y), (left, y), (50, 50, 50), 1)
            cv2.putText(
                canvas, f"{max_count * fraction:.1f}", (15, y + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (65, 65, 65), 1, cv2.LINE_AA
            )
            cv2.putText(
                canvas, f"{fraction * 100:.0f}%", (right + 10, y + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (55, 78, 210), 1, cv2.LINE_AA
            )

        for index in range(5):
            fraction = index / 4.0
            x = round(left + fraction * plot_width)
            value = value_range[0] + fraction * (value_range[1] - value_range[0])
            cv2.line(canvas, (x, bottom), (x, bottom + 5), (50, 50, 50), 1)
            cv2.putText(
                canvas, f"{value:.3g}", (x - 18, bottom + 23),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (65, 65, 65), 1, cv2.LINE_AA
            )
        cv2.putText(
            canvas, "Diameter (mm)", (left + plot_width // 2 - 55, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (45, 45, 45), 1, cv2.LINE_AA
        )
        cv2.putText(
            canvas, "Count", (16, top - 12), cv2.FONT_HERSHEY_SIMPLEX,
            0.42, (160, 90, 25), 1, cv2.LINE_AA
        )
        cv2.putText(
            canvas, "Cumulative", (right - 85, top - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (55, 78, 210), 1, cv2.LINE_AA
        )
        return canvas

    @staticmethod
    def _write_image(path: Path, image: np.ndarray) -> None:
        success, encoded = cv2.imencode(path.suffix or ".png", image)
        if not success:
            raise IOError(f"Unable to encode image: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded.tofile(str(path))

    def write_artifacts(self, processed_image_path: str) -> dict:
        """Write every BubAnalysis output beside the normal processed image."""
        source = Path(processed_image_path)
        stem = source.with_suffix("")
        paths = {
            "annotated": Path(str(stem) + "_bubble_annotated.png"),
            "mask": Path(str(stem) + "_bubble_mask.png"),
            "distribution": Path(str(stem) + "_diameter_distribution.png"),
            "csv": Path(str(stem) + "_bubbles.csv"),
            "summary": Path(str(stem) + "_bubble_summary.txt"),
        }
        self._write_image(paths["annotated"], self.overlay)
        self._write_image(paths["mask"], self.mask)
        self._write_image(paths["distribution"], self.render_distribution_chart())
        self.write_csv(str(paths["csv"]))
        with paths["summary"].open("w", encoding="utf-8") as stream:
            stream.write(self.summary() + "\n")
        return {name: str(path) for name, path in paths.items()}


class BubAnalysisProcessor:
    """Background-difference bubble segmentation and dimensional analysis."""

    def __init__(self, params: Optional[BubAnalysisParams] = None):
        self.params = params or BubAnalysisParams()

    @staticmethod
    def _to_gray8(image: np.ndarray) -> np.ndarray:
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image.dtype == np.uint8:
            return image
        return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    @staticmethod
    def _read_image(path: str) -> Optional[np.ndarray]:
        try:
            encoded = np.fromfile(path, dtype=np.uint8)
            return cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
        except Exception:
            return None

    def _foreground(self, gray: np.ndarray) -> np.ndarray:
        p = self.params
        background = self._read_image(p.background_path) if p.background_path else None
        if background is not None:
            background = self._to_gray8(background)
            if background.shape != gray.shape:
                background = cv2.resize(
                    background, (gray.shape[1], gray.shape[0]), cv2.INTER_AREA
                )
            foreground = cv2.subtract(background, gray)
        else:
            foreground = cv2.subtract(np.full_like(gray, 255), gray)

        foreground = cv2.normalize(foreground, None, 0, 255, cv2.NORM_MINMAX)
        foreground = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(
            foreground
        )
        if p.bilateral_filter:
            foreground = cv2.bilateralFilter(foreground, 7, 30, 30)
        return cv2.medianBlur(foreground, 3)

    def _initial_mask(self, foreground: np.ndarray) -> np.ndarray:
        _, mask = cv2.threshold(
            foreground, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        mask = ndimage.binary_fill_holes(mask > 0)
        labels, count = ndimage.label(mask)
        if count:
            sizes = np.bincount(labels.ravel())
            keep = sizes >= max(1, self.params.min_area)
            keep[0] = False
            mask = keep[labels]

        # Match MATLAB imclearborder: partial bubbles at the image edge are excluded.
        border_labels = np.unique(
            np.concatenate(
                (labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1])
            )
        )
        for label_id in border_labels:
            if label_id:
                mask[labels == label_id] = False
        return (mask.astype(np.uint8) * 255)

    def _split_labels(self, mask: np.ndarray, source: np.ndarray) -> np.ndarray:
        component_count, components = cv2.connectedComponents(mask)
        if not self.params.split_overlaps:
            return components.astype(np.int32)

        distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        if not np.any(distance > 0):
            return np.zeros(mask.shape, dtype=np.int32)

        # Build peaks per connected component.  A threshold based on the global
        # maximum makes small bubbles disappear whenever a much larger bubble
        # is present in the same image.
        local_maxima = distance == ndimage.maximum_filter(distance, size=9)
        markers = np.zeros(mask.shape, dtype=np.int32)
        next_marker = 1
        for component_id in range(1, component_count):
            region = components == component_id
            component_max = float(np.max(distance[region]))
            peaks = (
                region
                & local_maxima
                & (distance > max(1.0, component_max * 0.25))
            )
            peak_labels, peak_count = ndimage.label(peaks)
            if peak_count == 0:
                # Retain even very small valid components instead of allowing
                # the watershed background marker to consume them.
                flat_index = int(np.argmax(np.where(region, distance, -1.0)))
                y, x = np.unravel_index(flat_index, distance.shape)
                markers[y, x] = next_marker
                next_marker += 1
                continue
            for peak_id in range(1, peak_count + 1):
                markers[peak_labels == peak_id] = next_marker
                next_marker += 1

        if next_marker <= 2:
            return components.astype(np.int32)

        # OpenCV watershed requires known background=1, unknown pixels=0 and
        # foreground seeds>=2.  Previously the background was set to 0, so
        # bubble labels flooded into the whole image and were later rejected.
        markers += 1
        markers[(mask > 0) & (markers == 1)] = 0
        watershed_source = cv2.cvtColor(source, cv2.COLOR_GRAY2BGR)
        cv2.watershed(watershed_source, markers)
        labels = markers - 1
        labels[labels < 0] = 0
        return labels.astype(np.int32)

    def _is_focused(
        self, component: np.ndarray, gray: np.ndarray, gradient: np.ndarray
    ) -> bool:
        if not self.params.focused_filter:
            return True
        boundary = cv2.morphologyEx(component, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
        points = boundary > 0
        if not np.any(points):
            return False
        mean_gradient = float(np.mean(gradient[points]))
        mean_darkness = float(np.mean((255.0 - gray[points]) / 255.0))
        return (
            mean_gradient >= self.params.gradient_threshold
            or mean_darkness >= self.params.gray_threshold
        )

    def process(self, image: np.ndarray) -> BubAnalysisResult:
        if image is None or image.size == 0:
            raise ValueError("BubAnalysis received an empty image")

        p = self.params
        gray = self._to_gray8(image)
        foreground = self._foreground(gray)
        mask = self._initial_mask(foreground)
        labels = self._split_labels(mask, foreground)
        gradient = cv2.normalize(
            cv2.magnitude(
                cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3),
                cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3),
            ),
            None,
            0.0,
            1.0,
            cv2.NORM_MINMAX,
        )

        overlay = (
            cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            if image.ndim == 2
            else self._to_display_bgr(image)
        )
        accepted_mask = np.zeros(gray.shape, dtype=np.uint8)
        bubbles: List[BubbleMeasurement] = []

        for label_id in range(1, int(labels.max()) + 1):
            component = np.uint8(labels == label_id) * 255
            area = float(cv2.countNonZero(component))
            if area < p.min_area or not self._is_focused(component, gray, gradient):
                continue
            contours, _ = cv2.findContours(
                component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            if not contours:
                continue
            contour = max(contours, key=cv2.contourArea)
            perimeter = float(cv2.arcLength(contour, True))
            circularity = (
                float(4.0 * np.pi * area / (perimeter * perimeter))
                if perimeter > 0
                else 0.0
            )
            diameter = float(np.sqrt(4.0 * area / np.pi))
            if (
                diameter < p.min_diameter
                or diameter > p.max_diameter
                or circularity < p.min_circularity
            ):
                continue

            if len(contour) >= 5:
                (cx, cy), (axis_a, axis_b), angle = cv2.fitEllipse(contour)
                major, minor = max(axis_a, axis_b), min(axis_a, axis_b)
                cv2.ellipse(overlay, ((cx, cy), (axis_a, axis_b), angle), (0, 255, 0), 1)
            else:
                (cx, cy), radius = cv2.minEnclosingCircle(contour)
                major = minor = radius * 2.0
                angle = 0.0
                cv2.circle(overlay, (round(cx), round(cy)), round(radius), (0, 255, 0), 1)

            bubble_id = len(bubbles) + 1
            cv2.putText(
                overlay,
                str(bubble_id),
                (round(cx) + 3, round(cy) - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (0, 220, 255),
                1,
                cv2.LINE_AA,
            )
            accepted_mask[component > 0] = 255
            bubbles.append(
                BubbleMeasurement(
                    bubble_id=bubble_id,
                    center_x_px=float(cx),
                    center_y_px=float(cy),
                    diameter_px=diameter,
                    diameter_mm=diameter * p.scale_mm_per_px,
                    area_px2=area,
                    perimeter_px=perimeter,
                    circularity=circularity,
                    major_axis_px=float(major),
                    minor_axis_px=float(minor),
                    angle_deg=float(angle),
                )
            )

        cv2.putText(
            overlay,
            f"N={len(bubbles)}  D32={BubAnalysisResult(overlay, accepted_mask, bubbles).sauter_mean_diameter_mm:.4g} mm",
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 220, 255),
            1,
            cv2.LINE_AA,
        )
        return BubAnalysisResult(overlay=overlay, mask=accepted_mask, bubbles=bubbles)

    @staticmethod
    def _to_display_bgr(image: np.ndarray) -> np.ndarray:
        if image.dtype != np.uint8:
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        return image.copy()
