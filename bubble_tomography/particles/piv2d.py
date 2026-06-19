"""
二维 PIV 互相关计算模块
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import correlate2d


SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass
class PIV2DConfig:
    window_size: int = 32
    overlap_ratio: float = 0.5
    search_radius: int = 16
    dt: float = 1.0
    pixel_scale: float = 1.0
    snr_threshold: float = 1.2
    max_displacement: float = 32.0
    adaptive_enabled: bool = False
    adaptive_window_sizes: Tuple[int, ...] = (64, 32, 16)
    adaptive_residual_search_radius: int = 6
    optical_flow_enabled: bool = False
    optical_flow_pyr_scale: float = 0.5
    optical_flow_levels: int = 3
    optical_flow_winsize: int = 15
    optical_flow_iterations: int = 3
    outlier_filter_enabled: bool = False
    outlier_replace_enabled: bool = False
    outlier_median_threshold: float = 3.0
    outlier_max_speed: float = 0.0
    outlier_interp_iterations: int = 5


class PIV2DCalculator:
    def __init__(self, config: Optional[PIV2DConfig] = None):
        self.config = config or PIV2DConfig()

    @staticmethod
    def _to_gray_float(image: np.ndarray) -> np.ndarray:
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image.astype(np.float32)

    def compute_velocity_field(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
        exclusion_mask: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        img1 = self._to_gray_float(frame1)
        img2 = self._to_gray_float(frame2)
        if img1.shape != img2.shape:
            raise ValueError("两帧图像尺寸必须一致")

        mask = None
        if exclusion_mask is not None:
            mask = np.asarray(exclusion_mask, dtype=bool)
            if mask.shape != img1.shape:
                raise ValueError("无粒子区域掩膜尺寸必须与图像一致")

        if self.config.optical_flow_enabled:
            result = self._compute_optical_flow_refined_velocity_field(img1, img2, mask)
            return self._postprocess_vectors(result)

        if self.config.adaptive_enabled:
            result = self._compute_adaptive_velocity_field(img1, img2, mask)
            return self._postprocess_vectors(result)

        pass_result = self._compute_displacement_pass(
            img1,
            img2,
            mask,
            int(self.config.window_size),
            int(self.config.search_radius),
        )
        result = self._make_velocity_result(pass_result, "fixed", [int(self.config.window_size)])
        return self._postprocess_vectors(result)

    def _compute_displacement_pass(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        mask: Optional[np.ndarray],
        window_size: int,
        search_radius: int,
    ) -> Dict[str, np.ndarray]:
        ws = int(window_size)
        sr = int(search_radius)
        step = max(1, int(round(ws * (1.0 - self.config.overlap_ratio))))
        if ws < 8:
            raise ValueError("窗口尺寸不能小于 8")

        h, w = img1.shape
        half = ws // 2
        centers_y = list(range(half, h - half + 1, step))
        centers_x = list(range(half, w - half + 1, step))
        if not centers_x or not centers_y:
            raise ValueError("图像太小，无法按当前窗口参数生成速度场")

        x_grid, y_grid = np.meshgrid(np.array(centers_x), np.array(centers_y))
        dx = np.zeros_like(x_grid, dtype=np.float32)
        dy = np.zeros_like(y_grid, dtype=np.float32)
        snr = np.zeros_like(x_grid, dtype=np.float32)
        valid = np.zeros_like(x_grid, dtype=bool)
        skipped_mask = np.zeros_like(x_grid, dtype=bool)

        for iy, cy in enumerate(centers_y):
            y1 = cy - half
            y2 = y1 + ws
            for ix, cx in enumerate(centers_x):
                x1 = cx - half
                x2 = x1 + ws

                if mask is not None and mask[int(cy), int(cx)]:
                    skipped_mask[iy, ix] = True
                    continue

                win1 = img1[y1:y2, x1:x2]
                sy1 = max(0, y1 - sr)
                sy2 = min(h, y2 + sr)
                sx1 = max(0, x1 - sr)
                sx2 = min(w, x2 + sr)
                search = img2[sy1:sy2, sx1:sx2]

                if search.shape[0] < ws or search.shape[1] < ws:
                    continue

                win1n = win1 - float(win1.mean())
                searchn = search - float(search.mean())
                corr = correlate2d(searchn, win1n, mode="valid")
                if corr.size == 0:
                    continue

                peak_idx = np.unravel_index(np.argmax(corr), corr.shape)
                peak_val = float(corr[peak_idx])
                second_val = self._second_peak(corr, peak_idx)
                ratio = peak_val / max(second_val, 1e-6)

                peak_y, peak_x = peak_idx
                top_left_x = sx1 + peak_x
                top_left_y = sy1 + peak_y
                dx_px = top_left_x - x1
                dy_px = top_left_y - y1
                disp_px = float(np.hypot(dx_px, dy_px))

                snr[iy, ix] = ratio
                if ratio < self.config.snr_threshold or disp_px > self.config.max_displacement:
                    continue

                dx[iy, ix] = dx_px
                dy[iy, ix] = dy_px
                valid[iy, ix] = True

        return {
            "x": x_grid.astype(np.float32),
            "y": y_grid.astype(np.float32),
            "dx": dx,
            "dy": dy,
            "snr": snr,
            "valid": valid,
            "excluded": skipped_mask,
            "grid_shape": np.array(x_grid.shape, dtype=np.int32),
        }

    def _compute_adaptive_velocity_field(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        mask: Optional[np.ndarray],
    ) -> Dict[str, np.ndarray]:
        window_sizes = self._adaptive_window_sizes()
        cumulative = None
        passes = []

        for pass_idx, ws in enumerate(window_sizes):
            if cumulative is None:
                warped2 = img2
                search_radius = int(self.config.search_radius)
            else:
                dense_u, dense_v = self._dense_displacement(cumulative, img1.shape)
                warped2 = self._warp_second_frame(img2, dense_u, dense_v)
                search_radius = int(self.config.adaptive_residual_search_radius)

            residual = self._compute_displacement_pass(img1, warped2, mask, ws, search_radius)
            if cumulative is None:
                cumulative = residual
            else:
                base_u, base_v = self._sample_displacement(
                    cumulative,
                    residual["x"],
                    residual["y"],
                )
                residual_valid = residual["valid"]
                residual["dx"] = base_u + np.where(residual_valid, residual["dx"], 0.0)
                residual["dy"] = base_v + np.where(residual_valid, residual["dy"], 0.0)
                residual["valid"] = residual_valid | self._sample_valid(cumulative, residual["x"], residual["y"])
                cumulative = residual

            passes.append({
                "window_size": int(ws),
                "valid_count": int(np.sum(cumulative["valid"])),
                "total_count": int(cumulative["valid"].size),
            })

        result = self._make_velocity_result(cumulative, "adaptive", window_sizes)
        result["adaptive_passes"] = passes
        return result

    def _compute_optical_flow_refined_velocity_field(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        mask: Optional[np.ndarray],
    ) -> Dict[str, np.ndarray]:
        prior = self._compute_adaptive_velocity_field(img1, img2, mask)
        prior_displacement = {
            "x": prior["x"],
            "y": prior["y"],
            "dx": prior["u_px"],
            "dy": prior["v_px"],
            "valid": prior["valid"],
            "excluded": prior["excluded"],
        }
        dense_u, dense_v = self._dense_displacement(prior_displacement, img1.shape)
        warped2 = self._warp_second_frame(img2, dense_u, dense_v)
        flow = self._compute_dense_optical_flow(img1, warped2)

        residual_u, residual_v = self._sample_dense_field(
            flow[..., 0],
            flow[..., 1],
            prior["x"],
            prior["y"],
        )
        refined_dx = prior["u_px"] + np.where(prior["valid"], residual_u, 0.0)
        refined_dy = prior["v_px"] + np.where(prior["valid"], residual_v, 0.0)
        refined = {
            "x": prior["x"],
            "y": prior["y"],
            "dx": refined_dx.astype(np.float32),
            "dy": refined_dy.astype(np.float32),
            "snr": prior["snr"],
            "valid": prior["valid"],
            "excluded": prior["excluded"],
        }
        result = self._make_velocity_result(refined, "adaptive+optical_flow", self._adaptive_window_sizes())
        result["adaptive_passes"] = prior.get("adaptive_passes", [])
        result["optical_flow_residual_u_px"] = residual_u.astype(np.float32)
        result["optical_flow_residual_v_px"] = residual_v.astype(np.float32)
        result["optical_flow_dense_u_px"] = flow[..., 0].astype(np.float32)
        result["optical_flow_dense_v_px"] = flow[..., 1].astype(np.float32)
        return result

    def _adaptive_window_sizes(self) -> List[int]:
        sizes = [int(s) for s in self.config.adaptive_window_sizes if int(s) >= 8]
        if not sizes:
            sizes = [int(self.config.window_size)]
        sizes = sorted(set(sizes), reverse=True)
        return sizes

    def _make_velocity_result(
        self,
        displacement: Dict[str, np.ndarray],
        algorithm: str,
        window_sizes: List[int],
    ) -> Dict[str, np.ndarray]:
        u = displacement["dx"] * self.config.pixel_scale / self.config.dt
        v = displacement["dy"] * self.config.pixel_scale / self.config.dt
        speed = np.sqrt(u ** 2 + v ** 2)
        return {
            "x": displacement["x"].astype(np.float32),
            "y": displacement["y"].astype(np.float32),
            "u": u.astype(np.float32),
            "v": v.astype(np.float32),
            "u_px": displacement["dx"].astype(np.float32),
            "v_px": displacement["dy"].astype(np.float32),
            "speed": speed.astype(np.float32),
            "snr": displacement["snr"].astype(np.float32),
            "valid": displacement["valid"],
            "excluded": displacement["excluded"],
            "grid_shape": np.array(displacement["x"].shape, dtype=np.int32),
            "algorithm": algorithm,
            "window_sizes": np.array(window_sizes, dtype=np.int32),
        }

    def _postprocess_vectors(self, result: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        valid_before = result["valid"].copy()
        result["outlier"] = np.zeros_like(result["valid"], dtype=bool)
        result["replaced"] = np.zeros_like(result["valid"], dtype=bool)
        if not self.config.outlier_filter_enabled:
            result["valid_original"] = valid_before
            return result

        outlier = self._detect_outliers(result)
        outlier &= result["valid"]
        result["outlier"] = outlier
        result["valid_original"] = valid_before

        if self.config.outlier_replace_enabled:
            fill_mask = outlier | ~result["valid"]
            keep_mask = result["valid"] & ~outlier
            u_filled = self._interpolate_invalid_vectors(result["u"], keep_mask, fill_mask)
            v_filled = self._interpolate_invalid_vectors(result["v"], keep_mask, fill_mask)
            u_px_filled = self._interpolate_invalid_vectors(result["u_px"], keep_mask, fill_mask)
            v_px_filled = self._interpolate_invalid_vectors(result["v_px"], keep_mask, fill_mask)
            filled_positions = fill_mask & np.isfinite(u_filled) & np.isfinite(v_filled)
            result["u"] = np.where(filled_positions, u_filled, result["u"]).astype(np.float32)
            result["v"] = np.where(filled_positions, v_filled, result["v"]).astype(np.float32)
            result["u_px"] = np.where(filled_positions, u_px_filled, result["u_px"]).astype(np.float32)
            result["v_px"] = np.where(filled_positions, v_px_filled, result["v_px"]).astype(np.float32)
            result["valid"] = (keep_mask | filled_positions) & ~result.get("excluded", np.zeros_like(keep_mask, dtype=bool))
            result["replaced"] = filled_positions
        else:
            result["valid"] = result["valid"] & ~outlier

        result["speed"] = np.sqrt(result["u"] ** 2 + result["v"] ** 2).astype(np.float32)
        return result

    def _detect_outliers(self, result: Dict[str, np.ndarray]) -> np.ndarray:
        valid = result["valid"]
        outlier = np.zeros_like(valid, dtype=bool)
        if self.config.outlier_max_speed > 0:
            outlier |= result["speed"] > float(self.config.outlier_max_speed)

        threshold = float(self.config.outlier_median_threshold)
        if threshold > 0 and np.any(valid):
            local_u = self._local_nanmedian(result["u"], valid)
            local_v = self._local_nanmedian(result["v"], valid)
            residual = np.sqrt((result["u"] - local_u) ** 2 + (result["v"] - local_v) ** 2)
            outlier |= residual > threshold
        return outlier

    @staticmethod
    def _local_nanmedian(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
        padded_values = np.pad(values.astype(np.float32), 1, mode="edge")
        padded_valid = np.pad(valid.astype(bool), 1, mode="constant", constant_values=False)
        neighborhoods = []
        for dy in range(3):
            for dx in range(3):
                vals = padded_values[dy:dy + values.shape[0], dx:dx + values.shape[1]]
                mask = padded_valid[dy:dy + values.shape[0], dx:dx + values.shape[1]]
                neighborhoods.append(np.where(mask, vals, np.nan))
        stacked = np.stack(neighborhoods, axis=0)
        median = np.nanmedian(stacked, axis=0)
        return np.where(np.isfinite(median), median, values)

    def _interpolate_invalid_vectors(
        self,
        values: np.ndarray,
        keep_mask: np.ndarray,
        fill_mask: np.ndarray,
    ) -> np.ndarray:
        filled = values.astype(np.float32).copy()
        known = keep_mask.astype(np.float32)
        filled[~keep_mask] = 0.0
        kernel = np.ones((3, 3), dtype=np.float32)
        iterations = max(1, int(self.config.outlier_interp_iterations))
        for _ in range(iterations):
            if not np.any(fill_mask & (known == 0)):
                break
            sum_values = cv2.filter2D(filled, -1, kernel, borderType=cv2.BORDER_REPLICATE)
            sum_weights = cv2.filter2D(known, -1, kernel, borderType=cv2.BORDER_REPLICATE)
            can_fill = fill_mask & (known == 0) & (sum_weights > 0)
            if not np.any(can_fill):
                break
            filled[can_fill] = sum_values[can_fill] / sum_weights[can_fill]
            known[can_fill] = 1.0
        return np.where(known > 0, filled, np.nan).astype(np.float32)

    def _dense_displacement(
        self,
        displacement: Dict[str, np.ndarray],
        image_shape: Tuple[int, int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        h, w = image_shape
        yy, xx = np.mgrid[0:h, 0:w]
        return self._sample_displacement(displacement, xx.astype(np.float32), yy.astype(np.float32))

    def _sample_displacement(
        self,
        displacement: Dict[str, np.ndarray],
        target_x: np.ndarray,
        target_y: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        valid = displacement["valid"]
        dx = self._fill_invalid_regular(displacement["dx"], valid)
        dy = self._fill_invalid_regular(displacement["dy"], valid)
        y_coords = displacement["y"][:, 0].astype(np.float32)
        x_coords = displacement["x"][0, :].astype(np.float32)
        if len(y_coords) < 2 or len(x_coords) < 2:
            mean_dx = float(np.mean(dx[valid])) if np.any(valid) else 0.0
            mean_dy = float(np.mean(dy[valid])) if np.any(valid) else 0.0
            return (
                np.full(target_x.shape, mean_dx, dtype=np.float32),
                np.full(target_x.shape, mean_dy, dtype=np.float32),
            )
        points = np.column_stack([target_y.ravel(), target_x.ravel()])
        interp_dx = RegularGridInterpolator(
            (y_coords, x_coords),
            dx,
            bounds_error=False,
            fill_value=0.0,
        )
        interp_dy = RegularGridInterpolator(
            (y_coords, x_coords),
            dy,
            bounds_error=False,
            fill_value=0.0,
        )
        out_dx = interp_dx(points).reshape(target_x.shape).astype(np.float32)
        out_dy = interp_dy(points).reshape(target_x.shape).astype(np.float32)
        return out_dx, out_dy

    def _sample_valid(
        self,
        displacement: Dict[str, np.ndarray],
        target_x: np.ndarray,
        target_y: np.ndarray,
    ) -> np.ndarray:
        y_coords = displacement["y"][:, 0].astype(np.float32)
        x_coords = displacement["x"][0, :].astype(np.float32)
        if len(y_coords) < 2 or len(x_coords) < 2:
            return np.full(target_x.shape, bool(np.any(displacement["valid"])), dtype=bool)
        points = np.column_stack([target_y.ravel(), target_x.ravel()])
        interp_valid = RegularGridInterpolator(
            (y_coords, x_coords),
            displacement["valid"].astype(np.float32),
            bounds_error=False,
            fill_value=0.0,
        )
        return (interp_valid(points).reshape(target_x.shape) > 0.25)

    @staticmethod
    def _fill_invalid_regular(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
        filled = values.astype(np.float32).copy()
        known = valid.astype(np.float32)
        filled[~valid] = 0.0
        if np.all(valid):
            return filled
        kernel = np.ones((3, 3), dtype=np.float32)
        for _ in range(values.size):
            if np.all(known > 0):
                break
            sum_values = cv2.filter2D(filled, -1, kernel, borderType=cv2.BORDER_REPLICATE)
            sum_weights = cv2.filter2D(known, -1, kernel, borderType=cv2.BORDER_REPLICATE)
            can_fill = (known == 0) & (sum_weights > 0)
            if not np.any(can_fill):
                break
            filled[can_fill] = sum_values[can_fill] / sum_weights[can_fill]
            known[can_fill] = 1.0
        return filled

    @staticmethod
    def _warp_second_frame(img2: np.ndarray, dense_u: np.ndarray, dense_v: np.ndarray) -> np.ndarray:
        h, w = img2.shape
        yy, xx = np.mgrid[0:h, 0:w]
        map_x = (xx.astype(np.float32) + dense_u.astype(np.float32))
        map_y = (yy.astype(np.float32) + dense_v.astype(np.float32))
        return cv2.remap(
            img2,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT101,
        )

    def _compute_dense_optical_flow(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        first = self._normalize_for_flow(img1)
        second = self._normalize_for_flow(img2)
        return cv2.calcOpticalFlowFarneback(
            first,
            second,
            None,
            pyr_scale=float(self.config.optical_flow_pyr_scale),
            levels=int(self.config.optical_flow_levels),
            winsize=int(self.config.optical_flow_winsize),
            iterations=int(self.config.optical_flow_iterations),
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        ).astype(np.float32)

    @staticmethod
    def _normalize_for_flow(image: np.ndarray) -> np.ndarray:
        image = image.astype(np.float32)
        lo = float(np.percentile(image, 1.0))
        hi = float(np.percentile(image, 99.0))
        if hi <= lo:
            return np.zeros(image.shape, dtype=np.uint8)
        normalized = np.clip((image - lo) * 255.0 / (hi - lo), 0, 255)
        return normalized.astype(np.uint8)

    @staticmethod
    def _sample_dense_field(
        dense_u: np.ndarray,
        dense_v: np.ndarray,
        target_x: np.ndarray,
        target_y: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        map_x = target_x.astype(np.float32)
        map_y = target_y.astype(np.float32)
        sampled_u = cv2.remap(
            dense_u.astype(np.float32),
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        sampled_v = cv2.remap(
            dense_v.astype(np.float32),
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        return sampled_u.astype(np.float32), sampled_v.astype(np.float32)

    @staticmethod
    def _second_peak(corr: np.ndarray, peak_idx: Tuple[int, int]) -> float:
        if corr.size <= 1:
            return 0.0
        mask = np.ones(corr.shape, dtype=bool)
        py, px = peak_idx
        y1 = max(0, py - 1)
        y2 = min(corr.shape[0], py + 2)
        x1 = max(0, px - 1)
        x2 = min(corr.shape[1], px + 2)
        mask[y1:y2, x1:x2] = False
        remaining = corr[mask]
        if remaining.size == 0:
            return 0.0
        return float(np.max(remaining))

    @staticmethod
    def render_overlay(
        image: np.ndarray,
        result: Dict[str, np.ndarray],
        stride: int = 1,
        scale: float = 0.15,
    ) -> np.ndarray:
        if image.ndim == 2:
            canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            canvas = image.copy()

        valid = result["valid"]
        speed = result["speed"]
        max_speed = float(np.max(speed[valid])) if np.any(valid) else 1.0

        for iy in range(0, result["x"].shape[0], max(1, stride)):
            for ix in range(0, result["x"].shape[1], max(1, stride)):
                if not valid[iy, ix]:
                    continue
                x = int(round(result["x"][iy, ix]))
                y = int(round(result["y"][iy, ix]))
                dx = float(result["u"][iy, ix]) * scale
                dy = float(result["v"][iy, ix]) * scale
                spd = float(speed[iy, ix])
                color = PIV2DCalculator._speed_to_bgr(spd, max_speed)
                end_pt = (int(round(x + dx)), int(round(y + dy)))
                cv2.arrowedLine(canvas, (x, y), end_pt, color, 1, tipLength=0.3)

        return canvas

    @staticmethod
    def _speed_to_bgr(speed: float, max_speed: float) -> Tuple[int, int, int]:
        norm = 0.0 if max_speed <= 0 else np.clip(speed / max_speed, 0.0, 1.0)
        lut = cv2.applyColorMap(
            np.array([[int(round(norm * 255))]], dtype=np.uint8),
            cv2.COLORMAP_JET,
        )
        b, g, r = lut[0, 0].tolist()
        return int(b), int(g), int(r)

    @staticmethod
    def summarize_result(result: Dict[str, np.ndarray]) -> Dict[str, float]:
        valid = result["valid"]
        speed = result["speed"]
        u = result["u"]
        v = result["v"]
        snr = result["snr"]
        if not np.any(valid):
            return {
                "valid_count": 0,
                "total_count": int(valid.size),
                "mean_u": 0.0,
                "mean_v": 0.0,
                "mean_speed": 0.0,
                "max_speed": 0.0,
                "mean_snr": 0.0,
            }
        return {
            "valid_count": int(np.sum(valid)),
            "total_count": int(valid.size),
            "mean_u": float(np.mean(u[valid])),
            "mean_v": float(np.mean(v[valid])),
            "mean_speed": float(np.mean(speed[valid])),
            "max_speed": float(np.max(speed[valid])),
            "mean_snr": float(np.mean(snr[valid])),
        }

    def process_batch_directory(
        self,
        src_dir: str,
        dst_dir: str,
        exclusion_mask: Optional[np.ndarray] = None,
        frame_pairs: Optional[List[Tuple[int, int]]] = None,
        progress_callback=None,
        stop_checker=None,
    ) -> Tuple[int, int, List[str]]:
        src_path = Path(src_dir)
        dst_path = Path(dst_dir)
        dst_path.mkdir(parents=True, exist_ok=True)

        files = sorted(
            [p for p in src_path.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS]
        )
        if frame_pairs is None:
            pairs = [(i, i + 1) for i in range(max(0, len(files) - 1))]
        else:
            pairs = [
                (int(i), int(j))
                for i, j in frame_pairs
                if 0 <= int(i) < len(files) and 0 <= int(j) < len(files) and int(i) != int(j)
            ]
        total_pairs = len(pairs)
        success = 0
        outputs: List[str] = []

        for pair_idx, (idx1, idx2) in enumerate(pairs):
            if stop_checker and stop_checker():
                break
            f1, f2 = files[idx1], files[idx2]
            img1 = cv2.imread(str(f1), cv2.IMREAD_UNCHANGED)
            img2 = cv2.imread(str(f2), cv2.IMREAD_UNCHANGED)
            if img1 is None or img2 is None:
                continue

            result = self.compute_velocity_field(img1, img2, exclusion_mask=exclusion_mask)
            overlay = self.render_overlay(img1, result)
            summary = self.summarize_result(result)

            stem = f"{f1.stem}_to_{f2.stem}"
            overlay_path = dst_path / f"{stem}_overlay.png"
            data_path = dst_path / f"{stem}_field.npz"
            txt_path = dst_path / f"{stem}_summary.txt"

            cv2.imwrite(str(overlay_path), overlay)
            np.savez_compressed(
                data_path,
                x=result["x"],
                y=result["y"],
                u=result["u"],
                v=result["v"],
                u_px=result.get("u_px", result["u"] * self.config.dt / self.config.pixel_scale),
                v_px=result.get("v_px", result["v"] * self.config.dt / self.config.pixel_scale),
                speed=result["speed"],
                snr=result["snr"],
                valid=result["valid"],
                valid_original=result.get("valid_original", result["valid"]),
                outlier=result.get("outlier", np.zeros_like(result["valid"], dtype=bool)),
                replaced=result.get("replaced", np.zeros_like(result["valid"], dtype=bool)),
                excluded=result.get("excluded", np.zeros_like(result["valid"], dtype=bool)),
                algorithm=np.array(result.get("algorithm", "fixed")),
                window_sizes=result.get("window_sizes", np.array([self.config.window_size], dtype=np.int32)),
                optical_flow_residual_u_px=result.get(
                    "optical_flow_residual_u_px",
                    np.zeros_like(result["valid"], dtype=np.float32),
                ),
                optical_flow_residual_v_px=result.get(
                    "optical_flow_residual_v_px",
                    np.zeros_like(result["valid"], dtype=np.float32),
                ),
            )
            with open(txt_path, "w", encoding="utf-8") as f:
                for key, value in summary.items():
                    f.write(f"{key}: {value}\n")

            outputs.append(str(overlay_path))
            success += 1
            if progress_callback:
                progress_callback(pair_idx + 1, total_pairs, f"{f1.name} -> {f2.name}")

        return success, total_pairs, outputs
