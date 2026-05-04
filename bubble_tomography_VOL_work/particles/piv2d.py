"""
二维 PIV 互相关计算模块
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
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
    ) -> Dict[str, np.ndarray]:
        img1 = self._to_gray_float(frame1)
        img2 = self._to_gray_float(frame2)
        if img1.shape != img2.shape:
            raise ValueError("两帧图像尺寸必须一致")

        ws = int(self.config.window_size)
        sr = int(self.config.search_radius)
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
        u = np.zeros_like(x_grid, dtype=np.float32)
        v = np.zeros_like(y_grid, dtype=np.float32)
        snr = np.zeros_like(x_grid, dtype=np.float32)
        valid = np.zeros_like(x_grid, dtype=bool)

        for iy, cy in enumerate(centers_y):
            y1 = cy - half
            y2 = y1 + ws
            for ix, cx in enumerate(centers_x):
                x1 = cx - half
                x2 = x1 + ws

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

                u[iy, ix] = dx_px * self.config.pixel_scale / self.config.dt
                v[iy, ix] = dy_px * self.config.pixel_scale / self.config.dt
                valid[iy, ix] = True

        speed = np.sqrt(u ** 2 + v ** 2)
        return {
            "x": x_grid.astype(np.float32),
            "y": y_grid.astype(np.float32),
            "u": u,
            "v": v,
            "speed": speed,
            "snr": snr,
            "valid": valid,
            "grid_shape": np.array(x_grid.shape, dtype=np.int32),
        }

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
        progress_callback=None,
        stop_checker=None,
    ) -> Tuple[int, int, List[str]]:
        src_path = Path(src_dir)
        dst_path = Path(dst_dir)
        dst_path.mkdir(parents=True, exist_ok=True)

        files = sorted(
            [p for p in src_path.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS]
        )
        total_pairs = max(0, len(files) - 1)
        success = 0
        outputs: List[str] = []

        for i in range(total_pairs):
            if stop_checker and stop_checker():
                break
            f1, f2 = files[i], files[i + 1]
            img1 = cv2.imread(str(f1), cv2.IMREAD_UNCHANGED)
            img2 = cv2.imread(str(f2), cv2.IMREAD_UNCHANGED)
            if img1 is None or img2 is None:
                continue

            result = self.compute_velocity_field(img1, img2)
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
                speed=result["speed"],
                snr=result["snr"],
                valid=result["valid"],
            )
            with open(txt_path, "w", encoding="utf-8") as f:
                for key, value in summary.items():
                    f.write(f"{key}: {value}\n")

            outputs.append(str(overlay_path))
            success += 1
            if progress_callback:
                progress_callback(i + 1, total_pairs, f"{f1.name} -> {f2.name}")

        return success, total_pairs, outputs
