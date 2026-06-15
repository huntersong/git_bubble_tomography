"""General image editing utilities for single-image and batch processing."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# 鲁棒图像加载：cv2.imread 无法处理某些 TIFF（例：12-bit/非标准BitsPerSample），
# 此时回退到 PIL 读取。
# ---------------------------------------------------------------------------

def robust_imread(path: str, flags: int = cv2.IMREAD_UNCHANGED) -> "Optional[np.ndarray]":
    """读取图像，优先使用 cv2，失败时回退到 PIL（兼容非标准位深 TIFF）。

    参数:
        path:  图像文件路径
        flags: cv2 读取标志（默认为 IMREAD_UNCHANGED）
               - IMREAD_UNCHANGED: 保持原始位深和通道
               - IMREAD_GRAYSCALE: 强制灰度
               - IMREAD_COLOR:     强制 BGR 彩色

    返回:
        numpy 数组，失败时返回 None
    """
    img = cv2.imread(path, flags)
    if img is not None:
        return img

    # ---- cv2 失败，回退到 PIL ----
    try:
        from PIL import Image
        pil_img = Image.open(path)
        arr = np.array(pil_img)

        if flags == cv2.IMREAD_COLOR or flags == cv2.IMREAD_ANYCOLOR:
            # 需要 3 通道 BGR
            if arr.ndim == 2:
                return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
            if arr.shape[2] == 4:
                return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

        if flags == cv2.IMREAD_GRAYSCALE:
            if arr.ndim == 3:
                return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            return arr

        # IMREAD_UNCHANGED：与 OpenCV 行为一致
        #   - 灰度图：原样返回
        #   - 彩色 3 通道：PIL→RGB，转 BGR 以匹配 OpenCV 惯例
        #   - 彩色 4 通道：PIL→RGBA，转 BGRA 以匹配 OpenCV 惯例
        if arr.ndim == 3 and arr.shape[2] == 3:
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        if arr.ndim == 3 and arr.shape[2] == 4:
            return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)
        return arr
    except Exception:
        return None


@dataclass
class CropParams:
    enabled: bool = False
    x: int = 0
    y: int = 0
    w: int = 0
    h: int = 0


@dataclass
class GrayParams:
    enabled: bool = False


@dataclass
class MirrorParams:
    enabled: bool = False
    mode: str = "horizontal"


@dataclass
class RotateParams:
    enabled: bool = False
    mode: str = "cw90"
    angle: float = 0.0
    expand: bool = True
    border_value: int = 0


@dataclass
class BitDepthParams:
    enabled: bool = False
    source_bits: int = 0


@dataclass
class GrayMathParams:
    enabled: bool = False
    operation: str = "average"
    kernel_size: int = 3


@dataclass
class BrightnessContrastParams:
    enabled: bool = False
    alpha: float = 1.0
    beta: int = 0


@dataclass
class ArithmeticParams:
    enabled: bool = False
    operation: str = "none"
    operand_path: str = ""
    scalar_value: int = 0


@dataclass
class ThresholdParams:
    enabled: bool = False
    mode: str = "global"
    threshold_value: int = 128
    max_value: int = 255
    block_size: int = 11
    C: int = 2


@dataclass
class ImageEditConfig:
    crop: CropParams = field(default_factory=CropParams)
    gray: GrayParams = field(default_factory=GrayParams)
    mirror: MirrorParams = field(default_factory=MirrorParams)
    rotate: RotateParams = field(default_factory=RotateParams)
    bit_depth: BitDepthParams = field(default_factory=BitDepthParams)
    gray_math: GrayMathParams = field(default_factory=GrayMathParams)
    bc: BrightnessContrastParams = field(default_factory=BrightnessContrastParams)
    arithmetic: ArithmeticParams = field(default_factory=ArithmeticParams)
    threshold: ThresholdParams = field(default_factory=ThresholdParams)


SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class ImageEditor:
    """Stateless image processor. process() returns a new image array."""

    ALL_STEPS = [
        "crop",
        "gray",
        "mirror",
        "rotate",
        "bit_depth",
        "gray_math",
        "bc",
        "arithmetic",
        "threshold",
    ]

    STEP_LABELS = {
        "crop": "裁剪 (ROI)",
        "gray": "灰度转换",
        "mirror": "图像镜像",
        "rotate": "图像旋转",
        "bit_depth": "转 8 位图",
        "gray_math": "灰度值计算",
        "bc": "亮度/对比度",
        "arithmetic": "图像加/减法",
        "threshold": "阈值化",
    }

    def __init__(self, config: Optional[ImageEditConfig] = None):
        self.config = config or ImageEditConfig()

    def process(
        self,
        image: np.ndarray,
        operand_image: Optional[np.ndarray] = None,
        step_order: Optional[List[str]] = None,
    ) -> np.ndarray:
        img = image.copy()
        cfg = self.config
        steps = step_order if step_order is not None else [
            step for step in self.ALL_STEPS if self._is_step_enabled(step, cfg)
        ]
        for step in steps:
            img = self._run_step(img, step, operand_image)
        return img

    def _is_step_enabled(self, step: str, cfg: ImageEditConfig) -> bool:
        if step == "crop":
            return cfg.crop.enabled
        if step == "gray":
            return cfg.gray.enabled
        if step == "mirror":
            return cfg.mirror.enabled
        if step == "rotate":
            return cfg.rotate.enabled
        if step == "bit_depth":
            return cfg.bit_depth.enabled
        if step == "gray_math":
            return cfg.gray_math.enabled
        if step == "bc":
            return cfg.bc.enabled
        if step == "arithmetic":
            return cfg.arithmetic.enabled and cfg.arithmetic.operation != "none"
        if step == "threshold":
            return cfg.threshold.enabled
        return False

    def _run_step(
        self,
        img: np.ndarray,
        step: str,
        operand_image: Optional[np.ndarray],
    ) -> np.ndarray:
        cfg = self.config
        if step == "crop":
            return self._apply_crop(img, cfg.crop)
        if step == "gray":
            return self._apply_gray(img)
        if step == "mirror":
            return self._apply_mirror(img, cfg.mirror)
        if step == "rotate":
            return self._apply_rotate(img, cfg.rotate)
        if step == "bit_depth":
            return self._apply_bit_depth_to_8bit(img, cfg.bit_depth)
        if step == "gray_math":
            return self._apply_gray_math(img, cfg.gray_math)
        if step == "bc":
            return self._apply_brightness_contrast(img, cfg.bc)
        if step == "arithmetic":
            return self._apply_arithmetic(img, cfg.arithmetic, operand_image)
        if step == "threshold":
            return self._apply_threshold(img, cfg.threshold)
        return img

    @staticmethod
    def _apply_crop(img: np.ndarray, p: CropParams) -> np.ndarray:
        h, w = img.shape[:2]
        x1 = max(0, p.x)
        y1 = max(0, p.y)
        x2 = min(w, x1 + p.w) if p.w > 0 else w
        y2 = min(h, y1 + p.h) if p.h > 0 else h
        if x2 <= x1 or y2 <= y1:
            return img
        return img[y1:y2, x1:x2]

    @staticmethod
    def _apply_gray(img: np.ndarray) -> np.ndarray:
        if img.ndim == 2:
            return img
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def _apply_mirror(img: np.ndarray, p: MirrorParams) -> np.ndarray:
        flip_code = 1
        if p.mode == "vertical":
            flip_code = 0
        elif p.mode == "both":
            flip_code = -1
        return cv2.flip(img, flip_code)

    @staticmethod
    def _apply_rotate(img: np.ndarray, p: RotateParams) -> np.ndarray:
        if p.mode == "cw90":
            return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        if p.mode == "ccw90":
            return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        if p.mode == "180":
            return cv2.rotate(img, cv2.ROTATE_180)

        h, w = img.shape[:2]
        center = (w / 2.0, h / 2.0)
        matrix = cv2.getRotationMatrix2D(center, p.angle, 1.0)
        out_w, out_h = w, h
        if p.expand:
            cos = abs(matrix[0, 0])
            sin = abs(matrix[0, 1])
            out_w = int((h * sin) + (w * cos))
            out_h = int((h * cos) + (w * sin))
            matrix[0, 2] += (out_w / 2.0) - center[0]
            matrix[1, 2] += (out_h / 2.0) - center[1]
        border = [p.border_value] * img.shape[2] if img.ndim == 3 else p.border_value
        return cv2.warpAffine(
            img,
            matrix,
            (out_w, out_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=border,
        )

    @staticmethod
    def _apply_bit_depth_to_8bit(img: np.ndarray, p: BitDepthParams) -> np.ndarray:
        if p.source_bits == 24 and img.ndim == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        data = img.astype(np.float32)
        if p.source_bits in (12, 16):
            max_value = float((1 << p.source_bits) - 1)
        elif img.dtype == np.uint16:
            max_value = 65535.0
        elif img.dtype == np.uint8:
            if img.ndim == 3:
                return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return img.copy()
        else:
            max_value = float(np.nanmax(data)) if data.size else 255.0

        if max_value <= 0:
            out_shape = img.shape[:2] if img.ndim == 3 else img.shape
            return np.zeros(out_shape, dtype=np.uint8)
        result = np.clip(data * 255.0 / max_value, 0, 255).astype(np.uint8)
        if result.ndim == 3:
            result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        return result

    @staticmethod
    def _apply_gray_math(img: np.ndarray, p: GrayMathParams) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
        gray8 = ImageEditor._apply_bit_depth_to_8bit(gray, BitDepthParams())
        if p.operation == "average":
            k = max(1, p.kernel_size)
            if k % 2 == 0:
                k += 1
            return cv2.blur(gray8, (k, k))

        src = gray8.astype(np.float32) / 255.0
        if p.operation == "log":
            dst = np.log1p(src) / np.log(2.0)
        elif p.operation == "exp":
            dst = np.expm1(src) / (np.e - 1.0)
        elif p.operation == "sqrt":
            dst = np.sqrt(src)
        elif p.operation == "sqr":
            dst = np.square(src)
        else:
            dst = src
        return np.clip(dst * 255.0, 0, 255).astype(np.uint8)

    @staticmethod
    def _apply_brightness_contrast(
        img: np.ndarray,
        p: BrightnessContrastParams,
    ) -> np.ndarray:
        return cv2.convertScaleAbs(img, alpha=p.alpha, beta=p.beta)

    @staticmethod
    def _apply_arithmetic(
        img: np.ndarray,
        p: ArithmeticParams,
        operand_image: Optional[np.ndarray],
    ) -> np.ndarray:
        src = img.astype(np.float32)
        if operand_image is not None:
            op = operand_image.astype(np.float32)
            if op.shape[:2] != src.shape[:2]:
                op = cv2.resize(op, (src.shape[1], src.shape[0]))
            if src.ndim == 2 and op.ndim == 3:
                op = cv2.cvtColor(op, cv2.COLOR_BGR2GRAY).astype(np.float32)
            elif src.ndim == 3 and op.ndim == 2:
                op = cv2.cvtColor(op, cv2.COLOR_GRAY2BGR).astype(np.float32)
        else:
            op = np.float32(p.scalar_value)

        result = src + op if p.operation == "add" else src - op
        return np.clip(result, 0, 255).astype(np.uint8)

    @staticmethod
    def _apply_threshold(img: np.ndarray, p: ThresholdParams) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
        gray8 = ImageEditor._apply_bit_depth_to_8bit(gray, BitDepthParams())

        if p.mode == "global":
            _, result = cv2.threshold(gray8, p.threshold_value, p.max_value, cv2.THRESH_BINARY)
        elif p.mode == "otsu":
            _, result = cv2.threshold(gray8, 0, p.max_value, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif p.mode == "adaptive_mean":
            bs = p.block_size if p.block_size % 2 == 1 else p.block_size + 1
            result = cv2.adaptiveThreshold(
                gray8, p.max_value, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, bs, p.C
            )
        elif p.mode == "adaptive_gaussian":
            bs = p.block_size if p.block_size % 2 == 1 else p.block_size + 1
            result = cv2.adaptiveThreshold(
                gray8, p.max_value, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, bs, p.C
            )
        else:
            result = gray8
        return result

    def process_single_file(
        self,
        src_path: str,
        dst_path: str,
        operand_path: Optional[str] = None,
    ) -> bool:
        try:
            img = robust_imread(src_path)
            if img is None:
                return False
            op_img = None
            if operand_path and os.path.isfile(operand_path):
                op_img = robust_imread(operand_path)
            result = self.process(img, op_img)
            os.makedirs(os.path.dirname(dst_path) or ".", exist_ok=True)
            cv2.imwrite(dst_path, result)
            return True
        except Exception:
            return False

    def process_directory(
        self,
        src_dir: str,
        dst_dir: str,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> Tuple[int, int]:
        files = [
            f for f in Path(src_dir).iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS
        ]
        total = len(files)
        success = 0
        for i, f in enumerate(files):
            dst = str(Path(dst_dir) / f.name)
            if self.process_single_file(str(f), dst):
                success += 1
            if progress_callback:
                progress_callback(i + 1, total, f.name)
        return success, total

    @staticmethod
    def to_qimage_compatible(img: np.ndarray) -> np.ndarray:
        """Convert any image to a uint8 BGR array suitable for QImage display.
        
        Non-uint8 images are normalized to 0-255 using min-max scaling,
        preserving the full dynamic range for preview.
        """
        if img.dtype != np.uint8:
            vmin, vmax = float(img.min()), float(img.max())
            if vmax > vmin:
                img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            else:
                img = np.zeros(img.shape[:2], dtype=np.uint8)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img
