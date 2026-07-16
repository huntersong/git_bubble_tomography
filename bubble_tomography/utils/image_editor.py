"""General image editing utilities for single-image and batch processing."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

from utils.bub_analysis import BubAnalysisParams, BubAnalysisProcessor, BubAnalysisResult

MAX_ROTATE_OUTPUT_BYTES = 512 * 1024 * 1024
MAX_ROTATE_OUTPUT_SIDE = 32767
MAX_IMAGE_STEP_TEMP_BYTES = 512 * 1024 * 1024

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


def robust_imwrite(path: str, image: np.ndarray, params=None) -> bool:
    """Write images through imencode/tofile so Unicode Windows paths work."""
    try:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        extension = output.suffix.lower() or ".png"
        success, encoded = cv2.imencode(extension, image, params or [])
        if not success:
            return False
        encoded.tofile(str(output))
        return True
    except Exception:
        return False


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
class ImageAnalysisParams:
    """Shared controls for local image-analysis and particle-statistics nodes."""
    threshold: int = 128
    kernel_size: int = 15
    min_area: int = 3
    invert: bool = False


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
    bub_analysis: BubAnalysisParams = field(default_factory=BubAnalysisParams)
    segmentation: ImageAnalysisParams = field(default_factory=ImageAnalysisParams)
    focus_quality: ImageAnalysisParams = field(default_factory=ImageAnalysisParams)
    speckle_quality_map: ImageAnalysisParams = field(default_factory=ImageAnalysisParams)
    speckle_quality_time: ImageAnalysisParams = field(default_factory=ImageAnalysisParams)
    particle_counter: ImageAnalysisParams = field(default_factory=ImageAnalysisParams)
    particle_density: ImageAnalysisParams = field(default_factory=ImageAnalysisParams)
    particle_size_map: ImageAnalysisParams = field(default_factory=ImageAnalysisParams)
    particle_size_time: ImageAnalysisParams = field(default_factory=ImageAnalysisParams)
    particle_size_average: ImageAnalysisParams = field(default_factory=ImageAnalysisParams)


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
        "segmentation",
        "fft",
        "ifft",
        "focus_quality",
        "speckle_quality_map",
        "speckle_quality_time",
        "particle_counter",
        "particle_density",
        "particle_size_map",
        "particle_size_time",
        "particle_size_average",
        "bub_analysis",
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
        "segmentation": "图像分割",
        "fft": "FFT 傅里叶变换",
        "ifft": "IFFT 傅里叶逆变换",
        "focus_quality": "聚焦质量",
        "speckle_quality_map": "散斑质量图",
        "speckle_quality_time": "散斑质量随时间",
        "particle_counter": "粒子计数",
        "particle_density": "粒子播种密度",
        "particle_size_map": "粒子/散斑尺寸图",
        "particle_size_time": "粒子/散斑尺寸随时间",
        "particle_size_average": "粒子/散斑平均尺寸",
        "bub_analysis": "BubAnalysis 气泡识别",
    }

    def __init__(self, config: Optional[ImageEditConfig] = None):
        self.config = config or ImageEditConfig()
        self.last_bub_analysis_result: Optional[BubAnalysisResult] = None
        self._fft_phase: Optional[np.ndarray] = None
        self._fft_log_range = (0.0, 1.0)

    def process(
        self,
        image: np.ndarray,
        operand_image: Optional[np.ndarray] = None,
        step_order: Optional[List[str]] = None,
        step_params: Optional[List[object]] = None,
        operand_images: Optional[List[Optional[np.ndarray]]] = None,
    ) -> np.ndarray:
        self._validate_image(image, "输入图像")
        img = image.copy()
        cfg = self.config
        steps = step_order if step_order is not None else [
            step for step in self.ALL_STEPS if self._is_step_enabled(step, cfg)
        ]
        for index, step in enumerate(steps):
            original_params = None
            params_replaced = False
            if step_params is not None and index < len(step_params):
                params = step_params[index]
                if params is not None and hasattr(cfg, step):
                    original_params = getattr(cfg, step)
                    setattr(cfg, step, params)
                    params_replaced = True
            current_operand = operand_image
            if operand_images is not None and index < len(operand_images):
                current_operand = operand_images[index]
            try:
                img = self._run_step(img, step, current_operand)
            finally:
                if params_replaced:
                    setattr(cfg, step, original_params)
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
        if step == "bub_analysis":
            return cfg.bub_analysis.enabled
        if step in {
            "segmentation", "fft", "ifft", "focus_quality",
            "speckle_quality_map", "speckle_quality_time", "particle_counter",
            "particle_density", "particle_size_map", "particle_size_time",
            "particle_size_average",
        }:
            # New workflow nodes are selected explicitly through step_order.
            return False
        return False

    @staticmethod
    def _validate_image(img: np.ndarray, name: str = "图像") -> None:
        if img is None:
            raise ValueError(f"{name}为空，无法处理。")
        if not isinstance(img, np.ndarray):
            raise TypeError(f"{name}不是有效的 numpy 图像数组。")
        if img.size == 0 or img.ndim not in (2, 3):
            raise ValueError(f"{name}尺寸无效。")
        if img.ndim == 3 and img.shape[2] not in (1, 3, 4):
            raise ValueError(f"{name}通道数不支持: {img.shape[2]}。")

    @staticmethod
    def _estimate_bytes(shape, dtype) -> int:
        return int(np.prod(shape)) * np.dtype(dtype).itemsize

    @staticmethod
    def _check_temp_budget(bytes_needed: int, step_label: str) -> None:
        if bytes_needed > MAX_IMAGE_STEP_TEMP_BYTES:
            size_mb = bytes_needed / (1024 * 1024)
            limit_mb = MAX_IMAGE_STEP_TEMP_BYTES / (1024 * 1024)
            raise ValueError(
                f"{step_label}预计临时内存约 {size_mb:.0f} MB，超过安全上限 "
                f"{limit_mb:.0f} MB。请先裁剪/缩小图像，或分批处理。"
            )

    @staticmethod
    def _to_gray(img: np.ndarray, copy: bool = True) -> np.ndarray:
        ImageEditor._validate_image(img)
        if img.ndim == 2:
            return img.copy() if copy else img
        channels = img.shape[2]
        if channels == 1:
            gray = img[:, :, 0]
            return gray.copy() if copy else gray
        if channels == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if channels == 4:
            return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        raise ValueError(f"不支持的图像通道数: {channels}")

    @staticmethod
    def _to_bgr(img: np.ndarray) -> np.ndarray:
        ImageEditor._validate_image(img)
        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        channels = img.shape[2]
        if channels == 1:
            return cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2BGR)
        if channels == 3:
            return img.copy()
        if channels == 4:
            return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        raise ValueError(f"不支持的图像通道数: {channels}")

    @staticmethod
    def _match_operand_channels(src: np.ndarray, op: np.ndarray) -> np.ndarray:
        if src.ndim == 2:
            return ImageEditor._to_gray(op)
        src_channels = src.shape[2]
        if src_channels == 1:
            return ImageEditor._to_gray(op)[:, :, None]
        if src_channels == 3:
            return ImageEditor._to_bgr(op)
        if src_channels == 4:
            bgr = ImageEditor._to_bgr(op)
            alpha = np.full(src.shape[:2] + (1,), 255, dtype=bgr.dtype)
            return np.dstack((bgr, alpha))
        raise ValueError(f"不支持的源图像通道数: {src_channels}")

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
        if step == "segmentation":
            return self._apply_segmentation(img, cfg.segmentation)
        if step == "fft":
            return self._apply_fft(img)
        if step == "ifft":
            return self._apply_ifft(img)
        if step == "focus_quality":
            return self._apply_focus_quality(img, cfg.focus_quality)
        if step == "speckle_quality_map":
            return self._apply_speckle_quality_map(img, cfg.speckle_quality_map)
        if step == "speckle_quality_time":
            return self._apply_speckle_quality_time(img, cfg.speckle_quality_time)
        if step == "particle_counter":
            return self._apply_particle_counter(img, cfg.particle_counter)
        if step == "particle_density":
            return self._apply_particle_density(img, cfg.particle_density)
        if step == "particle_size_map":
            return self._apply_particle_size_map(img, cfg.particle_size_map)
        if step == "particle_size_time":
            return self._apply_particle_size_time(img, cfg.particle_size_time)
        if step == "particle_size_average":
            return self._apply_particle_size_average(img, cfg.particle_size_average)
        if step == "bub_analysis":
            self.last_bub_analysis_result = BubAnalysisProcessor(
                cfg.bub_analysis
            ).process(img)
            return self.last_bub_analysis_result.overlay
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
        return ImageEditor._to_gray(img)

    @staticmethod
    def _apply_mirror(img: np.ndarray, p: MirrorParams) -> np.ndarray:
        ImageEditor._validate_image(img)
        ImageEditor._check_temp_budget(img.nbytes, "图像镜像")
        flip_code = 1
        if p.mode == "vertical":
            flip_code = 0
        elif p.mode == "both":
            flip_code = -1
        return cv2.flip(img, flip_code)

    @staticmethod
    def _apply_rotate(img: np.ndarray, p: RotateParams) -> np.ndarray:
        if img is None or img.size == 0:
            return img
        if p.mode == "cw90":
            return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        if p.mode == "ccw90":
            return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        if p.mode == "180":
            return cv2.rotate(img, cv2.ROTATE_180)

        h, w = img.shape[:2]
        angle = float(p.angle)
        if not np.isfinite(angle):
            angle = 0.0
        angle = ((angle + 180.0) % 360.0) - 180.0
        if abs(angle) < 1e-9:
            return img.copy()

        center = (w / 2.0, h / 2.0)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        out_w, out_h = w, h
        if p.expand:
            cos = abs(matrix[0, 0])
            sin = abs(matrix[0, 1])
            out_w = max(1, int(np.ceil((h * sin) + (w * cos))))
            out_h = max(1, int(np.ceil((h * cos) + (w * sin))))
            matrix[0, 2] += (out_w / 2.0) - center[0]
            matrix[1, 2] += (out_h / 2.0) - center[1]
        else:
            out_w = max(1, int(out_w))
            out_h = max(1, int(out_h))

        channels = img.shape[2] if img.ndim == 3 else 1
        output_bytes = int(out_w) * int(out_h) * channels * img.dtype.itemsize
        if (
            out_w > MAX_ROTATE_OUTPUT_SIDE
            or out_h > MAX_ROTATE_OUTPUT_SIDE
            or output_bytes > MAX_ROTATE_OUTPUT_BYTES
        ):
            size_mb = output_bytes / (1024 * 1024)
            raise ValueError(
                "自定义旋转输出尺寸过大: "
                f"{out_w}x{out_h}, 约 {size_mb:.0f} MB。"
                "请关闭自动扩展画布，或先裁剪/缩小图像后再旋转。"
            )

        if np.issubdtype(img.dtype, np.integer):
            info = np.iinfo(img.dtype)
            border_value = int(np.clip(p.border_value, info.min, info.max))
        else:
            border_value = float(p.border_value)
        border = tuple([border_value] * min(channels, 4)) if img.ndim == 3 else border_value
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
        ImageEditor._validate_image(img)
        if p.source_bits == 24 and img.ndim == 3:
            return ImageEditor._to_gray(img)

        if img.dtype == np.uint8:
            return ImageEditor._to_gray(img) if img.ndim == 3 else img.copy()

        if p.source_bits in (12, 16):
            max_value = float((1 << p.source_bits) - 1)
        elif img.dtype == np.uint16:
            max_value = 65535.0
        elif np.issubdtype(img.dtype, np.integer):
            info = np.iinfo(img.dtype)
            max_value = float(info.max)
        else:
            ImageEditor._check_temp_budget(
                ImageEditor._estimate_bytes(img.shape, np.float32) + img.size,
                "转 8 位图",
            )
            data = np.nan_to_num(img.astype(np.float32), copy=False)
            max_value = float(np.nanmax(data)) if data.size else 255.0

        if max_value <= 0:
            out_shape = img.shape[:2] if img.ndim == 3 else img.shape
            return np.zeros(out_shape, dtype=np.uint8)
        if np.issubdtype(img.dtype, np.integer):
            result = cv2.convertScaleAbs(img, alpha=255.0 / max_value, beta=0)
        else:
            result = np.clip(data * 255.0 / max_value, 0, 255).astype(np.uint8)
        if result.ndim == 3:
            result = ImageEditor._to_gray(result)
        return result

    @staticmethod
    def _apply_gray_math(img: np.ndarray, p: GrayMathParams) -> np.ndarray:
        gray = ImageEditor._to_gray(img)
        gray8 = ImageEditor._apply_bit_depth_to_8bit(gray, BitDepthParams())
        if p.operation == "average":
            k = max(1, p.kernel_size)
            if k % 2 == 0:
                k += 1
            return cv2.blur(gray8, (k, k))

        ImageEditor._check_temp_budget(
            ImageEditor._estimate_bytes(gray8.shape, np.float32) * 2,
            "灰度值计算",
        )
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
        ImageEditor._validate_image(img)
        ImageEditor._check_temp_budget(img.nbytes, "亮度/对比度")
        alpha = float(p.alpha)
        beta = float(p.beta)
        if not np.isfinite(alpha):
            alpha = 1.0
        if not np.isfinite(beta):
            beta = 0.0
        alpha = float(np.clip(alpha, -100.0, 100.0))
        beta = float(np.clip(beta, -100000.0, 100000.0))
        return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    @staticmethod
    def _apply_arithmetic(
        img: np.ndarray,
        p: ArithmeticParams,
        operand_image: Optional[np.ndarray],
    ) -> np.ndarray:
        ImageEditor._validate_image(img)
        ImageEditor._check_temp_budget(
            ImageEditor._estimate_bytes(img.shape, np.float32) * 3,
            "图像加/减法",
        )
        src = img.astype(np.float32)
        if operand_image is not None:
            ImageEditor._validate_image(operand_image, "运算图像")
            op_img = operand_image
            if op_img.shape[:2] != img.shape[:2]:
                op_img = cv2.resize(op_img, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_AREA)
            op_img = ImageEditor._match_operand_channels(img, op_img)
            op = op_img.astype(np.float32)
        else:
            op = np.float32(p.scalar_value)

        if p.operation == "add":
            result = src + op
        elif p.operation == "subtract":
            result = src - op
        else:
            result = src
        return np.clip(result, 0, 255).astype(np.uint8)

    @staticmethod
    def _apply_threshold(img: np.ndarray, p: ThresholdParams) -> np.ndarray:
        gray = ImageEditor._to_gray(img)
        gray8 = ImageEditor._apply_bit_depth_to_8bit(gray, BitDepthParams())
        max_value = int(np.clip(p.max_value, 0, 255))
        threshold_value = int(np.clip(p.threshold_value, 0, 255))

        if p.mode == "global":
            _, result = cv2.threshold(gray8, threshold_value, max_value, cv2.THRESH_BINARY)
        elif p.mode == "otsu":
            _, result = cv2.threshold(gray8, 0, max_value, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif p.mode == "adaptive_mean":
            bs = max(3, int(p.block_size))
            bs = bs if bs % 2 == 1 else bs + 1
            result = cv2.adaptiveThreshold(
                gray8, max_value, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, bs, p.C
            )
        elif p.mode == "adaptive_gaussian":
            bs = max(3, int(p.block_size))
            bs = bs if bs % 2 == 1 else bs + 1
            result = cv2.adaptiveThreshold(
                gray8, max_value, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, bs, p.C
            )
        else:
            result = gray8
        return result

    @staticmethod
    def _analysis_gray8(img: np.ndarray) -> np.ndarray:
        gray = ImageEditor._to_gray(img)
        return ImageEditor._apply_bit_depth_to_8bit(gray, BitDepthParams())

    @staticmethod
    def _odd_kernel(value: int, minimum: int = 3) -> int:
        value = max(minimum, int(value))
        return value if value % 2 else value + 1

    @staticmethod
    def _normalize_u8(data: np.ndarray) -> np.ndarray:
        data = np.nan_to_num(np.asarray(data, dtype=np.float32), copy=False)
        low = float(np.min(data)) if data.size else 0.0
        high = float(np.max(data)) if data.size else 0.0
        if high <= low:
            return np.zeros(data.shape, dtype=np.uint8)
        return np.clip((data - low) * (255.0 / (high - low)), 0, 255).astype(np.uint8)

    @staticmethod
    def _apply_segmentation(img: np.ndarray, p: ImageAnalysisParams) -> np.ndarray:
        gray = ImageEditor._analysis_gray8(img)
        mode = cv2.THRESH_BINARY_INV if p.invert else cv2.THRESH_BINARY
        _, mask = cv2.threshold(gray, int(np.clip(p.threshold, 0, 255)), 255, mode)
        return mask

    def _apply_fft(self, img: np.ndarray) -> np.ndarray:
        gray = self._analysis_gray8(img).astype(np.float32)
        self._check_temp_budget(gray.size * 32, "FFT 傅里叶变换")
        spectrum = np.fft.fftshift(np.fft.fft2(gray))
        self._fft_phase = np.angle(spectrum).astype(np.float32)
        log_magnitude = np.log1p(np.abs(spectrum)).astype(np.float32)
        self._fft_log_range = (
            float(np.min(log_magnitude)), float(np.max(log_magnitude))
        )
        return self._normalize_u8(log_magnitude)

    def _apply_ifft(self, img: np.ndarray) -> np.ndarray:
        amplitude_image = self._analysis_gray8(img).astype(np.float32)
        if self._fft_phase is not None and self._fft_phase.shape == amplitude_image.shape:
            low, high = self._fft_log_range
            log_magnitude = low + (amplitude_image / 255.0) * max(high - low, 0.0)
            magnitude = np.expm1(log_magnitude)
            spectrum = magnitude * np.exp(1j * self._fft_phase)
        else:
            spectrum = amplitude_image.astype(np.complex64)
        self._check_temp_budget(spectrum.nbytes * 2, "IFFT 傅里叶逆变换")
        reconstructed = np.real(np.fft.ifft2(np.fft.ifftshift(spectrum)))
        return self._normalize_u8(reconstructed)

    @staticmethod
    def _apply_focus_quality(img: np.ndarray, p: ImageAnalysisParams) -> np.ndarray:
        ImageEditor._check_temp_budget(img.shape[0] * img.shape[1] * 20, "聚焦质量")
        gray = ImageEditor._analysis_gray8(img).astype(np.float32)
        laplacian = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
        energy = laplacian * laplacian
        local = cv2.GaussianBlur(
            energy, (ImageEditor._odd_kernel(p.kernel_size),) * 2, 0
        )
        return cv2.applyColorMap(ImageEditor._normalize_u8(local), cv2.COLORMAP_TURBO)

    @staticmethod
    def _speckle_quality(gray: np.ndarray, kernel_size: int) -> np.ndarray:
        ImageEditor._check_temp_budget(gray.size * 24, "散斑质量")
        source = gray.astype(np.float32)
        kernel = (ImageEditor._odd_kernel(kernel_size),) * 2
        mean = cv2.blur(source, kernel)
        mean_sq = cv2.blur(source * source, kernel)
        deviation = np.sqrt(np.maximum(mean_sq - mean * mean, 0.0))
        return deviation / np.maximum(mean, 1.0)

    @staticmethod
    def _apply_speckle_quality_map(img: np.ndarray, p: ImageAnalysisParams) -> np.ndarray:
        gray = ImageEditor._analysis_gray8(img)
        quality = ImageEditor._speckle_quality(gray, p.kernel_size)
        return cv2.applyColorMap(ImageEditor._normalize_u8(quality), cv2.COLORMAP_VIRIDIS)

    @staticmethod
    def _metric_image(img: np.ndarray, title: str, value: float, unit: str = "") -> np.ndarray:
        if img.dtype == np.uint8:
            canvas = ImageEditor._to_bgr(img)
        else:
            canvas = ImageEditor._to_bgr(ImageEditor._analysis_gray8(img))
        overlay = canvas.copy()
        cv2.rectangle(overlay, (12, 12), (min(canvas.shape[1] - 1, 430), 82), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.65, canvas, 0.35, 0, canvas)
        cv2.putText(canvas, title, (24, 39), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (80, 220, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"{value:.3f}{unit}", (24, 69), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
        return canvas

    @staticmethod
    def _apply_speckle_quality_time(img: np.ndarray, p: ImageAnalysisParams) -> np.ndarray:
        gray = ImageEditor._analysis_gray8(img)
        value = float(np.mean(ImageEditor._speckle_quality(gray, p.kernel_size)))
        return ImageEditor._metric_image(img, "Speckle quality", value)

    @staticmethod
    def _particle_mask(img: np.ndarray, p: ImageAnalysisParams) -> np.ndarray:
        gray = ImageEditor._analysis_gray8(img)
        mode = cv2.THRESH_BINARY_INV if p.invert else cv2.THRESH_BINARY
        _, mask = cv2.threshold(gray, int(np.clip(p.threshold, 0, 255)), 255, mode)
        return mask

    @staticmethod
    def _particle_components(img: np.ndarray, p: ImageAnalysisParams):
        ImageEditor._check_temp_budget(img.shape[0] * img.shape[1] * 12, "粒子统计")
        mask = ImageEditor._particle_mask(img, p)
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
        valid = [i for i in range(1, count) if int(stats[i, cv2.CC_STAT_AREA]) >= max(1, p.min_area)]
        return mask, labels, stats, centroids, valid

    @staticmethod
    def _apply_particle_counter(img: np.ndarray, p: ImageAnalysisParams) -> np.ndarray:
        _, _, stats, centroids, valid = ImageEditor._particle_components(img, p)
        output = ImageEditor._to_bgr(ImageEditor._analysis_gray8(img))
        for number, index in enumerate(valid, 1):
            x, y, w, h, _ = stats[index]
            cv2.rectangle(output, (x, y), (x + w - 1, y + h - 1), (0, 255, 80), 1)
            cx, cy = centroids[index]
            cv2.putText(output, str(number), (int(cx), int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 220, 255), 1, cv2.LINE_AA)
        return ImageEditor._metric_image(output, "Particle count", float(len(valid)))

    @staticmethod
    def _apply_particle_density(img: np.ndarray, p: ImageAnalysisParams) -> np.ndarray:
        mask = ImageEditor._particle_mask(img, p).astype(np.float32) / 255.0
        density = cv2.blur(mask, (ImageEditor._odd_kernel(p.kernel_size),) * 2)
        return cv2.applyColorMap(np.clip(density * 255.0, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)

    @staticmethod
    def _particle_size_field(img: np.ndarray, p: ImageAnalysisParams):
        _, labels, stats, _, valid = ImageEditor._particle_components(img, p)
        field = np.zeros(labels.shape, dtype=np.float32)
        diameters = []
        for index in valid:
            area = float(stats[index, cv2.CC_STAT_AREA])
            diameter = float(np.sqrt(4.0 * area / np.pi))
            field[labels == index] = diameter
            diameters.append(diameter)
        return field, diameters

    @staticmethod
    def _apply_particle_size_map(img: np.ndarray, p: ImageAnalysisParams) -> np.ndarray:
        field, _ = ImageEditor._particle_size_field(img, p)
        return cv2.applyColorMap(ImageEditor._normalize_u8(field), cv2.COLORMAP_TURBO)

    @staticmethod
    def _apply_particle_size_time(img: np.ndarray, p: ImageAnalysisParams) -> np.ndarray:
        _, diameters = ImageEditor._particle_size_field(img, p)
        value = float(np.mean(diameters)) if diameters else 0.0
        return ImageEditor._metric_image(img, "Mean particle size", value, " px")

    @staticmethod
    def _apply_particle_size_average(img: np.ndarray, p: ImageAnalysisParams) -> np.ndarray:
        _, diameters = ImageEditor._particle_size_field(img, p)
        value = float(np.mean(diameters)) if diameters else 0.0
        return ImageEditor._metric_image(img, "Average particle size", value, " px")

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
            return robust_imwrite(dst_path, result)
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
