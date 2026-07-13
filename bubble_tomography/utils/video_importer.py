"""Video import and frame export helpers for BubbleTomography."""

from __future__ import annotations

import math
import os
import struct
import zlib
import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Tuple

import cv2
import numpy as np


VIDEO_EXTS = {
    ".cine", ".mp4", ".mov", ".avi", ".mkv", ".mxf", ".wmv", ".mpg",
    ".mpeg", ".m4v",
}

_P10_LIN_LUT_B64 = (
    "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"
)


def _p10_lut() -> np.ndarray:
    if not hasattr(_p10_lut, "_lut"):
        raw = zlib.decompress(base64.b64decode(_P10_LIN_LUT_B64))
        _p10_lut._lut = np.frombuffer(raw, dtype="<u2")
    return _p10_lut._lut


def linearize_p10_to_12bit(frame: np.ndarray) -> np.ndarray:
    linear = _p10_lut()[np.clip(frame, 0, 1023)]
    return np.interp(linear, [64, 4064], [0, 4095]).astype(np.uint16)


@dataclass
class VideoInfo:
    path: str
    format_name: str
    width: int
    height: int
    frame_count: int
    fps: float
    bit_depth: int
    duration_s: float
    raw_bayer: str = ""
    storage_bit_depth: int = 0
    acquisition_bit_depth: int = 0
    compression_label: str = ""

    def summary(self) -> str:
        fps_text = f"{self.fps:.3g} fps" if self.fps > 0 else "未知帧率"
        duration_text = f"{self.duration_s:.2f} s" if self.duration_s > 0 else "未知时长"
        return (
            f"{self.format_name} | {self.width} x {self.height} | "
            f"{self.bit_depth}-bit | {self.frame_count} 帧 | {fps_text} | {duration_text}"
        )


@dataclass
class VideoAdjustmentConfig:
    bit_min: int = 0
    bit_max: int = 4095
    brightness: float = 0.0
    gain: float = 1.0
    gamma: float = 1.0
    knee: float = 1.0
    color_mode: str = "gray"
    filter_mode: str = "none"
    flip_horizontal: bool = False
    flip_vertical: bool = False
    rotate_ccw: bool = False
    rotate_cw: bool = False
    crop_enabled: bool = False
    crop_x: int = 0
    crop_y: int = 0
    crop_w: int = 0
    crop_h: int = 0
    resample_enabled: bool = False
    resample_w: int = 0
    resample_h: int = 0
    source_max: int = 4095


class CineReader:
    """Small Phantom CINE reader for common uncompressed 8/12/16-bit files.

    The CINE container has several historical variants. This parser handles the
    standard Vision Research header, BITMAPINFO image header, and image offset
    table. If a rare compressed or non-standard CINE is encountered, callers get
    a clear ValueError instead of silent corruption.
    """

    HEADER_STRUCT = struct.Struct("<2sHHHiIiIIIIQ")
    BITMAPINFO_PREFIX = struct.Struct("<IiiHHII")

    def __init__(self, path: str):
        self.path = str(path)
        self._file = open(self.path, "rb")
        self._parse_headers()

    def close(self):
        try:
            self._file.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()

    def _parse_headers(self):
        self._file.seek(0)
        raw = self._file.read(self.HEADER_STRUCT.size)
        if len(raw) < self.HEADER_STRUCT.size:
            raise ValueError("CINE 文件头不完整。")
        (
            marker,
            _header_size,
            compression,
            version,
            _first_movie_image,
            total_image_count,
            _first_image_no,
            image_count,
            off_image_header,
            _off_setup,
            off_image_offsets,
            _trigger_time,
        ) = self.HEADER_STRUCT.unpack(raw)
        if marker != b"CI":
            raise ValueError("不是有效的 Phantom CINE 文件。")
        if compression not in (0, 1):
            raise ValueError(f"暂不支持压缩 CINE 格式 compression={compression}。")

        self.version = version
        self.frame_count = int(image_count or total_image_count)
        if self.frame_count <= 0:
            raise ValueError("CINE 文件中没有可读取帧。")

        self._file.seek(off_image_header)
        bmp = self._file.read(self.BITMAPINFO_PREFIX.size)
        if len(bmp) < self.BITMAPINFO_PREFIX.size:
            raise ValueError("CINE 图像头不完整。")
        (
            _bmp_size,
            width,
            height,
            _planes,
            bit_count,
            bi_compression,
            size_image,
        ) = self.BITMAPINFO_PREFIX.unpack(bmp)
        if width == 0 or height == 0:
            raise ValueError("CINE 图像尺寸无效。")
        if bi_compression not in (0, 256):
            raise ValueError(f"暂不支持压缩 CINE 图像 biCompression={bi_compression}。")
        if bit_count not in (8, 10, 12, 16, 24):
            raise ValueError(f"暂不支持 {bit_count}-bit CINE 帧。")

        self.width = abs(int(width))
        self.height = abs(int(height))
        self._bottom_up = height > 0
        self.bit_depth = int(bit_count)
        self.storage_bit_depth = self.bit_depth
        self.bi_compression = int(bi_compression)
        self.size_image = int(size_image) if size_image else self._expected_frame_bytes()
        self.bit_depth = self._resolve_bit_depth(int(bit_count), int(bi_compression), self.size_image)
        self.acquisition_bit_depth = 12 if self.bi_compression == 256 and self.bit_depth == 10 else self.bit_depth

        self._file.seek(off_image_offsets)
        offset_raw = self._file.read(self.frame_count * 8)
        if len(offset_raw) < self.frame_count * 8:
            raise ValueError("CINE 帧偏移表不完整。")
        self.offsets = list(struct.unpack(f"<{self.frame_count}Q", offset_raw))

    def _expected_frame_bytes(self) -> int:
        pixels = self.width * self.height
        if self.bit_depth in (8,):
            return pixels
        if self.bit_depth == 10:
            return int(math.ceil(pixels * 10 / 8))
        if self.bit_depth == 12:
            return int(math.ceil(pixels * 12 / 8))
        if self.bit_depth == 16:
            return pixels * 2
        if self.bit_depth == 24:
            return pixels * 3
        return pixels

    def _resolve_bit_depth(self, bit_count: int, bi_compression: int, size_image: int) -> int:
        if bi_compression != 256:
            return bit_count
        pixels = self.width * self.height
        packed10 = int(math.ceil(pixels * 10 / 8))
        packed12 = int(math.ceil(pixels * 12 / 8))
        if size_image == packed10:
            return 10
        if size_image == packed12:
            return 12
        return bit_count

    def info(self) -> VideoInfo:
        if self.bi_compression == 256 and self.bit_depth == 10:
            compression_label = "P10 (Packed 10 log)"
        elif self.bi_compression == 256 and self.bit_depth == 12:
            compression_label = "Packed 12"
        else:
            compression_label = "Uncompressed"
        return VideoInfo(
            path=self.path,
            format_name="Phantom CINE",
            width=self.width,
            height=self.height,
            frame_count=self.frame_count,
            fps=0.0,
            bit_depth=self.bit_depth,
            duration_s=0.0,
            raw_bayer="CINE RAW",
            storage_bit_depth=self.bit_depth,
            acquisition_bit_depth=self.acquisition_bit_depth,
            compression_label=compression_label,
        )

    def read_frame(self, index: int) -> np.ndarray:
        if not 0 <= index < self.frame_count:
            raise IndexError(index)
        offset = int(self.offsets[index])
        self._file.seek(offset)
        first = self._file.read(4)
        if len(first) < 4:
            raise ValueError(f"CINE 第 {index} 帧数据不完整。")
        annotation_size = struct.unpack("<I", first)[0]
        if 4 <= annotation_size <= 65536:
            self._file.seek(offset + annotation_size)
        else:
            self._file.seek(offset)
        data = self._file.read(self.size_image)
        frame = self._decode_pixels(data)
        if self._bottom_up and self.bi_compression == 0:
            frame = np.flipud(frame)
        return frame

    def _decode_pixels(self, data: bytes) -> np.ndarray:
        pixels = self.width * self.height
        if self.bit_depth == 8:
            arr = np.frombuffer(data[:pixels], dtype=np.uint8)
            return arr.reshape(self.height, self.width)
        if self.bit_depth == 16:
            arr = np.frombuffer(data[:pixels * 2], dtype="<u2")
            return arr.reshape(self.height, self.width)
        if self.bit_depth == 10:
            packed = np.frombuffer(data, dtype=np.uint8)
            usable = (packed.size // 5) * 5
            if usable < 5:
                raise ValueError("CINE packed 10-bit frame is too small.")
            groups = packed[:usable].reshape(-1, 5).astype(np.uint16)
            p0 = (groups[:, 0] << 2) | (groups[:, 1] >> 6)
            p1 = ((groups[:, 1] & 0x3F) << 4) | (groups[:, 2] >> 4)
            p2 = ((groups[:, 2] & 0x0F) << 6) | (groups[:, 3] >> 2)
            p3 = ((groups[:, 3] & 0x03) << 8) | groups[:, 4]
            arr = np.empty(groups.shape[0] * 4, dtype=np.uint16)
            arr[0::4] = p0
            arr[1::4] = p1
            arr[2::4] = p2
            arr[3::4] = p3
            frame = arr[:pixels].reshape(self.height, self.width)
            if self.bi_compression == 256 and self.acquisition_bit_depth == 12:
                frame = linearize_p10_to_12bit(frame)
            return frame
        if self.bit_depth == 12:
            packed = np.frombuffer(data, dtype=np.uint8)
            usable = (packed.size // 3) * 3
            if usable < 3:
                raise ValueError("CINE packed frame is too small.")
            triples = packed[:usable].reshape(-1, 3).astype(np.uint16)
            p0 = triples[:, 0] | ((triples[:, 1] & 0x0F) << 8)
            p1 = (triples[:, 1] >> 4) | (triples[:, 2] << 4)
            arr = np.empty(triples.shape[0] * 2, dtype=np.uint16)
            arr[0::2] = p0
            arr[1::2] = p1
            return arr[:pixels].reshape(self.height, self.width)
        if self.bit_depth == 24:
            arr = np.frombuffer(data[:pixels * 3], dtype=np.uint8)
            return arr.reshape(self.height, self.width, 3)
        raise ValueError(f"Unsupported CINE bit depth: {self.bit_depth}")


class OpenCVVideoReader:
    def __init__(self, path: str):
        self.path = str(path)
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise ValueError("OpenCV 无法打开该视频。")
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)

    def close(self):
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()

    def info(self) -> VideoInfo:
        duration = self.frame_count / self.fps if self.fps > 0 and self.frame_count > 0 else 0.0
        suffix = Path(self.path).suffix.upper().lstrip(".") or "Video"
        return VideoInfo(
            path=self.path,
            format_name=suffix,
            width=self.width,
            height=self.height,
            frame_count=self.frame_count,
            fps=self.fps,
            bit_depth=24,
            duration_s=duration,
        )

    def read_frame(self, index: int) -> np.ndarray:
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(index)))
        ok, frame = self.cap.read()
        if not ok or frame is None:
            raise ValueError(f"无法读取第 {index} 帧。")
        return frame


def open_video_reader(path: str):
    suffix = Path(path).suffix.lower()
    if suffix == ".cine":
        return CineReader(path)
    return OpenCVVideoReader(path)


def probe_video(path: str) -> VideoInfo:
    with open_video_reader(path) as reader:
        return reader.info()


def convert_frame_bit_depth(frame: np.ndarray, bit_depth: int) -> np.ndarray:
    if bit_depth == 24:
        if frame.ndim == 2:
            gray8 = normalize_to_uint8(frame)
            return cv2.cvtColor(gray8, cv2.COLOR_GRAY2BGR)
        return normalize_to_uint8(frame)
    gray = frame if frame.ndim == 2 else cv2.cvtColor(normalize_to_uint8(frame), cv2.COLOR_BGR2GRAY)
    if bit_depth == 8:
        return normalize_to_uint8(gray)
    if bit_depth == 12:
        arr = gray.astype(np.float32)
        maxv = float(np.max(arr)) if arr.size else 0.0
        if maxv <= 255.0:
            return np.clip(arr * (4095.0 / 255.0), 0, 4095).astype(np.uint16)
        if maxv <= 1023.0:
            return np.clip(arr * (4095.0 / 1023.0), 0, 4095).astype(np.uint16)
        if maxv > 4095.0:
            return np.clip(arr * (4095.0 / maxv), 0, 4095).astype(np.uint16)
        return np.clip(arr, 0, 4095).astype(np.uint16)
    raise ValueError("输出位深仅支持 8、12、24 位。")


def prepare_adjustment_frame(frame: np.ndarray, source_max: int = 4095) -> np.ndarray:
    if frame.ndim == 3:
        gray = cv2.cvtColor(normalize_to_uint8(frame), cv2.COLOR_BGR2GRAY).astype(np.float32)
        return gray * (float(source_max) / 255.0)
    arr = frame.astype(np.float32)
    maxv = float(np.max(arr)) if arr.size else 0.0
    if source_max >= 4095 and 0.0 < maxv <= 1023.0:
        arr = arr * (4095.0 / 1023.0)
    return np.clip(arr, 0, float(source_max))


def apply_video_adjustments(frame: np.ndarray, config: Optional[VideoAdjustmentConfig] = None) -> np.ndarray:
    if config is None:
        return frame
    work = prepare_adjustment_frame(frame, config.source_max)
    if config.crop_enabled and config.crop_w > 0 and config.crop_h > 0:
        h, w = work.shape[:2]
        x = max(0, min(int(config.crop_x), max(0, w - 1)))
        y = max(0, min(int(config.crop_y), max(0, h - 1)))
        cw = max(1, min(int(config.crop_w), w - x))
        ch = max(1, min(int(config.crop_h), h - y))
        work = work[y:y + ch, x:x + cw]
    if config.resample_enabled and config.resample_w > 0 and config.resample_h > 0:
        work = cv2.resize(
            work,
            (int(config.resample_w), int(config.resample_h)),
            interpolation=cv2.INTER_AREA,
        )
    if config.flip_horizontal:
        work = cv2.flip(work, 1)
    if config.flip_vertical:
        work = cv2.flip(work, 0)
    if config.rotate_ccw:
        work = cv2.rotate(work, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if config.rotate_cw:
        work = cv2.rotate(work, cv2.ROTATE_90_CLOCKWISE)
    if config.filter_mode == "median":
        work = cv2.medianBlur(np.clip(work, 0, 65535).astype(np.uint16), 3).astype(np.float32)
    elif config.filter_mode == "gaussian":
        work = cv2.GaussianBlur(work, (3, 3), 0)

    bit_min = float(min(config.bit_min, config.bit_max - 1))
    bit_max = float(max(config.bit_max, config.bit_min + 1))
    norm = np.clip((work - bit_min) / (bit_max - bit_min), 0.0, 1.0)
    gamma = max(0.05, float(config.gamma))
    norm = np.power(norm, 1.0 / gamma)
    norm = norm * max(0.0, float(config.gain)) + float(config.brightness) / 100.0
    knee = max(0.05, float(config.knee))
    if abs(knee - 1.0) > 1e-6:
        norm = 1.0 - np.power(1.0 - np.clip(norm, 0.0, 1.0), knee)
    return np.clip(norm * float(config.source_max), 0, float(config.source_max)).astype(np.float32)


def render_adjusted_display(frame: np.ndarray, config: Optional[VideoAdjustmentConfig] = None) -> np.ndarray:
    adjusted = apply_video_adjustments(frame, config)
    gray8 = normalize_to_uint8(adjusted)
    color_mode = (config.color_mode if config else "gray").lower()
    if color_mode == "jet":
        return cv2.applyColorMap(gray8, cv2.COLORMAP_JET)
    if color_mode == "hot":
        return cv2.applyColorMap(gray8, cv2.COLORMAP_HOT)
    if color_mode == "turbo":
        return cv2.applyColorMap(gray8, cv2.COLORMAP_TURBO)
    return cv2.cvtColor(gray8, cv2.COLOR_GRAY2BGR)


def normalize_to_uint8(frame: np.ndarray) -> np.ndarray:
    if frame.dtype == np.uint8:
        return frame
    arr = frame.astype(np.float32)
    maxv = float(np.max(arr)) if arr.size else 0.0
    minv = float(np.min(arr)) if arr.size else 0.0
    if maxv <= minv:
        return np.zeros(frame.shape, dtype=np.uint8)
    return np.clip((arr - minv) * 255.0 / (maxv - minv), 0, 255).astype(np.uint8)


def export_video_frames(
    path: str,
    output_dir: str,
    bit_depth: int = 8,
    image_format: str = "png",
    start: int = 0,
    end: Optional[int] = None,
    step: int = 1,
    prefix: Optional[str] = None,
    adjustment_config: Optional[VideoAdjustmentConfig] = None,
    progress=None,
) -> int:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    image_format = image_format.lower().lstrip(".")
    if image_format not in {"png", "tif", "tiff", "bmp"}:
        raise ValueError("图片格式仅支持 PNG、TIF、TIFF、BMP。")

    count = 0
    with open_video_reader(path) as reader:
        info = reader.info()
        final = info.frame_count - 1 if end is None else min(int(end), info.frame_count - 1)
        start = max(0, int(start))
        step = max(1, int(step))
        stem = prefix or Path(path).stem
        total = max(0, ((final - start) // step) + 1) if final >= start else 0
        for out_index, frame_index in enumerate(range(start, final + 1, step), start=1):
            frame = reader.read_frame(frame_index)
            if adjustment_config is not None and bit_depth == 24 and adjustment_config.color_mode != "gray":
                converted = render_adjusted_display(frame, adjustment_config)
            else:
                if adjustment_config is not None:
                    frame = apply_video_adjustments(frame, adjustment_config)
                converted = convert_frame_bit_depth(frame, bit_depth)
            dst = output / f"{stem}_frame_{frame_index:06d}_{bit_depth}bit.{image_format}"
            ok, encoded = cv2.imencode(f".{image_format}", converted)
            if not ok:
                raise IOError(f"无法写入图片: {dst}")
            encoded.tofile(str(dst))
            count += 1
            if progress is not None:
                pct = int(out_index * 100 / max(1, total))
                progress(pct, frame_index, str(dst))
    return count
