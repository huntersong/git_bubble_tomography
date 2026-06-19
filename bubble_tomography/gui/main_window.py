"""
气泡三维层析重建系统 - PyQt5 GUI主窗口
"""

import sys
import os
import json
import re
import time
import cv2
import numpy as np
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Dict, List, Optional

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QGroupBox, QLabel, QLineEdit, QPushButton,
    QComboBox, QSpinBox, QDoubleSpinBox, QListWidget, QListWidgetItem,
    QFileDialog, QProgressBar, QTextEdit, QSplitter,
    QMessageBox, QScrollArea, QGridLayout, QCheckBox, QSlider,
    QStatusBar, QToolBar, QAction, QFrame, QListWidget,
    QStackedWidget, QButtonGroup, QAbstractItemView, QToolBox,
    QMenu, QTreeWidget, QTreeWidgetItem, QSizePolicy
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QModelIndex, QSize, QFileSystemWatcher
from PyQt5.QtGui import QImage, QPixmap, QIcon, QIntValidator, QFont, QColor, QDrag
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from calibration.camera_calibrator import MultiCameraCalibrator, CameraParams
from mart.mart_reconstructor import (
    MARTReconstructor, SMARTReconstructor, ConvSMARTReconstructor,
    ReconstructionConfig, MARTConfig, create_reconstructor,
    TomographicReconstructor,
)
from utils.image_processor import BubbleImageProcessor
from visualization.visualizer import ResultVisualizer
from particles.particle_reconstructor import (
    Particle3DReconstructor, TriangulationConfig
)
from particles.velocity_field import (
    VelocityFieldCalculator, CorrelationConfig
)
from particles.piv2d import PIV2DCalculator, PIV2DConfig, SUPPORTED_EXTS as PIV2D_EXTS
from raytrace.raytrace_reconstructor import RaytraceProcessor
from utils.image_editor import (
    ImageEditor, ImageEditConfig,
    CropParams, GrayParams, BrightnessContrastParams,
    MirrorParams, RotateParams, BitDepthParams, GrayMathParams,
    ArithmeticParams, ThresholdParams,
    robust_imread,
)
from utils.cpu_parallel import default_worker_count, limited_opencv_threads


class CalibrationWorker(QThread):
    """标定工作线程"""
    progress = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, calibrator: MultiCameraCalibrator,
                 camera_images: Dict[str, List[str]]):
        super().__init__()
        self.calibrator = calibrator
        self.camera_images = camera_images

    def run(self):
        try:
            results = {}
            for cam_id, img_paths in self.camera_images.items():
                self.progress.emit(f"正在标定相机 {cam_id}...")
                params = self.calibrator.calibrate_camera(cam_id, img_paths)
                results[cam_id] = params
            self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))


class PreviewImageLoader(QThread):
    """Read one preview image and detect calibration marks off the UI thread."""
    loaded = pyqtSignal(int, str, object, object)
    error = pyqtSignal(int, str)

    def __init__(self, request_id: int, image_path: str, detection_config: Optional[dict] = None):
        super().__init__()
        self.request_id = request_id
        self.image_path = image_path
        self.detection_config = detection_config or {}

    def run(self):
        image = robust_imread(self.image_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            self.error.emit(self.request_id, self.image_path)
            return

        detection = None
        if self.detection_config:
            try:
                detector = MultiCameraCalibrator(**self.detection_config)
                source = image
                if image.ndim == 2:
                    source = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                elif image.ndim == 3 and image.shape[2] == 4:
                    source = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
                observation = detector.detect_pattern_observation(source)
                if observation is not None:
                    detection = {
                        "points": observation.image_points.reshape(-1, 2).tolist(),
                        "point_ids": [tuple(point_id) for point_id in observation.point_ids],
                    }
            except Exception as exc:
                detection = {"error": str(exc)}

        self.loaded.emit(self.request_id, self.image_path, image, detection)


class ReconstructionWorker(QThread):
    """重建工作线程"""
    progress = pyqtSignal(str)
    iteration_done = pyqtSignal(int, float)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, reconstructor: TomographicReconstructor,
                 projections: Dict[str, np.ndarray],
                 camera_params: Dict[str, dict]):
        super().__init__()
        self.reconstructor = reconstructor
        self.projections = projections
        self.camera_params = camera_params

    def run(self):
        try:
            algo = self.reconstructor.config.algorithm
            self.progress.emit(f"开始 {algo} 重建...")
            errors = []

            def callback(iteration, volume, error):
                self.iteration_done.emit(iteration + 1, error)
                errors.append(error)
                self.progress.emit(f"迭代 {iteration + 1}/{self.reconstructor.config.max_iterations}")

            volume = self.reconstructor.reconstruct(
                self.projections, self.camera_params, callback
            )
            points, normals = self.reconstructor.extract_bubble_point_cloud()
            stats = self.reconstructor.get_volume_stats()

            self.finished.emit({
                'volume': volume,
                'points': points,
                'normals': normals,
                'stats': stats,
                'errors': errors
            })
        except Exception as e:
            self.error.emit(str(e))


class BatchReconstructionWorker(QThread):
    """批量重建后台工作线程。"""
    progress = pyqtSignal(str)
    timepoint_done = pyqtSignal(int, dict)  # timepoint_index, result
    all_done = pyqtSignal(dict)  # {timepoint_index: result}
    error = pyqtSignal(str)

    def __init__(self, reconstructor: TomographicReconstructor,
                 bubble_images_sequence: Dict[int, Dict[str, np.ndarray]],
                 camera_params: Dict[str, dict],
                 reference_images: Dict[str, np.ndarray],
                 image_processor: BubbleImageProcessor,
                 projection_type: str):
        super().__init__()
        self.reconstructor = reconstructor
        self.bubble_images_sequence = bubble_images_sequence
        self.camera_params = camera_params
        self.reference_images = reference_images
        self.image_processor = image_processor
        self.projection_type = projection_type
        self._camera_params_for_preprocess = {}

    def run(self):
        try:
            all_results = {}
            total = len(self.bubble_images_sequence)

            for cam_id, cp in self.camera_params.items():
                self._camera_params_for_preprocess[cam_id] = {
                    'camera_matrix': cp['camera_matrix'],
                    'dist_coeffs': cp.get('dist_coeffs', None)
                }

            for tp_idx, cam_imgs in sorted(self.bubble_images_sequence.items()):
                self.progress.emit(
                    f"[{tp_idx+1}/{total}] 处理时间?t{tp_idx} ..."
                )

                # ???
                projections = self.image_processor.prepare_projection_data(
                    cam_imgs, self._camera_params_for_preprocess,
                    self.reference_images,
                    projection_type=self.projection_type
                )

                # MART重建
                errors = []
                def callback(iteration, volume, error):
                    errors.append(error)

                volume = self.reconstructor.reconstruct(
                    projections, self.camera_params, callback
                )
                points, normals = self.reconstructor.extract_bubble_point_cloud()
                stats = self.reconstructor.get_volume_stats()

                result = {
                    'volume': volume,
                    'points': points,
                    'normals': normals,
                    'stats': stats,
                    'errors': errors,
                    'projections': projections
                }
                all_results[tp_idx] = result
                self.timepoint_done.emit(tp_idx, result)

            self.all_done.emit(all_results)
        except Exception as e:
            self.error.emit(str(e))


class BatchPIVWorker(QThread):
    """Batch PIV worker."""
    progress = pyqtSignal(str)
    timepoint_done = pyqtSignal(int, dict)
    all_done = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, particle_sequence: Dict[int, Dict[str, np.ndarray]],
                 calibrator, triang_config, vel_config, dt, domain_size):
        super().__init__()
        self.particle_sequence = particle_sequence
        self.calibrator = calibrator
        self.triang_config = triang_config
        self.vel_config = vel_config
        self.dt = dt
        self.domain_size = domain_size

    def run(self):
        try:
            all_results = {}
            sorted_tps = sorted(self.particle_sequence.keys())
            reconstructor = Particle3DReconstructor(self.triang_config)

            total_pairs = len(sorted_tps) - 1
            if total_pairs < 1:
                self.error.emit("至少需要 2 个时间点才能计算速度")
                return

            for i in range(len(sorted_tps) - 1):
                tp1 = sorted_tps[i]
                tp2 = sorted_tps[i + 1]
                self.progress.emit(
                    f"[{i+1}/{total_pairs}] 处理 t{tp1} 到 t{tp2} ..."
                )

                # 粒子重建
                p3d_1 = reconstructor.reconstruct_particles(
                    self.particle_sequence[tp1], self.calibrator
                )
                p3d_2 = reconstructor.reconstruct_particles(
                    self.particle_sequence[tp2], self.calibrator
                )

                # 速度场
                calculator = VelocityFieldCalculator(
                    config=self.vel_config,
                    domain_size=self.domain_size,
                    dt=self.dt
                )
                vel_result = calculator.compute_velocity_field(p3d_1, p3d_2)

                pair_result = {
                    'particles_3d_frame1': p3d_1,
                    'particles_3d_frame2': p3d_2,
                    'velocity_result': vel_result
                }
                all_results[tp1] = pair_result
                self.timepoint_done.emit(tp1, pair_result)

            self.all_done.emit(all_results)
        except Exception as e:
            self.error.emit(str(e))


class BatchPIV2DWorker(QThread):
    """Batch 2D PIV worker."""
    progress = pyqtSignal(int, int, str)
    finished = pyqtSignal(int, int, list)
    error = pyqtSignal(str)

    def __init__(
        self,
        src_dir: str,
        dst_dir: str,
        config: PIV2DConfig,
        exclusion_mask: Optional[np.ndarray] = None,
        frame_pairs: Optional[List[tuple]] = None,
    ):
        super().__init__()
        self.src_dir = src_dir
        self.dst_dir = dst_dir
        self.config = config
        self.exclusion_mask = None if exclusion_mask is None else np.asarray(exclusion_mask, dtype=bool).copy()
        self.frame_pairs = frame_pairs
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            calculator = PIV2DCalculator(self.config)
            success, total, outputs = calculator.process_batch_directory(
                self.src_dir,
                self.dst_dir,
                exclusion_mask=self.exclusion_mask,
                frame_pairs=self.frame_pairs,
                progress_callback=lambda done, total, name: self.progress.emit(done, total, name),
                stop_checker=lambda: self._stop,
            )
            self.finished.emit(success, total, outputs)
        except Exception as e:
            self.error.emit(str(e))


class PIV2DPreviewWidget(QWidget):
    """Grayscale image preview with colorbar, pixel readout, and optional vectors."""

    mask_changed = pyqtSignal(object)

    def __init__(self, placeholder: str = "预览区域", parent=None):
        super().__init__(parent)
        self._image = None
        self._result = None
        self._title = placeholder
        self._vector_scale = 0.15
        self._vector_color_mode = "speed"
        self._colorbar = None
        self._exclusion_mask = None
        self._mask_selection_enabled = False
        self._mask_shape = "rectangle"
        self._mask_press = None
        self._mask_drag_patch = None
        self._mask_polygon_points = []
        self._correlation_window_size = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.figure = Figure(figsize=(4, 3), tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.axes = self.figure.add_subplot(111)
        self.canvas.setMinimumSize(260, 200)
        self.canvas.mpl_connect("button_press_event", self._on_mouse_press)
        self.canvas.mpl_connect("button_release_event", self._on_mouse_release)
        self.canvas.mpl_connect("motion_notify_event", self._on_mouse_move)
        layout.addWidget(self.canvas, stretch=1)

        self.info_label = QLabel("灰度值: --")
        self.info_label.setStyleSheet("color: #444; font-family: Consolas;")
        layout.addWidget(self.info_label)
        self._render()

    def set_correlation_window_size(self, size: Optional[int]):
        self._correlation_window_size = None if size is None else int(size)

    def set_image_path(self, path: str):
        image = robust_imread(path, cv2.IMREAD_UNCHANGED)
        if image is None:
            return
        self.set_image_array(image, os.path.basename(path))

    def set_image_array(self, image: np.ndarray, title: str = ""):
        self._image = self._to_gray(image)
        self._result = None
        self._title = title or self._title
        if self._exclusion_mask is not None and self._exclusion_mask.shape != self._image.shape[:2]:
            self._exclusion_mask = None
        self._render()

    def set_vector_result(self, image: np.ndarray, result: dict, title: str = ""):
        self._image = self._to_gray(image)
        self._result = result
        self._title = title or "速度矢量"
        if self._exclusion_mask is not None and self._exclusion_mask.shape != self._image.shape[:2]:
            self._exclusion_mask = None
        self._render()

    def set_vector_style(self, scale: float, color_mode: str):
        self._vector_scale = float(scale)
        self._vector_color_mode = color_mode
        self._render()

    def clear(self, text: Optional[str] = None):
        self._image = None
        self._result = None
        self._exclusion_mask = None
        if text:
            self._title = text
        self.info_label.setText("灰度值: --")
        self._render()

    def begin_mask_selection(self, shape: str = "rectangle"):
        if self._image is None:
            return False
        self._mask_shape = shape
        self._mask_selection_enabled = True
        self._mask_press = None
        self._mask_polygon_points = []
        self._remove_drag_patch()
        hints = {
            "rectangle": "拖拽选择矩形无粒子区域",
            "circle": "拖拽选择圆形无粒子区域",
            "triangle": "拖拽选择三角形无粒子区域",
            "polygon": "左键添加多边形顶点，双击结束",
        }
        self.info_label.setText(hints.get(shape, "选择无粒子区域"))
        return True

    def set_exclusion_mask(self, mask: Optional[np.ndarray]):
        if mask is None:
            self._exclusion_mask = None
        else:
            mask = np.asarray(mask, dtype=bool)
            if self._image is not None and mask.shape != self._image.shape[:2]:
                return
            self._exclusion_mask = mask.copy()
        self._render()

    def clear_exclusion_mask(self):
        self._mask_selection_enabled = False
        self._mask_press = None
        self._mask_polygon_points = []
        self._exclusion_mask = None
        self.mask_changed.emit(None)
        self._render()

    def exclusion_mask(self) -> Optional[np.ndarray]:
        if self._exclusion_mask is None:
            return None
        return self._exclusion_mask.copy()

    def _render(self):
        self.figure.clear()
        self.axes = self.figure.add_subplot(111)
        self._colorbar = None

        if self._image is None:
            self.axes.text(0.5, 0.5, self._title, ha="center", va="center")
            self.axes.set_axis_off()
            self.canvas.draw_idle()
            return

        im = self.axes.imshow(self._image, cmap="gray", origin="upper")
        self._colorbar = self.figure.colorbar(im, ax=self.axes, fraction=0.046, pad=0.04)
        self.axes.set_title(self._title, fontsize=10)
        self.axes.set_axis_off()

        self._draw_exclusion_mask()

        if self._result is not None:
            self._draw_vectors()

        self.canvas.draw_idle()

    def _draw_exclusion_mask(self):
        if self._exclusion_mask is None or self._image is None:
            return
        if self._exclusion_mask.shape != self._image.shape[:2] or not np.any(self._exclusion_mask):
            return
        overlay = np.zeros((*self._exclusion_mask.shape, 4), dtype=np.float32)
        overlay[self._exclusion_mask] = [1.0, 0.0, 0.0, 0.28]
        self.axes.imshow(overlay, origin="upper")

    def _draw_vectors(self):
        result = self._result
        valid = result.get("valid")
        if valid is None or not np.any(valid):
            return

        x = result["x"][valid]
        y = result["y"][valid]
        u = result["u"][valid] * self._vector_scale
        v = result["v"][valid] * self._vector_scale

        if self._vector_color_mode == "speed":
            colors = result["speed"][valid]
            quiver = self.axes.quiver(
                x, y, u, v, colors,
                cmap="jet", angles="xy", scale_units="xy", scale=1,
                width=0.003,
            )
            self.figure.colorbar(quiver, ax=self.axes, fraction=0.046, pad=0.10)
        else:
            color_map = {
                "red": "#d32f2f",
                "green": "#2e7d32",
                "blue": "#1565c0",
                "yellow": "#f9a825",
                "white": "#ffffff",
            }
            self.axes.quiver(
                x, y, u, v,
                color=color_map.get(self._vector_color_mode, "#d32f2f"),
                angles="xy", scale_units="xy", scale=1, width=0.003,
            )

    def _on_mouse_press(self, event):
        if self._is_right_button(event):
            self._show_context_menu(event)
            return

        if not self._mask_selection_enabled or self._image is None or event.inaxes is not self.axes:
            return
        if event.xdata is None or event.ydata is None:
            return

        if self._mask_shape == "polygon":
            self._mask_polygon_points.append((float(event.xdata), float(event.ydata)))
            if getattr(event, "dblclick", False) and len(self._mask_polygon_points) >= 3:
                self._finish_polygon_mask()
            else:
                self._draw_polygon_preview()
            return

        self._mask_press = (float(event.xdata), float(event.ydata))
        self._remove_drag_patch()

    def _on_mouse_release(self, event):
        if not self._mask_selection_enabled or self._image is None or self._mask_press is None:
            return
        if event.inaxes is not self.axes or event.xdata is None or event.ydata is None:
            self._mask_press = None
            self._remove_drag_patch()
            return

        h, w = self._image.shape[:2]
        x0, y0 = self._mask_press
        x1, y1 = float(event.xdata), float(event.ydata)
        x_min = max(0, min(w, int(np.floor(min(x0, x1)))))
        x_max = max(0, min(w, int(np.ceil(max(x0, x1)))))
        y_min = max(0, min(h, int(np.floor(min(y0, y1)))))
        y_max = max(0, min(h, int(np.ceil(max(y0, y1)))))

        self._mask_selection_enabled = False
        self._mask_press = None
        self._remove_drag_patch()

        if x_max <= x_min or y_max <= y_min:
            self.info_label.setText("无粒子区域未改变")
            return

        mask = self._shape_mask_from_bounds(x_min, x_max, y_min, y_max)
        self._apply_mask(mask, f"已添加{self._mask_shape_label()}无粒子区域: x={x_min}:{x_max}, y={y_min}:{y_max}")

    def _on_mouse_move(self, event):
        if self._mask_selection_enabled and self._image is not None and self._mask_press is not None:
            self._update_drag_patch(event)
            return
        if self._image is None or event.inaxes is not self.axes:
            self.info_label.setText("灰度值: --")
            return
        if event.xdata is None or event.ydata is None:
            self.info_label.setText("灰度值: --")
            return
        x = int(round(event.xdata))
        y = int(round(event.ydata))
        h, w = self._image.shape[:2]
        if 0 <= x < w and 0 <= y < h:
            value = float(self._image[y, x])
            self.info_label.setText(f"x={x}  y={y}  灰度值={value:.3f}")
        else:
            self.info_label.setText("灰度值: --")

    def _update_drag_patch(self, event):
        if event.inaxes is not self.axes or event.xdata is None or event.ydata is None:
            return
        from matplotlib.patches import Circle, Polygon, Rectangle

        x0, y0 = self._mask_press
        x1, y1 = float(event.xdata), float(event.ydata)
        self._remove_drag_patch()
        if self._mask_shape == "circle":
            self._mask_drag_patch = Circle(
                ((x0 + x1) / 2.0, (y0 + y1) / 2.0),
                max(1.0, min(abs(x1 - x0), abs(y1 - y0)) / 2.0),
                fill=False,
                edgecolor="#ff2d2d",
                linewidth=1.5,
                linestyle="--",
            )
        elif self._mask_shape == "triangle":
            points = [
                ((x0 + x1) / 2.0, min(y0, y1)),
                (min(x0, x1), max(y0, y1)),
                (max(x0, x1), max(y0, y1)),
            ]
            self._mask_drag_patch = Polygon(
                points,
                closed=True,
                fill=False,
                edgecolor="#ff2d2d",
                linewidth=1.5,
                linestyle="--",
            )
        else:
            self._mask_drag_patch = Rectangle(
                (min(x0, x1), min(y0, y1)),
                abs(x1 - x0),
                abs(y1 - y0),
                fill=False,
                edgecolor="#ff2d2d",
                linewidth=1.5,
                linestyle="--",
            )
        self.axes.add_patch(self._mask_drag_patch)
        self.canvas.draw_idle()

    def _draw_polygon_preview(self):
        from matplotlib.patches import Polygon

        self._remove_drag_patch()
        if not self._mask_polygon_points:
            return
        self._mask_drag_patch = Polygon(
            self._mask_polygon_points,
            closed=False,
            fill=False,
            edgecolor="#ff2d2d",
            linewidth=1.5,
            linestyle="--",
        )
        self.axes.add_patch(self._mask_drag_patch)
        self.canvas.draw_idle()

    def _remove_drag_patch(self):
        if self._mask_drag_patch is not None:
            try:
                self._mask_drag_patch.remove()
            except ValueError:
                pass
            self._mask_drag_patch = None

    @staticmethod
    def _is_right_button(event) -> bool:
        button = getattr(event, "button", None)
        return button == 3 or str(button).lower().endswith("right")

    def _shape_mask_from_bounds(self, x_min: int, x_max: int, y_min: int, y_max: int) -> np.ndarray:
        h, w = self._image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        if self._mask_shape == "circle":
            center = ((x_min + x_max) // 2, (y_min + y_max) // 2)
            radius = max(1, min(x_max - x_min, y_max - y_min) // 2)
            cv2.circle(mask, center, radius, 1, thickness=-1)
        elif self._mask_shape == "triangle":
            points = np.array(
                [
                    [(x_min + x_max) // 2, y_min],
                    [x_min, y_max],
                    [x_max, y_max],
                ],
                dtype=np.int32,
            )
            cv2.fillPoly(mask, [points], 1)
        else:
            mask[y_min:y_max, x_min:x_max] = 1
        return mask.astype(bool)

    def _finish_polygon_mask(self):
        if self._image is None or len(self._mask_polygon_points) < 3:
            self.info_label.setText("多边形至少需要3个顶点")
            return
        h, w = self._image.shape[:2]
        points = np.array(
            [
                [int(np.clip(round(x), 0, w - 1)), int(np.clip(round(y), 0, h - 1))]
                for x, y in self._mask_polygon_points
            ],
            dtype=np.int32,
        )
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [points], 1)
        self._mask_selection_enabled = False
        self._mask_press = None
        self._mask_polygon_points = []
        self._remove_drag_patch()
        self._apply_mask(mask.astype(bool), "已添加多边形无粒子区域")

    def _apply_mask(self, mask: np.ndarray, message: str):
        if self._image is None or mask is None or not np.any(mask):
            self.info_label.setText("无粒子区域未改变")
            return
        h, w = self._image.shape[:2]
        if self._exclusion_mask is None:
            self._exclusion_mask = np.zeros((h, w), dtype=bool)
        self._exclusion_mask |= mask.astype(bool)
        self.mask_changed.emit(self._exclusion_mask.copy())
        self.info_label.setText(message)
        self._render()

    def _show_context_menu(self, event):
        menu = QMenu(self)
        text = (
            f"当前互相关窗口尺寸: {self._correlation_window_size} px"
            if self._correlation_window_size
            else "当前互相关窗口尺寸: 未设置"
        )
        action = menu.addAction(text)
        action.setEnabled(False)
        if getattr(event, "guiEvent", None) is not None:
            menu.exec_(self.canvas.mapToGlobal(event.guiEvent.pos()))
        else:
            menu.exec_(self.canvas.mapToGlobal(self.canvas.rect().center()))

    def _mask_shape_label(self) -> str:
        return {
            "rectangle": "矩形",
            "circle": "圆形",
            "triangle": "三角形",
            "polygon": "多边形",
        }.get(self._mask_shape, "几何")

    @staticmethod
    def _to_gray(image: np.ndarray) -> np.ndarray:
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image.astype(np.float32)


class CalibrationPreviewWidget(QWidget):
    """Interactive calibration image viewer with colorbar, pixel readout, and ruler."""
    rulerDistanceChanged = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._datasets: Dict[str, List[str]] = {}
        self._camera_ids: List[str] = []
        self._current_array = None
        self._current_gray = None
        self._current_is_static = False
        self._current_title = ""
        self._current_with_colorbar = True
        self._detection_config: Optional[dict] = None
        self._current_detection = None
        self._preview_request_id = 0
        self._preview_loaders = []
        self._ruler_enabled = False
        self._ruler_points = []
        self._ruler_dragging = False
        self._ruler_distance_px = 0.0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        camera_row = QHBoxLayout()
        camera_row.addWidget(QLabel("相机/视图:"))
        self.prev_camera_btn = QPushButton("上一组")
        self.prev_camera_btn.clicked.connect(self._show_previous_camera)
        camera_row.addWidget(self.prev_camera_btn)

        self.camera_combo = QComboBox()
        self.camera_combo.currentIndexChanged.connect(self._on_camera_changed)
        camera_row.addWidget(self.camera_combo, stretch=1)

        self.next_camera_btn = QPushButton("下一组")
        self.next_camera_btn.clicked.connect(self._show_next_camera)
        camera_row.addWidget(self.next_camera_btn)
        layout.addLayout(camera_row)

        self.figure = Figure(figsize=(6, 4), tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.axes = self.figure.add_subplot(111)
        self._colorbar = None
        self.canvas.setMinimumSize(420, 320)
        self.canvas.mpl_connect("motion_notify_event", self._on_mouse_move)
        self.canvas.mpl_connect("button_press_event", self._on_mouse_press)
        self.canvas.mpl_connect("button_release_event", self._on_mouse_release)
        self.canvas.setContextMenuPolicy(Qt.CustomContextMenu)
        self.canvas.customContextMenuRequested.connect(self._show_context_menu)
        layout.addWidget(self.canvas, stretch=1)

        slider_row = QHBoxLayout()
        slider_row.addWidget(QLabel("序列:"))
        self.image_slider = QSlider(Qt.Horizontal)
        self.image_slider.setMinimum(0)
        self.image_slider.setMaximum(0)
        self.image_slider.valueChanged.connect(self._on_slider_changed)
        slider_row.addWidget(self.image_slider, stretch=1)
        self.image_index_label = QLabel("0 / 0")
        slider_row.addWidget(self.image_index_label)
        layout.addLayout(slider_row)

        self.info_label = QLabel("未加载标定图像")
        self.info_label.setStyleSheet("color: #666;")
        layout.addWidget(self.info_label)

        self.pixel_label = QLabel("像素值: --")
        self.pixel_label.setStyleSheet("color: #444; font-family: Consolas;")
        layout.addWidget(self.pixel_label)

        self.clear()

    def set_detection_config(self, config: Optional[dict]):
        self._detection_config = dict(config) if config else None

    def clear(self):
        self._datasets = {}
        self._camera_ids = []
        self._current_array = None
        self._current_gray = None
        self._current_is_static = False
        self._current_title = ""
        self._current_detection = None
        self._ruler_points = []
        self._ruler_dragging = False
        self._ruler_distance_px = 0.0

        self.camera_combo.blockSignals(True)
        self.camera_combo.clear()
        self.camera_combo.blockSignals(False)
        self.image_slider.blockSignals(True)
        self.image_slider.setRange(0, 0)
        self.image_slider.setValue(0)
        self.image_slider.blockSignals(False)
        self.image_index_label.setText("0 / 0")
        self.info_label.setText("未加载标定图像")
        self.pixel_label.setText("像素值: --")
        self.prev_camera_btn.setEnabled(False)
        self.next_camera_btn.setEnabled(False)
        self.camera_combo.setEnabled(False)
        self.image_slider.setEnabled(False)
        self._render_placeholder("预览区域")

    def set_image_sets(
        self,
        datasets: Dict[str, List[str]],
        selected_key: Optional[str] = None,
    ):
        filtered = {
            key: [path for path in paths if path]
            for key, paths in datasets.items()
            if paths
        }
        self._datasets = filtered
        self._camera_ids = list(filtered.keys())
        self._current_array = None
        self._current_gray = None
        self._current_is_static = False
        self._current_title = ""
        self._current_detection = None

        self.camera_combo.blockSignals(True)
        self.camera_combo.clear()
        self.camera_combo.addItems(self._camera_ids)
        self.camera_combo.blockSignals(False)

        has_data = bool(self._camera_ids)
        self.camera_combo.setEnabled(has_data and len(self._camera_ids) > 1)
        self.prev_camera_btn.setEnabled(has_data and len(self._camera_ids) > 1)
        self.next_camera_btn.setEnabled(has_data and len(self._camera_ids) > 1)
        self.image_slider.setEnabled(has_data)

        if not has_data:
            self.clear()
            return

        index = self._camera_ids.index(selected_key) if selected_key in filtered else 0
        self.camera_combo.setCurrentIndex(index)
        self._update_slider_for_current_camera()
        self._render_current_dataset_image()

    def show_static_image(self, image_path: str, title: Optional[str] = None):
        self._current_is_static = True
        self.camera_combo.setEnabled(False)
        self.prev_camera_btn.setEnabled(False)
        self.next_camera_btn.setEnabled(False)
        self.image_slider.setEnabled(False)
        self.image_index_label.setText("-")
        self.info_label.setText(title or os.path.basename(image_path))
        self.pixel_label.setText("像素值: --")
        self._request_preview_image(
            image_path,
            title or os.path.basename(image_path),
            with_colorbar=False,
        )

    def enable_ruler(self, enabled: bool = True):
        self._ruler_enabled = enabled
        self._ruler_dragging = False
        if enabled:
            self.info_label.setText(f"{self._current_title} | 标尺: 拖动鼠标测量像素距离")
        elif self._current_title:
            self.info_label.setText(self._current_title)

    def clear_ruler(self):
        self._ruler_points = []
        self._ruler_dragging = False
        self._ruler_distance_px = 0.0
        self.rulerDistanceChanged.emit(0.0)
        self._redraw_current()

    def ruler_distance_px(self) -> float:
        return float(self._ruler_distance_px)

    def _current_camera_key(self) -> Optional[str]:
        if not self._camera_ids:
            return None
        index = self.camera_combo.currentIndex()
        if index < 0 or index >= len(self._camera_ids):
            return None
        return self._camera_ids[index]

    def _update_slider_for_current_camera(self):
        camera_key = self._current_camera_key()
        count = len(self._datasets.get(camera_key, [])) if camera_key else 0
        self.image_slider.blockSignals(True)
        self.image_slider.setRange(0, max(count - 1, 0))
        self.image_slider.setValue(0)
        self.image_slider.blockSignals(False)
        self.image_index_label.setText(f"1 / {count}" if count else "0 / 0")

    def _render_current_dataset_image(self):
        camera_key = self._current_camera_key()
        if camera_key is None:
            self._render_placeholder("未加载标定图像")
            return

        image_paths = self._datasets.get(camera_key, [])
        if not image_paths:
            self._render_placeholder("当前视图没有图像")
            return

        image_index = min(self.image_slider.value(), len(image_paths) - 1)
        image_path = image_paths[image_index]
        title = f"{camera_key} | {os.path.basename(image_path)}"
        self._current_is_static = False
        self.image_index_label.setText(f"{image_index + 1} / {len(image_paths)}")
        self.info_label.setText(title)
        self.pixel_label.setText("像素值: --")
        self._request_preview_image(image_path, title, with_colorbar=True)

    def _request_preview_image(self, image_path: str, title: str, with_colorbar: bool):
        self._preview_request_id += 1
        request_id = self._preview_request_id
        self._current_title = title
        self._current_with_colorbar = with_colorbar
        self._current_array = None
        self._current_gray = None
        self._render_placeholder("正在加载预览...")

        loader = PreviewImageLoader(request_id, image_path, self._detection_config)
        loader.loaded.connect(
            lambda rid, path, image, detection, label=title, colorbar=with_colorbar:
                self._on_preview_loaded(rid, path, image, detection, label, colorbar)
        )
        loader.error.connect(self._on_preview_error)
        loader.finished.connect(lambda worker=loader: self._cleanup_preview_loader(worker))
        self._preview_loaders.append(loader)
        loader.start()

    def _cleanup_preview_loader(self, loader):
        if loader in self._preview_loaders:
            self._preview_loaders.remove(loader)

    def _on_preview_error(self, request_id: int, _image_path: str):
        if request_id != self._preview_request_id:
            return
        self._current_array = None
        self._current_gray = None
        self._render_placeholder("无法读取预览图像")

    def _on_preview_loaded(self, request_id: int, _image_path: str, image, detection, title: str, with_colorbar: bool):
        if request_id != self._preview_request_id:
            return

        self._current_array = image
        self._current_detection = detection
        if image.ndim == 2:
            self._current_gray = image
        elif image.ndim == 3 and image.shape[2] >= 3:
            self._current_gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)
        else:
            self._current_gray = np.squeeze(image)

        render_source = self._current_gray if with_colorbar else image
        self._render_array(render_source, title=title, with_colorbar=with_colorbar)
        self._update_detection_status()

    def _render_array(self, array, title: str = "", with_colorbar: bool = True):
        if array is None:
            self._render_placeholder("预览区域")
            return

        self.figure.clear()
        self.axes = self.figure.add_subplot(111)
        self._colorbar = None

        if array.ndim == 2:
            image_artist = self.axes.imshow(array, cmap="gray", origin="upper")
            if with_colorbar:
                self._colorbar = self.figure.colorbar(image_artist, ax=self.axes)
                self._colorbar.set_label("Pixel value")
        elif array.ndim == 3 and array.shape[2] >= 3:
            rgb = cv2.cvtColor(array[:, :, :3], cv2.COLOR_BGR2RGB)
            self.axes.imshow(rgb, origin="upper")
        else:
            squeezed = np.squeeze(array)
            image_artist = self.axes.imshow(squeezed, cmap="gray", origin="upper")
            if with_colorbar:
                self._colorbar = self.figure.colorbar(image_artist, ax=self.axes)
                self._colorbar.set_label("Pixel value")

        self.axes.set_title(title)
        self.axes.set_xlabel("X (px)")
        self.axes.set_ylabel("Y (px)")
        self._draw_detection_overlay()
        self._draw_ruler_overlay()
        self.canvas.draw_idle()

    def _render_placeholder(self, text: str):
        self.figure.clear()
        self.axes = self.figure.add_subplot(111)
        self.axes.text(0.5, 0.5, text, ha="center", va="center", transform=self.axes.transAxes)
        self.axes.set_axis_off()
        self.canvas.draw_idle()

    def _redraw_current(self):
        if self._current_array is None:
            return
        source = self._current_gray if self._current_with_colorbar else self._current_array
        self._render_array(source, title=self._current_title, with_colorbar=self._current_with_colorbar)

    def _show_context_menu(self, pos):
        menu = QMenu(self)
        toggle_action = menu.addAction("关闭标尺" if self._ruler_enabled else "打开标尺")
        clear_action = menu.addAction("清除标尺")
        clear_action.setEnabled(bool(self._ruler_points))
        action = menu.exec_(self.canvas.mapToGlobal(pos))
        if action == toggle_action:
            self.enable_ruler(not self._ruler_enabled)
        elif action == clear_action:
            self.clear_ruler()

    def _show_previous_camera(self):
        if not self._camera_ids:
            return
        index = (self.camera_combo.currentIndex() - 1) % len(self._camera_ids)
        self.camera_combo.setCurrentIndex(index)

    def _show_next_camera(self):
        if not self._camera_ids:
            return
        index = (self.camera_combo.currentIndex() + 1) % len(self._camera_ids)
        self.camera_combo.setCurrentIndex(index)

    def _on_camera_changed(self, _index: int):
        self._update_slider_for_current_camera()
        self._render_current_dataset_image()

    def _on_slider_changed(self, _value: int):
        if self._current_is_static:
            return
        self._render_current_dataset_image()

    def _on_mouse_press(self, event):
        if not self._ruler_enabled or event.inaxes != self.axes or event.button != 1:
            return
        if event.xdata is None or event.ydata is None:
            return
        self._ruler_points = [
            (float(event.xdata), float(event.ydata)),
            (float(event.xdata), float(event.ydata)),
        ]
        self._ruler_dragging = True
        self._update_ruler_distance()
        self._redraw_current()

    def _on_mouse_release(self, event):
        if not self._ruler_enabled or not self._ruler_dragging:
            return
        self._ruler_dragging = False
        if event.inaxes == self.axes and event.xdata is not None and event.ydata is not None:
            self._ruler_points[-1] = (float(event.xdata), float(event.ydata))
        self._update_ruler_distance()
        self._redraw_current()

    def _on_mouse_move(self, event):
        if event.inaxes != self.axes or self._current_array is None:
            self.pixel_label.setText("像素值: --")
            return

        if event.xdata is None or event.ydata is None:
            self.pixel_label.setText("像素值: --")
            return

        if self._ruler_enabled and self._ruler_dragging and len(self._ruler_points) == 2:
            self._ruler_points[-1] = (float(event.xdata), float(event.ydata))
            self._update_ruler_distance()
            self._redraw_current()

        x = int(np.clip(round(event.xdata), 0, self._current_array.shape[1] - 1))
        y = int(np.clip(round(event.ydata), 0, self._current_array.shape[0] - 1))
        value = self._current_array[y, x]

        if np.isscalar(value):
            gray_value = int(self._current_gray[y, x]) if self._current_gray is not None else int(value)
            self.pixel_label.setText(f"x={x}, y={y}, value={gray_value}")
        else:
            raw_values = np.asarray(value).tolist()
            gray_value = int(self._current_gray[y, x]) if self._current_gray is not None else "-"
            self.pixel_label.setText(
                f"x={x}, y={y}, gray={gray_value}, raw={raw_values}"
            )

    def _draw_ruler_overlay(self):
        if len(self._ruler_points) != 2:
            return
        (x1, y1), (x2, y2) = self._ruler_points
        self.axes.plot([x1, x2], [y1, y2], color="#ff3b30", linewidth=2)
        self.axes.scatter([x1, x2], [y1, y2], color="#ff3b30", s=28)
        if self._ruler_distance_px > 0:
            self.axes.text(
                (x1 + x2) / 2,
                (y1 + y2) / 2,
                f"{self._ruler_distance_px:.2f} px",
                color="white",
                bbox={"facecolor": "#222", "alpha": 0.75, "pad": 3},
            )

    def _draw_detection_overlay(self):
        if not self._current_detection or not self._current_detection.get("points"):
            return

        points = np.array(self._current_detection["points"], dtype=np.float32)
        if points.size == 0:
            return

        self.axes.scatter(
            points[:, 0],
            points[:, 1],
            s=34,
            facecolors="none",
            edgecolors="#00d084",
            linewidths=1.6,
        )

        point_ids = self._current_detection.get("point_ids", [])
        for index, point in enumerate(points):
            if index < len(point_ids):
                label = ",".join(f"{float(value):g}" for value in point_ids[index])
            else:
                label = str(index + 1)
            self.axes.text(
                point[0] + 4,
                point[1] - 4,
                label,
                color="#ffd60a",
                fontsize=7,
                bbox={"facecolor": "#111", "alpha": 0.55, "pad": 1},
            )

    def _update_detection_status(self):
        if not self._current_detection:
            self.info_label.setText(f"{self._current_title} | 已识别圆点/角点")
            return
        if self._current_detection.get("error"):
            self.info_label.setText(f"{self._current_title} | 识别失败")
            return
        count = len(self._current_detection.get("points", []))
        self.info_label.setText(f"{self._current_title} | 已识别 {count} 个圆点/角点")

    def _update_ruler_distance(self):
        if len(self._ruler_points) != 2:
            self._ruler_distance_px = 0.0
        else:
            (x1, y1), (x2, y2) = self._ruler_points
            self._ruler_distance_px = float(np.hypot(x2 - x1, y2 - y1))
        self.rulerDistanceChanged.emit(self._ruler_distance_px)


SUPPORTED_IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.pgm', '.ppm'}


class FileTreePanel(QWidget):
    """
    文件树面板：展示工作目录下的图像文件。
    - 按子文件夹（图像组）分组，可展开/收缩
    - 点击图片节点时发出信号供主窗口预览
    - 右键菜单：展开/收缩/复制路径/重命名
    - 惰性加载：点击组节点时才发出预览信号，展开/收缩不触发预览
    """

    # 信号：点击了一张图片，参数为图片绝对路径
    image_selected = pyqtSignal(str)
    # 信号：点击了一个图像组（文件夹），参数为该文件夹下第一张图片路径
    group_selected = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._work_dir: str = ""
        self._load_timer = QTimer(self)
        self._load_timer.setSingleShot(True)
        self._load_timer.setInterval(100)  # 防抖 100ms
        self._pending_path: str = ""
        self._load_timer.timeout.connect(self._emit_pending_signal)

        # 文件系统监视器：工作目录内容变化时自动刷新
        self._fs_watcher = QFileSystemWatcher(self)
        self._fs_watcher.directoryChanged.connect(self._on_dir_changed)
        self._fs_watcher.fileChanged.connect(self._on_file_changed)

        # 防抖定时器：避免短时间内多次触发 refresh
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(500)  # 500ms 防抖
        self._refresh_timer.timeout.connect(self.refresh)

        self._build_ui()

    # ------------------------------------------------------------------
    # UI 构建
    # ------------------------------------------------------------------
    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # 标题行
        header = QWidget()
        header.setStyleSheet("background-color: #1e2d3d;")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(8, 6, 6, 6)
        header_layout.setSpacing(4)

        title = QLabel("📁 文件树")
        title.setStyleSheet("color: #ecf0f1; font-size: 12px; font-weight: bold;")
        header_layout.addWidget(title, stretch=1)

        # 刷新按钮
        self._refresh_btn = QPushButton("⟳")
        self._refresh_btn.setToolTip("刷新文件树")
        self._refresh_btn.setFixedSize(24, 24)
        self._refresh_btn.setStyleSheet("""
            QPushButton {
                background: transparent; color: #bdc3c7;
                border: none; font-size: 14px;
            }
            QPushButton:hover { color: #ecf0f1; }
        """)
        self._refresh_btn.clicked.connect(self.refresh)
        header_layout.addWidget(self._refresh_btn)

        # 全部展开/收缩
        self._expand_btn = QPushButton("⊞")
        self._expand_btn.setToolTip("全部展开/收缩")
        self._expand_btn.setFixedSize(24, 24)
        self._expand_btn.setCheckable(True)
        self._expand_btn.setStyleSheet("""
            QPushButton {
                background: transparent; color: #bdc3c7;
                border: none; font-size: 14px;
            }
            QPushButton:hover { color: #ecf0f1; }
            QPushButton:checked { color: #3498db; }
        """)
        self._expand_btn.clicked.connect(self._toggle_expand_all)
        header_layout.addWidget(self._expand_btn)

        layout.addWidget(header)

        # 路径标签
        self._path_label = QLabel("未设置工作目录")
        self._path_label.setStyleSheet(
            "color: #7f8c8d; font-size: 10px; padding: 3px 8px; "
            "background-color: #263545;"
        )
        self._path_label.setWordWrap(True)
        self._path_label.setMaximumHeight(36)
        layout.addWidget(self._path_label)

        # 树形控件
        self._tree = QTreeWidget()
        self._tree.setHeaderHidden(True)
        self._tree.setIndentation(14)
        self._tree.setAnimated(True)
        self._tree.setUniformRowHeights(True)  # 统一行高提升性能
        self._tree.setIconSize(QSize(16, 16))
        self._tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self._tree.customContextMenuRequested.connect(self._on_context_menu)
        self._tree.setStyleSheet("""
            QTreeWidget {
                background-color: #1a2535;
                color: #bdc3c7;
                border: none;
                font-size: 12px;
            }
            QTreeWidget::item {
                padding: 3px 4px;
                border-radius: 3px;
            }
            QTreeWidget::item:hover {
                background-color: #2c3e50;
                color: #ecf0f1;
            }
            QTreeWidget::item:selected {
                background-color: #2980b9;
                color: #ffffff;
            }
            QTreeWidget::branch {
                background-color: #1a2535;
            }
            QTreeWidget::branch:has-children:!has-siblings:closed,
            QTreeWidget::branch:closed:has-children:has-siblings {
                border-image: none;
                image: none;
            }
            QScrollBar:vertical {
                background: #1a2535; width: 8px; border-radius: 4px;
            }
            QScrollBar::handle:vertical {
                background: #34495e; border-radius: 4px; min-height: 20px;
            }
            QScrollBar::handle:vertical:hover { background: #4a6785; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)
        self._tree.itemClicked.connect(self._on_item_clicked)
        layout.addWidget(self._tree, stretch=1)

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------
    def set_work_dir(self, path: str):
        """设置工作目录并刷新树，同时启动文件系统监视。"""
        # 取消监视旧目录
        old_dirs = self._fs_watcher.directories()
        if old_dirs:
            self._fs_watcher.removePaths(old_dirs)
        old_files = self._fs_watcher.files()
        if old_files:
            self._fs_watcher.removePaths(old_files)

        self._work_dir = path

        # 开始监视新工作目录
        if path and os.path.isdir(path):
            self._fs_watcher.addPath(path)
            # 同时监视所有直属子目录
            try:
                for sub in Path(path).iterdir():
                    if sub.is_dir() and not sub.name.startswith('.'):
                        self._fs_watcher.addPath(str(sub))
            except Exception:
                pass

        self.refresh()

    def watch_extra_dir(self, path: str):
        """追加监视一个额外目录（如批处理输出目录）。"""
        if path and os.path.isdir(path) and path not in self._fs_watcher.directories():
            self._fs_watcher.addPath(path)

    def _on_dir_changed(self, path: str):
        """目录内容变化时触发防抖刷新，并确保新子目录被监视。"""
        # 若是新增子目录，追加监视
        if os.path.isdir(path) and path not in self._fs_watcher.directories():
            self._fs_watcher.addPath(path)
        self._refresh_timer.start()

    def _on_file_changed(self, path: str):
        """单个文件变化（如重命名后旧路径通知）时触发防抖刷新。"""
        self._refresh_timer.start()

    def refresh(self):
        """重新扫描工作目录，重建树节点。"""
        self._tree.setUpdatesEnabled(False)  # 批量更新期间禁止重绘
        self._tree.clear()
        if not self._work_dir or not os.path.isdir(self._work_dir):
            self._path_label.setText("未设置工作目录")
            self._tree.setUpdatesEnabled(True)
            return

        self._path_label.setText(self._work_dir)

        root_dir = Path(self._work_dir)

        # 刷新后重新同步监视的子目录列表
        watched_dirs = set(self._fs_watcher.directories())
        for sub in root_dir.iterdir():
            if sub.is_dir() and not sub.name.startswith('.'):
                s = str(sub)
                if s not in watched_dirs:
                    self._fs_watcher.addPath(s)

        # 先收集根目录下直属图片
        root_images = sorted(
            p for p in root_dir.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTS
        )

        # 再收集子文件夹
        subdirs = sorted(
            p for p in root_dir.iterdir()
            if p.is_dir() and not p.name.startswith('.')
        )

        # 根目录直属图片单独分组
        if root_images:
            group_item = QTreeWidgetItem(self._tree)
            group_item.setText(0, f"📂 {root_dir.name}  ({len(root_images)} 张)")
            group_item.setData(0, Qt.UserRole, str(root_dir))
            group_item.setData(0, Qt.UserRole + 1, "group")
            group_item.setForeground(0, QColor("#3498db"))
            group_item.setExpanded(False)
            for img_path in root_images:
                self._add_image_item(group_item, img_path)

        # 子文件夹
        for sub in subdirs:
            imgs = sorted(
                p for p in sub.iterdir()
                if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTS
            )
            if not imgs:
                continue
            group_item = QTreeWidgetItem(self._tree)
            group_item.setText(0, f"📂 {sub.name}  ({len(imgs)} 张)")
            group_item.setData(0, Qt.UserRole, str(sub))
            group_item.setData(0, Qt.UserRole + 1, "group")
            group_item.setForeground(0, QColor("#3498db"))
            group_item.setExpanded(False)
            for img_path in imgs:
                self._add_image_item(group_item, img_path)

        if self._tree.topLevelItemCount() == 0:
            empty_item = QTreeWidgetItem(self._tree)
            empty_item.setText(0, "（无图像文件）")
            empty_item.setForeground(0, QColor("#7f8c8d"))

        self._tree.setUpdatesEnabled(True)  # 恢复重绘

    def _add_image_item(self, parent: QTreeWidgetItem, img_path: Path):
        item = QTreeWidgetItem(parent)
        item.setText(0, f"  🖼 {img_path.name}")
        item.setData(0, Qt.UserRole, str(img_path))
        item.setData(0, Qt.UserRole + 1, "image")
        item.setForeground(0, QColor("#95a5a6"))

    def _toggle_expand_all(self, checked: bool):
        self._tree.setUpdatesEnabled(False)
        if checked:
            self._tree.expandAll()
            self._expand_btn.setText("⊟")
        else:
            self._tree.collapseAll()
            self._expand_btn.setText("⊞")
        self._tree.setUpdatesEnabled(True)

    # ------------------------------------------------------------------
    # 点击事件（带防抖，避免快速点击导致堆积加载）
    # ------------------------------------------------------------------
    def _on_item_clicked(self, item: QTreeWidgetItem, col: int):
        path = item.data(0, Qt.UserRole)
        kind = item.data(0, Qt.UserRole + 1)
        if not path:
            return

        if kind == "image":
            self._pending_path = path
            self._pending_kind = "image"
            self._load_timer.start()  # 防抖
        elif kind == "group":
            # 点击组节点：仅展开/收缩，不触发预览
            # 预览需通过右键菜单或点击子图片触发
            item.setExpanded(not item.isExpanded())

    def _emit_pending_signal(self):
        """防抖定时器到期，实际发出信号。"""
        path = self._pending_path
        kind = getattr(self, '_pending_kind', 'image')
        if not path:
            return
        if kind == "image":
            self.image_selected.emit(path)
        elif kind == "group":
            self.group_selected.emit(path)

    # ------------------------------------------------------------------
    # 右键菜单
    # ------------------------------------------------------------------
    def _on_context_menu(self, pos):
        """右键菜单：展开/收缩/复制路径/重命名。"""
        item = self._tree.itemAt(pos)
        if item is None:
            return

        path = item.data(0, Qt.UserRole)
        kind = item.data(0, Qt.UserRole + 1)

        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu {
                background-color: #1e2d3d;
                color: #ecf0f1;
                border: 1px solid #34495e;
                padding: 4px;
            }
            QMenu::item {
                padding: 6px 24px;
                border-radius: 3px;
            }
            QMenu::item:selected {
                background-color: #2980b9;
            }
            QMenu::separator {
                height: 1px;
                background: #34495e;
                margin: 4px 8px;
            }
        """)

        # --- 展开/收缩（组节点才有子节点） ---
        if item.childCount() > 0:
            if item.isExpanded():
                act_collapse = menu.addAction("🔽 收缩")
                act_collapse.triggered.connect(lambda: item.setExpanded(False))
            else:
                act_expand = menu.addAction("▶ 展开")
                act_expand.triggered.connect(lambda: item.setExpanded(True))

            # 展开所有 / 收缩所有
            menu.addSeparator()
            act_expand_all = menu.addAction("⊟ 展开所有")
            act_expand_all.triggered.connect(self._tree.expandAll)
            act_collapse_all = menu.addAction("⊞ 收缩所有")
            act_collapse_all.triggered.connect(self._tree.collapseAll)

        # --- 预览（组节点预览第一张，图片节点直接预览） ---
        if path:
            menu.addSeparator()
            if kind == "group":
                act_preview = menu.addAction("👁 预览第一张")
                act_preview.triggered.connect(
                    lambda: self._preview_group(item))
            elif kind == "image":
                act_preview = menu.addAction("👁 预览图片")
                act_preview.triggered.connect(
                    lambda: self.image_selected.emit(path))

        # --- 复制路径 ---
        if path:
            menu.addSeparator()
            act_copy = menu.addAction("📋 复制路径")
            act_copy.triggered.connect(lambda: self._copy_path(path))

        # --- 重命名 ---
        if path and kind == "image":
            act_rename = menu.addAction("✏️ 重命名")
            act_rename.triggered.connect(lambda: self._rename_item(item, path))

        # --- 删除（移入回收站） ---
        if path:
            act_delete = menu.addAction("🗑 删除")
            act_delete.triggered.connect(lambda: self._delete_item(item, path))

        menu.exec_(self._tree.viewport().mapToGlobal(pos))

    def _preview_group(self, item: QTreeWidgetItem):
        """预览组节点的第一张图片。"""
        child = item.child(0)
        if child:
            first_img = child.data(0, Qt.UserRole)
            if first_img:
                self.group_selected.emit(first_img)

    def _copy_path(self, path: str):
        """复制文件路径到剪贴板。"""
        from PyQt5.QtWidgets import QApplication as _QApp
        clipboard = _QApp.clipboard()
        clipboard.setText(path)
        # 显示状态提示
        if hasattr(self, '_path_label'):
            self._path_label.setText(f"已复制: {path}")
            QTimer.singleShot(2000, lambda: self._path_label.setText(self._work_dir or "未设置工作目录"))

    def _rename_item(self, item: QTreeWidgetItem, old_path: str):
        """重命名图片文件。"""
        from PyQt5.QtWidgets import QInputDialog
        old_p = Path(old_path)
        new_name, ok = QInputDialog.getText(
            self, "重命名", "新文件名:", text=old_p.name)
        if not ok or not new_name or new_name == old_p.name:
            return
        new_path = old_p.parent / new_name
        if new_path.exists():
            QMessageBox.warning(self, "重命名失败", f"文件已存在: {new_path}")
            return
        try:
            old_p.rename(new_path)
            # 更新树节点
            item.setText(0, f"  🖼 {new_name}")
            item.setData(0, Qt.UserRole, str(new_path))
            self._path_label.setText(f"已重命名: {new_name}")
            QTimer.singleShot(2000, lambda: self._path_label.setText(self._work_dir or "未设置工作目录"))
        except OSError as e:
            QMessageBox.warning(self, "重命名失败", str(e))

    def _delete_item(self, item: QTreeWidgetItem, path: str):
        """删除文件（移入系统回收站）。"""
        reply = QMessageBox.question(
            self, "确认删除",
            f"确定要删除以下文件吗？\n{path}\n\n文件将移入系统回收站。",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply != QMessageBox.Yes:
            return
        try:
            import send2trash
            send2trash.send2trash(path)
        except ImportError:
            # send2trash 不可用时回退到永久删除（二次确认）
            reply2 = QMessageBox.warning(
                self, "永久删除",
                "未安装 send2trash 库，文件将永久删除（不可恢复）！\n是否继续？",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply2 == QMessageBox.Yes:
                os.remove(path)
            else:
                return
        except Exception as e:
            QMessageBox.warning(self, "删除失败", str(e))
            return
        # 从树中移除节点
        parent = item.parent()
        if parent:
            parent.removeChild(item)
            # 更新父节点的图片计数
            remaining = parent.childCount()
            if remaining == 0:
                # 组内无图片了，移除组节点
                tree = parent.treeWidget()
                if tree:
                    (tree.invisibleRootItem() if parent.parent() is None
                     else parent.parent()).removeChild(parent)
            else:
                dir_name = Path(parent.data(0, Qt.UserRole)).name
                parent.setText(0, f"📂 {dir_name}  ({remaining} 张)")
        else:
            self._tree.takeTopLevelItem(self._tree.indexOfTopLevelItem(item))


class ImagePreviewPanel(QWidget):
    """窗口最右侧：显示文件树中选中的图片（大图预览），可折叠。"""

    # 信号：面板折叠/展开状态变化，参数(bool)表示是否展开
    collapse_state_changed = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_path: str = ""
        self._expanded: bool = True
        self._expanded_width: int = 320   # 展开时宽度（会被主窗口 setFixedWidth 覆盖）
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ---- 标题栏（始终可见）----
        self._title_bar = QWidget()
        self._title_bar.setFixedHeight(32)
        self._title_bar.setStyleSheet("background-color: #1e2d3d;")
        tb_layout = QHBoxLayout(self._title_bar)
        tb_layout.setContentsMargins(8, 0, 8, 0)
        tb_layout.setSpacing(6)

        # 折叠按钮
        self._collapse_btn = QPushButton("▸")
        self._collapse_btn.setToolTip("折叠/展开预览面板")
        self._collapse_btn.setFixedSize(20, 20)
        self._collapse_btn.setStyleSheet("""
            QPushButton {
                background: transparent; color: #bdc3c7;
                border: none; font-size: 14px; font-weight: bold;
            }
            QPushButton:hover { color: #ecf0f1; }
        """)
        self._collapse_btn.clicked.connect(self._toggle_collapse)
        tb_layout.addWidget(self._collapse_btn)

        title = QLabel("🖼 图片预览")
        title.setStyleSheet("color: #ecf0f1; font-size: 12px; font-weight: bold;")
        tb_layout.addWidget(title, stretch=1)

        self._fit_btn = QPushButton("适应窗口")
        self._fit_btn.setCheckable(True)
        self._fit_btn.setChecked(True)
        self._fit_btn.setFixedSize(72, 24)
        self._fit_btn.setStyleSheet("""
            QPushButton {
                background: transparent; color: #bdc3c7;
                border: 1px solid #34495e; font-size: 11px;
                border-radius: 3px;
            }
            QPushButton:checked { color: #3498db; border-color: #3498db; }
            QPushButton:hover { color: #ecf0f1; }
        """)
        tb_layout.addWidget(self._fit_btn)
        layout.addWidget(self._title_bar)

        # ---- 可折叠的内容区 ----
        self._content = QWidget()
        content_layout = QVBoxLayout(self._content)
        content_layout.setContentsMargins(4, 4, 4, 4)
        content_layout.setSpacing(4)

        # 滚动区域
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("""
            QScrollArea { background-color: #111c28; border: none; }
            QScrollBar:vertical {
                background: #1a2535; width: 8px; border-radius: 4px;
            }
            QScrollBar::handle:vertical {
                background: #34495e; border-radius: 4px; min-height: 20px;
            }
            QScrollBar::handle:vertical:hover { background: #4a6785; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
        """)
        self._scroll = scroll

        self._img_label = QLabel()
        self._img_label.setAlignment(Qt.AlignCenter)
        self._img_label.setStyleSheet("background-color: #111c28; color: #7f8c8d; font-size: 12px;")
        self._img_label.setText("请从左侧文件树选择图片")
        scroll.setWidget(self._img_label)
        content_layout.addWidget(scroll, stretch=1)

        # 路径标签
        self._path_label = QLabel("未选择图片")
        self._path_label.setStyleSheet("color: #7f8c8d; font-size: 10px; padding: 2px 6px;")
        self._path_label.setWordWrap(True)
        self._path_label.setMaximumHeight(32)
        content_layout.addWidget(self._path_label)

        layout.addWidget(self._content, stretch=1)

    # ------------------------------------------------------------------
    # 折叠 / 展开
    # ------------------------------------------------------------------
    def _toggle_collapse(self):
        if self._expanded:
            self.collapse()
        else:
            self.expand()

    def collapse(self):
        """折叠预览面板，只保留标题栏（收缩到最右边）。"""
        self._expanded = False
        self._expanded_width = self.width()
        self._content.hide()
        self._collapse_btn.setText("▸")
        # 收缩到仅标题栏窄条（约10px），紧贴最右边
        self.setFixedWidth(10)
        self.collapse_state_changed.emit(False)

    def expand(self):
        """展开预览面板。"""
        self._expanded = True
        self._content.show()
        self._collapse_btn.setText("◂")
        self.setFixedWidth(self._expanded_width)
        # 重新适配图片
        if hasattr(self, '_orig_pixmap') and not self._orig_pixmap.isNull():
            self._apply_pixmap()
        self.collapse_state_changed.emit(True)

    def is_expanded(self) -> bool:
        return self._expanded

    # ------------------------------------------------------------------
    # 图片加载
    # ------------------------------------------------------------------
    def load_image(self, img_path: str):
        """加载并显示图片。"""
        self._current_path = img_path
        self._path_label.setText(os.path.basename(img_path))

        # 如果当前折叠，自动展开
        if not self._expanded:
            self.expand()

        try:
            pix = QPixmap(img_path)
            if pix.isNull():
                img = robust_imread(img_path, cv2.IMREAD_UNCHANGED)
                if img is not None:
                    # 非8位图像先归一化到0-255以便显示
                    if img.dtype != np.uint8:
                        vmin, vmax = float(img.min()), float(img.max())
                        if vmax > vmin:
                            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                        else:
                            img = np.zeros(img.shape[:2], dtype=np.uint8)
                    if img.ndim == 2:
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                    elif img.ndim == 3 and img.shape[2] == 4:
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
                    elif img.ndim == 3 and img.shape[2] == 1:
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                    else:
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    h, w, ch = img_rgb.shape
                    # 使用 .tobytes() 创建数据副本，避免 numpy 数组被 GC 后 QImage 引用悬垂指针
                    qimg = QImage(img_rgb.data.tobytes(), w, h, w * ch, QImage.Format_RGB888)
                    pix = QPixmap.fromImage(qimg)
        except Exception:
            pix = QPixmap()

        if pix.isNull():
            self._img_label.setText("无法加载图片")
            return

        self._orig_pixmap = pix
        self._apply_pixmap()

    def _apply_pixmap(self):
        if not hasattr(self, '_orig_pixmap') or self._orig_pixmap.isNull():
            return
        if self._fit_btn.isChecked():
            scaled = self._orig_pixmap.scaled(
                self._img_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        else:
            scaled = self._orig_pixmap
        self._img_label.setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._expanded and hasattr(self, '_fit_btn') and self._fit_btn.isChecked():
            self._apply_pixmap()

    def clear(self):
        self._current_path = ""
        self._path_label.setText("未选择图片")
        self._img_label.setText("请从左侧文件树选择图片")
        if hasattr(self, '_orig_pixmap'):
            self._orig_pixmap = QPixmap()


class _IEColorbarWidget(QWidget):
    """图像处理查看器的颜色条组件，显示 min-max 值映射。"""

    def __init__(self, sp_func, parent=None):
        super().__init__(parent)
        self._sp = sp_func
        self._vmin = 0.0
        self._vmax = 255.0
        self.setMinimumWidth(self._sp(200))

    def set_range(self, vmin, vmax):
        self._vmin = vmin
        self._vmax = vmax
        self.update()

    def paintEvent(self, event):
        from PyQt5.QtGui import QPainter, QLinearGradient, QPen
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        h = self.height()
        margin = self._sp(40)
        bar_h = self._sp(8)
        y0 = (h - bar_h) // 2

        # 渐变色条（灰度：黑→白）
        gradient = QLinearGradient(margin, 0, w - margin, 0)
        gradient.setColorAt(0.0, QColor(0, 0, 0))
        gradient.setColorAt(1.0, QColor(255, 255, 255))
        painter.fillRect(margin, y0, w - 2 * margin, bar_h, gradient)

        # 边框
        painter.setPen(QPen(QColor(100, 100, 100), 1))
        painter.drawRect(margin, y0, w - 2 * margin, bar_h)

        # 标签
        painter.setPen(QColor(170, 170, 170))
        font = painter.font()
        font.setPointSize(max(7, int(9 * self._sp(1) / 1.0)))
        painter.setFont(font)

        vmin_str = f"{self._vmin:.0f}" if abs(self._vmin) < 1e4 else f"{self._vmin:.1e}"
        vmax_str = f"{self._vmax:.0f}" if abs(self._vmax) < 1e4 else f"{self._vmax:.1e}"

        painter.drawText(0, y0 + bar_h + self._sp(12), vmin_str)
        painter.drawText(w - margin, y0 + bar_h + self._sp(12), vmax_str)
        painter.drawText(w // 2, y0 + bar_h + self._sp(12), f"{(self._vmin + self._vmax) / 2:.0f}")

        painter.end()


class _IEImageViewer(QLabel):
    """图像处理查看器的增强QLabel，支持鼠标悬停显示像素坐标和值。"""

    pixel_hover = pyqtSignal(int, int, object)  # x, y, pixel_value

    def __init__(self, text="", sp_func=None, parent=None):
        super().__init__(text, parent)
        self._sp = sp_func or (lambda x: x)
        self.setAlignment(Qt.AlignCenter)
        self.setMouseTracking(True)
        self._image_data = None  # numpy array
        self._scale_x = 1.0
        self._scale_y = 1.0
        self._offset_x = 0
        self._offset_y = 0

    def set_image_data(self, img):
        """设置原始图像数据用于像素值读取。"""
        self._image_data = img

    def setPixmap(self, pixmap):
        """重写setPixmap，记录缩放信息。"""
        super().setPixmap(pixmap)
        if pixmap and self._image_data is not None:
            ih, iw = self._image_data.shape[:2]
            pw = pixmap.width()
            ph = pixmap.height()
            if pw > 0 and ph > 0:
                self._scale_x = iw / pw
                self._scale_y = ih / ph
                # 计算图片在label中的偏移（居中显示）
                self._offset_x = (self.width() - pw) // 2
                self._offset_y = (self.height() - ph) // 2

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # 重新计算偏移
        if self.pixmap() and self._image_data is not None:
            pw = self.pixmap().width()
            ph = self.pixmap().height()
            self._offset_x = (self.width() - pw) // 2
            self._offset_y = (self.height() - ph) // 2

    def mouseMoveEvent(self, event):
        """鼠标移动时发出像素坐标和值。"""
        if self._image_data is None or self.pixmap() is None:
            self.pixel_hover.emit(-1, -1, None)
            return

        x = event.pos().x() - self._offset_x
        y = event.pos().y() - self._offset_y

        pw = self.pixmap().width()
        ph = self.pixmap().height()

        if 0 <= x < pw and 0 <= y < ph:
            img_x = int(x * self._scale_x)
            img_y = int(y * self._scale_y)
            ih, iw = self._image_data.shape[:2]
            if 0 <= img_y < ih and 0 <= img_x < iw:
                val = self._image_data[img_y, img_x]
                if isinstance(val, np.ndarray):
                    val = tuple(val.tolist())
                self.pixel_hover.emit(img_x, img_y, val)
                return

        self.pixel_hover.emit(-1, -1, None)

    def leaveEvent(self, event):
        self.pixel_hover.emit(-1, -1, None)


class _IEAlgoCard(QFrame):
    """操作面板算法卡片，支持点击和拖拽。"""

    add_requested = pyqtSignal(str)  # step_key

    def __init__(self, key, label, desc, sp_func, parent=None):
        super().__init__(parent)
        self._sp = sp_func
        self._key = key
        self._label = label
        self.setCursor(Qt.PointingHandCursor)
        self.setProperty("algo_key", key)
        self.setStyleSheet(
            "QFrame { background: #383838; border: 1px solid #4a4a4a; "
            "  border-radius: 4px; padding: 4px; }"
            "QFrame:hover { background: #454545; border-color: #6a6a6a; }"
            "QLabel { color: #ddd; border: none; }"
        )
        c_lay = QVBoxLayout(self)
        c_lay.setContentsMargins(self._sp(6), self._sp(4), self._sp(6), self._sp(4))
        c_lay.setSpacing(0)

        name_lbl = QLabel(f"<b>{label}</b>")
        name_lbl.setStyleSheet("font-size: 12px; color: #eee; border: none;")
        c_lay.addWidget(name_lbl)

        desc_lbl = QLabel(desc)
        desc_lbl.setStyleSheet("font-size: 10px; color: #999; border: none;")
        c_lay.addWidget(desc_lbl)

    def mousePressEvent(self, event):
        """点击添加到工作流。"""
        if event.button() == Qt.LeftButton:
            self.add_requested.emit(self._key)
            self._drag_start = event.pos()

    def mouseMoveEvent(self, event):
        """拖拽算法卡片到工作流画布。"""
        if not (event.buttons() & Qt.LeftButton):
            return
        drag = QDrag(self)
        mime = drag.mimeData()
        mime.setText(self._key)
        # 创建拖拽预览
        pixmap = self.grab()
        drag.setPixmap(pixmap.scaled(
            min(pixmap.width(), 150), min(pixmap.height(), 60),
            Qt.KeepAspectRatio, Qt.SmoothTransformation))
        drag.setHotSpot(event.pos())
        drag.exec_(Qt.CopyAction)


class _IEWorkflowCanvas(QWidget):
    """工作流画布，支持拖拽添加节点和节点拖拽排序。"""

    node_drop_requested = pyqtSignal(str)  # step_key
    node_move_requested = pyqtSignal(str, int)  # step_key, new_index

    def __init__(self, sp_func, parent=None):
        super().__init__(parent)
        self._sp = sp_func
        self.setAcceptDrops(True)
        self._drag_insert_line = None
        self._drag_insert_idx = -1

    def dragEnterEvent(self, event):
        if event.mimeData().hasText():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasText():
            event.acceptProposedAction()
            # 计算插入位置指示
            self._update_drop_indicator(event.pos())
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasText():
            step_key = event.mimeData().text()
            event.acceptProposedAction()
            # 计算插入位置
            insert_idx = self._calc_insert_index(event.pos())
            self.node_drop_requested.emit(step_key)
            self._clear_drop_indicator()
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        self._clear_drop_indicator()

    def _calc_insert_index(self, pos):
        """根据鼠标位置计算插入到画布中的节点索引。"""
        # 查找 ie_canvas_layout
        content = self.findChild(QScrollArea)
        if not content:
            return -1
        canvas_content = content.widget()
        if not canvas_content:
            return -1
        layout = canvas_content.layout()
        if not layout:
            return -1

        # 映射坐标到 canvas_content
        local_pos = self.mapTo(canvas_content, pos)
        y = local_pos.y()

        count = layout.count()
        for i in range(count):
            item = layout.itemAt(i)
            if item and item.widget():
                widget = item.widget()
                rect = widget.geometry()
                if y < rect.center().y():
                    return i
        return count - 1  # 在 stretch 之前

    def _update_drop_indicator(self, pos):
        """显示插入位置指示线。"""
        self._clear_drop_indicator()
        idx = self._calc_insert_index(pos)
        if idx < 0:
            return
        content = self.findChild(QScrollArea)
        if not content:
            return
        canvas_content = content.widget()
        if not canvas_content:
            return
        layout = canvas_content.layout()
        if not layout or idx >= layout.count():
            return

        item = layout.itemAt(idx)
        if item and item.widget():
            widget = item.widget()
            # 创建指示线
            line = QFrame(canvas_content)
            line.setFixedHeight(3)
            line.setStyleSheet("background: #4CAF50; border: none; border-radius: 1px;")
            # 插入到布局中
            layout.insertWidget(idx, line)
            self._drag_insert_line = line
            self._drag_insert_idx = idx

    def _clear_drop_indicator(self):
        """清除插入位置指示线。"""
        if self._drag_insert_line is not None:
            self._drag_insert_line.deleteLater()
            self._drag_insert_line = None
        self._drag_insert_idx = -1


class BubbleTomographyGUI(QMainWindow):
    """气泡三维层析重建系统主窗口"""

    @staticmethod
    def _detect_ui_scale() -> float:
        """根据 DPI 和屏幕尺寸检测 UI 缩放比例。"""
        app = QApplication.instance()
        if app is None:
            return 1.0

        screen = app.primaryScreen()
        if screen is None:
            return 1.0

        geometry = screen.availableGeometry()
        logical_dpi = screen.logicalDotsPerInch() or 96.0
        dpi_scale = logical_dpi / 96.0
        resolution_scale = min(
            geometry.width() / 1920.0,
            geometry.height() / 1080.0
        )

        scale = dpi_scale
        if geometry.width() >= 3200 or geometry.height() >= 1800:
            scale = max(scale, min(1.35, resolution_scale))

        return max(1.0, min(scale, 1.5))

    def _sp(self, px: int) -> int:
        """按 UI 缩放比例返回像素值。"""
        return max(1, int(round(px * self.ui_scale)))

    def _scale_stylesheet(self, stylesheet: str) -> str:
        """缩放样式表中的 px 值。"""
        if not stylesheet:
            return stylesheet

        def repl(match):
            return f"{self._sp(int(match.group(1)))}px"

        return re.sub(r"(\d+)px", repl, stylesheet)

    def _apply_scaled_fonts(self):
        """设置全局字体，适配 4K 屏幕。"""
        app = QApplication.instance()
        if app is None:
            return

        font = QFont("Microsoft YaHei")
        font.setStyleStrategy(QFont.PreferAntialias)
        font.setPointSizeF(max(10.0, min(15.0, 10.0 * self.ui_scale)))
        app.setFont(font)

    def _scale_existing_stylesheets(self):
        """缩放已有控件上的样式表。"""
        widgets = [self, *self.findChildren(QWidget)]
        for widget in widgets:
            stylesheet = widget.styleSheet()
            if stylesheet:
                widget.setStyleSheet(self._scale_stylesheet(stylesheet))

    def __init__(self):
        super().__init__()
        self.ui_scale = self._detect_ui_scale()
        self.setWindowTitle("三维多相流场测量软件")
        self.setMinimumSize(1200, 800)

        # 数据存储
        self.calibrator: Optional[MultiCameraCalibrator] = None
        self.reconstructor: Optional[TomographicReconstructor] = None
        self.image_processor = BubbleImageProcessor()
        self.visualizer = ResultVisualizer()

        self.camera_calib_images: Dict[str, List[str]] = {}
        self.camera_bubble_images: Dict[str, np.ndarray] = {}
        self.camera_reference_images: Dict[str, np.ndarray] = {}
        self.calibration_results: Dict[str, CameraParams] = {}
        self.projections: Dict[str, np.ndarray] = {}

        # 批量时间序列数据
        self.bubble_timepoint_images: Dict[int, Dict[str, np.ndarray]] = {}
        self.bubble_timepoint_names: Dict[int, str] = {}  # {index: folder_name}
        self.particle_timepoint_images: Dict[int, Dict[str, np.ndarray]] = {}
        self.particle_timepoint_names: Dict[int, str] = {}
        self.particle_sequence_paths: Dict[str, List[str]] = {}
        self.particle_active_camera_ids: List[str] = []
        self.bubble_batch_results: Dict[int, dict] = {}
        self.piv_batch_results: Dict[int, dict] = {}
        self.current_bubble_timepoint = 0
        self.current_piv_timepoint = 0

        # 单相机标定数据
        self.single_camera_params: Optional[dict] = None   # {camera_matrix, dist_coeffs, rms}
        self.stereo_params: Optional[dict] = None          # {R, T, E, F, rms}

        # 射线追踪数据
        self.rt_processor: Optional[RaytraceProcessor] = None
        self.rt_compute_thread = None

        # 通用图像编辑器数据
        self.ie_config: ImageEditConfig = ImageEditConfig()
        self.ie_single_path: str = ""           # 当前单张图像路径
        self.ie_operand_path: str = ""          # 加减法第二张图路径
        self.ie_src_dir: str = ""               # 批量输入目录
        self.ie_dst_dir: str = ""               # 批量输出目录
        self.ie_worker_thread = None            # 批量处理后台线程
        self._ie_current_preview: Optional[np.ndarray] = None  # 处理后预览图

        # 二维PIV数据
        self.piv2d_frame1_path: str = ""
        self.piv2d_frame2_path: str = ""
        self.piv2d_src_dir: str = ""
        self.piv2d_dst_dir: str = ""
        self.piv2d_time_groups: List[dict] = []
        self.piv2d_worker_thread = None
        self._piv2d_last_result: Optional[dict] = None
        self._piv2d_last_image: Optional[np.ndarray] = None
        self._piv2d_exclusion_mask: Optional[np.ndarray] = None
        self._piv2d_processing_start = 0.0
        self._piv2d_processing_prefix = "二维PIV"
        self._piv2d_elapsed_timer = QTimer(self)
        self._piv2d_elapsed_timer.timeout.connect(self._piv2d_update_elapsed_time)

        # 工作目录
        self._work_dir: str = ""

        self._init_ui()
        self._create_menubar()
        self._apply_global_style()
        self._apply_scaled_fonts()
        self._scale_existing_stylesheets()

    def _init_ui(self):
        """初始化 UI 主布局：左侧导航 + 右侧内容页。"""
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # === 左侧导航 ===
        self._nav_panel = QWidget()
        self._nav_panel.setFixedWidth(self._sp(180))
        self._nav_panel.setStyleSheet("""
            QWidget#navPanel {
                background-color: #2c3e50;
            }
        """)
        self._nav_panel.setObjectName("navPanel")
        nav_layout = QVBoxLayout(self._nav_panel)
        nav_layout.setContentsMargins(
            self._sp(8), self._sp(16), self._sp(8), self._sp(16)
        )
        nav_layout.setSpacing(self._sp(4))

        # ??
        title_label = QLabel("三维多相流场测量软件")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet(
            "color: #ecf0f1; font-size: 14px; font-weight: bold; "
            "padding: 8px 0px 16px 0px;"
        )
        nav_layout.addWidget(title_label)

        nav_layout.addWidget(self._make_separator())

        # 导航按钮
        self._nav_btn_group = QButtonGroup(self)
        self._nav_btn_group.setExclusive(True)

        self._nav_buttons = {}
        nav_items = [
            ("calibration",    "相机标定"),
            ("reconstruction", "气泡重建"),
            ("raytrace",       "单相机3D重建"),
            ("particle",       "Particle / PIV"),
            ("piv2d",          "二维PIV"),
            ("image_editor",   "图像处理"),
        ]

        for page_id, text in nav_items:
            btn = QPushButton(text)
            btn.setCheckable(True)
            btn.setMinimumHeight(self._sp(44))
            btn.setCursor(Qt.PointingHandCursor)
            btn.setStyleSheet("""
                QPushButton {
                    text-align: left;
                    padding: 8px 14px;
                    border: none;
                    border-radius: 6px;
                    color: #bdc3c7;
                    font-size: 13px;
                    background-color: transparent;
                }
                QPushButton:hover {
                    background-color: #34495e;
                    color: #ecf0f1;
                }
                QPushButton:checked {
                    background-color: #3498db;
                    color: white;
                    font-weight: bold;
                }
            """)
            self._nav_btn_group.addButton(btn)
            self._nav_buttons[page_id] = btn
            nav_layout.addWidget(btn)

        image_editor_btn = self._nav_buttons.get("image_editor")
        if image_editor_btn is not None:
            nav_layout.removeWidget(image_editor_btn)
            nav_layout.insertWidget(0, image_editor_btn)

        nav_layout.addStretch()

        # 版本信息
        ver_label = QLabel("v1.0")
        ver_label.setAlignment(Qt.AlignCenter)
        ver_label.setStyleSheet("color: #7f8c8d; font-size: 11px; padding: 8px;")
        nav_layout.addWidget(ver_label)

        # === 右侧内?(QStackedWidget) ===
        self.content_stack = QStackedWidget()
        self.content_stack.setStyleSheet("QStackedWidget { border: none; }")

        # Page 0: 相机标定
        self._create_calibration_page()

        # Page 1: 气泡重建
        self._create_reconstruction_page()

        # Page 2: 单相?D重建（射线追踼
        self._create_raytrace_page()

        # Page 3: 粒子追踪与PIV
        self._create_particle_page()

        # Page 4: 二维PIV
        self._create_piv2d_page()

        # Page 5: 通用图像处理
        self._create_image_editor_page()
        image_editor_page = self.content_stack.widget(5)
        if image_editor_page is not None:
            self.content_stack.removeWidget(image_editor_page)
            self.content_stack.insertWidget(0, image_editor_page)
            self.content_stack.setCurrentIndex(0)

        # 默认选中图像处理
        self._nav_buttons["image_editor"].setChecked(True)

        # === 文件树面板（导航栏右侧） ===
        self._file_tree_panel = FileTreePanel()
        self._file_tree_panel.setFixedWidth(self._sp(220))
        self._file_tree_panel.image_selected.connect(self._on_filetree_image_selected)
        self._file_tree_panel.group_selected.connect(self._on_filetree_image_selected)

        # === 图片预览面板（最右侧）===
        self._preview_panel = ImagePreviewPanel()
        self._preview_panel.setFixedWidth(self._sp(320))
        self._file_tree_panel.image_selected.connect(self._preview_panel.load_image)
        self._file_tree_panel.group_selected.connect(self._preview_panel.load_image)

        # 同步面板自带折叠按钮的状态
        self._preview_panel.collapse_state_changed.connect(self._sync_preview_toggle_state_from_panel)

        main_layout.addWidget(self._nav_panel)
        main_layout.addWidget(self._file_tree_panel)
        main_layout.addWidget(self.content_stack, stretch=1)
        main_layout.addWidget(self._preview_panel)

        # 连接导航切换
        self._nav_btn_group.buttonClicked.connect(self._on_nav_changed)

        # 状态栏
        self.statusBar().showMessage("就绪")

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.statusBar().addPermanentWidget(self.progress_bar)
        self.processing_time_label = QLabel("处理时间: --")
        self.statusBar().addPermanentWidget(self.processing_time_label)

    def _make_separator(self):
        """创建分隔线。"""
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("color: #4a6785;")
        return line

    # 注：_on_nav_changed 定义在文件末尾（使用正确的页面索引映射）

    def _set_work_directory(self):
        """设置工作目录，并刷新文件树。"""
        directory = QFileDialog.getExistingDirectory(
            self,
            "选择工作目录",
            self._work_dir or os.path.expanduser("~"),
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )
        if directory:
            self._work_dir = directory
            self._file_tree_panel.set_work_dir(directory)
            self.statusBar().showMessage(f"工作目录已设置: {directory}")

    def _on_filetree_image_selected(self, img_path: str):
        """文件树点击图片时，在右侧当前页面中加载预览（通用处理）。"""
        if not os.path.isfile(img_path):
            return
        # 若当前在图像处理页面，则直接加载到图像处理预览
        current_idx = self.content_stack.currentIndex()
        try:
            if current_idx == 0:  # 图像处理页（已移动到 index 0）
                if hasattr(self, 'ie_single_path') and hasattr(self, '_ie_load_orig_preview'):
                    self.ie_single_path = img_path
                    if hasattr(self, 'ie_single_label'):
                        self.ie_single_label.setText(os.path.basename(img_path))
                        self.ie_single_label.setStyleSheet("")
                    self._ie_load_orig_preview(img_path)
                    if hasattr(self, '_ie_request_preview'):
                        self._ie_request_preview()
                    return
        except Exception:
            pass
        # 其他情况：弹出一个简易图像查看对话框
        self._show_image_viewer(img_path)

    def _show_image_viewer(self, img_path: str):
        """弹出独立图像查看窗口（用于文件树预览）。"""
        from PyQt5.QtWidgets import QDialog
        dlg = QDialog(self)
        dlg.setWindowTitle(os.path.basename(img_path))
        dlg.resize(800, 600)
        layout = QVBoxLayout(dlg)

        label = QLabel()
        label.setAlignment(Qt.AlignCenter)
        label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        try:
            img = robust_imread(img_path, cv2.IMREAD_UNCHANGED)
            if img is not None:
                # 非8位图像先归一化到0-255以便显示
                if img.dtype != np.uint8:
                    vmin, vmax = float(img.min()), float(img.max())
                    if vmax > vmin:
                        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    else:
                        img = np.zeros(img.shape[:2], dtype=np.uint8)
                if img.ndim == 2:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                elif img.ndim == 3 and img.shape[2] == 4:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
                elif img.ndim == 3 and img.shape[2] == 1:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                else:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                h, w, ch = img_rgb.shape
                # 使用 .tobytes() 创建数据副本，避免 numpy 数组被 GC 后 QImage 引用悬垂指针
                qimg = QImage(img_rgb.data.tobytes(), w, h, w * ch, QImage.Format_RGB888)
                pix = QPixmap.fromImage(qimg)
                scaled = pix.scaled(780, 560, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                label.setPixmap(scaled)
            else:
                label.setText("无法读取图像")
        except Exception as e:
            label.setText(f"预览失败: {e}")

        layout.addWidget(label)
        info = QLabel(img_path)
        info.setStyleSheet("color: #888; font-size: 10px;")
        info.setAlignment(Qt.AlignCenter)
        layout.addWidget(info)
        dlg.exec_()
        return

    def _create_menubar(self):
        """创建菜单栏。"""
        menubar = self.menuBar()

        # ===== 文件菜单 =====
        file_menu = menubar.addMenu("文件(&F)")

        # --- 设置工作目录（最顶部） ---
        set_workdir_action = QAction("设置工作目录(&W)...", self)
        set_workdir_action.setShortcut("Ctrl+W")
        set_workdir_action.triggered.connect(self._set_work_directory)
        file_menu.addAction(set_workdir_action)

        file_menu.addSeparator()

        save_calib = QAction("保存标定...", self)
        save_calib.setShortcut("Ctrl+S")
        save_calib.triggered.connect(self._save_calibration)
        file_menu.addAction(save_calib)

        load_calib = QAction("加载标定...", self)
        load_calib.setShortcut("Ctrl+O")
        load_calib.triggered.connect(self._load_calibration)
        file_menu.addAction(load_calib)

        file_menu.addSeparator()

        export_ply = QAction("导出点云 (PLY)...", self)
        export_ply.triggered.connect(lambda: self._export_point_cloud('ply'))
        file_menu.addAction(export_ply)

        export_pcd = QAction("导出点云 (PCD)...", self)
        export_pcd.triggered.connect(lambda: self._export_point_cloud('pcd'))
        file_menu.addAction(export_pcd)

        file_menu.addSeparator()

        quit_action = QAction("退出", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        # ===== 编辑菜单 =====
        edit_menu = menubar.addMenu("编辑(&E)")

        clear_log_action = QAction("清空当前日志", self)
        clear_log_action.triggered.connect(self._clear_current_log)
        edit_menu.addAction(clear_log_action)

        # ===== 窗口菜单 =====
        window_menu = menubar.addMenu("窗口(&W)")

        nav_to_calib = QAction("相机标定", self)
        nav_to_calib.triggered.connect(lambda: self._navigate_to(0))
        window_menu.addAction(nav_to_calib)

        nav_to_recon = QAction("气泡重建", self)
        nav_to_recon.triggered.connect(lambda: self._navigate_to(1))
        window_menu.addAction(nav_to_recon)

        nav_to_raytrace = QAction("单相机3D重建", self)
        nav_to_raytrace.triggered.connect(lambda: self._navigate_to(2))
        window_menu.addAction(nav_to_raytrace)

        nav_to_particle = QAction("Particle / PIV", self)
        nav_to_particle.triggered.connect(lambda: self._navigate_to(3))
        window_menu.addAction(nav_to_particle)

        nav_to_piv2d = QAction("二维PIV", self)
        nav_to_piv2d.triggered.connect(lambda: self._navigate_to(4))
        window_menu.addAction(nav_to_piv2d)

        nav_to_ie = QAction("图像处理", self)
        nav_to_ie.triggered.connect(lambda: self._navigate_to(5))
        window_menu.addAction(nav_to_ie)

        window_menu.addSeparator()

        # "图片预览 适应窗口" 切换（可勾选）
        self._toggle_preview_action = QAction("图片预览 适应窗口", self)
        self._toggle_preview_action.setCheckable(True)
        self._toggle_preview_action.setChecked(True)
        self._toggle_preview_action.triggered.connect(self._on_toggle_preview)
        window_menu.addAction(self._toggle_preview_action)

        # ===== 帮助菜单 =====
        help_menu = menubar.addMenu("帮助(&H)")

        open_manual_action = QAction("打开PDF说明...", self)
        open_manual_action.triggered.connect(self._open_pdf_manual)
        help_menu.addAction(open_manual_action)

        help_menu.addSeparator()

        about_action = QAction("关于", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

        # ---- 菜单栏右侧：预览切换按钮 ----
        corner_widget = QWidget()
        corner_widget.setStyleSheet("background: transparent;")
        corner_layout = QHBoxLayout(corner_widget)
        corner_layout.setContentsMargins(0, 0, 8, 0)
        corner_layout.setSpacing(0)

        self._preview_toggle_btn = QPushButton("图片预览 适应窗口")
        self._preview_toggle_btn.setCheckable(True)
        self._preview_toggle_btn.setChecked(True)
        self._preview_toggle_btn.setFixedHeight(24)
        self._preview_toggle_btn.setCursor(Qt.PointingHandCursor)
        self._preview_toggle_btn.setStyleSheet("""
            QPushButton {
                background-color: #1e2d3d;
                color: #bdc3c7;
                border: 1px solid #34495e;
                border-radius: 3px;
                padding: 2px 12px;
                font-size: 12px;
            }
            QPushButton:checked {
                color: #3498db;
                border-color: #3498db;
            }
            QPushButton:hover {
                color: #ecf0f1;
                border-color: #4a6785;
            }
        """)
        self._preview_toggle_btn.clicked.connect(self._on_toggle_preview)
        corner_layout.addWidget(self._preview_toggle_btn)
        menubar.setCornerWidget(corner_widget, Qt.TopRightCorner)

    def _on_toggle_preview(self):
        """切换右侧预览面板的可见性。"""
        if self._preview_panel.is_expanded():
            self._preview_panel.collapse()
        else:
            self._preview_panel.expand()
        self._sync_preview_toggle_state()

    def _sync_preview_toggle_state(self):
        """同步菜单项和按钮的勾选状态到预览面板。"""
        visible = self._preview_panel.is_expanded()
        self._toggle_preview_action.setChecked(visible)
        if hasattr(self, '_preview_toggle_btn'):
            self._preview_toggle_btn.setChecked(visible)

    def _sync_preview_toggle_state_from_panel(self, expanded: bool):
        """由面板信号触发：同步菜单和按钮状态。"""
        self._toggle_preview_action.setChecked(expanded)
        if hasattr(self, '_preview_toggle_btn'):
            self._preview_toggle_btn.setChecked(expanded)

    def _apply_global_style(self):
        """应用全局样式 — DaVis 10 深色主题。"""
        self.setStyleSheet(self._scale_stylesheet("""
            /* === 全局字体 === */
            * {
                font-family: "Microsoft YaHei", "SimHei", "Segoe UI", sans-serif;
                font-size: 13px;
            }
            /* === DaVis 深色主题核心 === */
            QMainWindow {
                background-color: #1e1e1e;
            }
            QWidget {
                background-color: #2b2b2b;
                color: #ddd;
            }
            QMenuBar {
                background-color: #2b2b2b;
                color: #ddd;
                border-bottom: 1px solid #444;
                font-size: 13px;
                padding: 2px;
            }
            QMenuBar::item {
                padding: 4px 12px;
                background: transparent;
            }
            QMenuBar::item:selected {
                background: #3c3c3c;
                border-radius: 3px;
            }
            QMenu {
                background-color: #2b2b2b;
                color: #ddd;
                border: 1px solid #444;
                font-size: 13px;
            }
            QMenu::item {
                padding: 5px 30px 5px 20px;
            }
            QMenu::item:selected {
                background-color: #3c3c3c;
            }
            QMenu::separator {
                height: 1px;
                background: #444;
                margin: 4px 10px;
            }
            QStatusBar {
                background-color: #2b2b2b;
                color: #aaa;
                border-top: 1px solid #444;
                font-size: 12px;
            }
            QGroupBox {
                font-size: 13px;
                font-weight: bold;
                color: #ccc;
                border: 1px solid #444;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 15px;
                background-color: #2b2b2b;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: #ccc;
            }
            QLabel {
                font-size: 13px;
                color: #ddd;
                background-color: transparent;
            }
            QPushButton {
                font-size: 13px;
                padding: 5px 14px;
                background-color: #3c3c3c;
                color: #ddd;
                border: 1px solid #555;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #4a4a4a;
                border-color: #666;
            }
            QPushButton:pressed {
                background-color: #555;
            }
            QPushButton:disabled {
                background-color: #333;
                color: #666;
            }
            QLineEdit {
                font-size: 13px;
                padding: 4px 8px;
                background-color: #3c3c3c;
                color: #eee;
                border: 1px solid #555;
                border-radius: 4px;
            }
            QComboBox {
                font-size: 13px;
                padding: 4px 8px;
                background-color: #3c3c3c;
                color: #eee;
                border: 1px solid #555;
                border-radius: 4px;
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            QComboBox QAbstractItemView {
                background-color: #3c3c3c;
                color: #eee;
                border: 1px solid #555;
                selection-background-color: #4a4a4a;
            }
            QSpinBox, QDoubleSpinBox {
                font-size: 13px;
                padding: 3px 6px;
                background-color: #3c3c3c;
                color: #eee;
                border: 1px solid #555;
                border-radius: 4px;
            }
            QTextEdit {
                font-size: 12px;
                background-color: #1e1e1e;
                color: #ccc;
                border: 1px solid #444;
                border-radius: 4px;
            }
            QListWidget {
                font-size: 13px;
                background-color: #1e1e1e;
                color: #ddd;
                border: 1px solid #444;
                border-radius: 4px;
            }
            QListWidget::item {
                padding: 3px;
            }
            QListWidget::item:selected {
                background-color: #3c3c3c;
                color: #fff;
            }
            QListWidget::item:hover {
                background-color: #333;
            }
            QSlider::groove:horizontal {
                background: #555;
                height: 6px;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #2196F3;
                width: 14px;
                margin: -4px 0;
                border-radius: 7px;
            }
            QSlider::handle:horizontal:hover {
                background: #42A5F5;
            }
            QProgressBar {
                font-size: 12px;
                background-color: #2b2b2b;
                border: 1px solid #444;
                border-radius: 4px;
                text-align: center;
                color: #ddd;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 3px;
            }
            QTabWidget::pane {
                border: 1px solid #444;
                background-color: #2b2b2b;
            }
            QTabWidget::tab {
                font-size: 13px;
                padding: 6px 16px;
                background-color: #333;
                color: #aaa;
                border: 1px solid #444;
                border-bottom: none;
                border-radius: 4px 4px 0 0;
                margin-right: 2px;
            }
            QTabWidget::tab:selected {
                background-color: #3c3c3c;
                color: #fff;
            }
            QTabWidget::tab:hover {
                background-color: #444;
            }
            QToolBar {
                font-size: 13px;
                background-color: #2b2b2b;
                border-bottom: 1px solid #444;
                spacing: 4px;
                padding: 2px;
            }
            QScrollArea {
                font-size: 13px;
                border: none;
                background-color: transparent;
            }
            QScrollBar:vertical {
                background: #2b2b2b;
                width: 8px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical {
                background: #555;
                border-radius: 4px;
                min-height: 30px;
            }
            QScrollBar::handle:vertical:hover {
                background: #666;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QScrollBar:horizontal {
                background: #2b2b2b;
                height: 8px;
                border-radius: 4px;
            }
            QScrollBar::handle:horizontal {
                background: #555;
                border-radius: 4px;
                min-width: 30px;
            }
            QScrollBar::handle:horizontal:hover {
                background: #666;
            }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
                width: 0px;
            }
            QSplitter::handle {
                background: #444;
            }
            QSplitter::handle:horizontal {
                width: 2px;
            }
            QSplitter::handle:vertical {
                height: 2px;
            }
            QCheckBox {
                color: #ddd;
                spacing: 6px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border-radius: 3px;
                border: 1px solid #666;
                background: #3c3c3c;
            }
            QCheckBox::indicator:checked {
                background: #4CAF50;
                border-color: #4CAF50;
            }
            QCheckBox::indicator:hover {
                border-color: #888;
            }
            QRadioButton {
                color: #ddd;
                spacing: 6px;
            }
            QRadioButton::indicator {
                width: 14px;
                height: 14px;
                border-radius: 7px;
                border: 1px solid #666;
                background: #3c3c3c;
            }
            QRadioButton::indicator:checked {
                background: #2196F3;
                border-color: #2196F3;
            }
            QToolTip {
                background-color: #3c3c3c;
                color: #eee;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 4px;
            }
            QTreeWidget {
                background-color: #1e1e1e;
                color: #ddd;
                border: 1px solid #444;
                border-radius: 4px;
            }
            QTreeWidget::item {
                padding: 2px;
            }
            QTreeWidget::item:selected {
                background-color: #3c3c3c;
            }
            QTreeWidget::item:hover {
                background-color: #333;
            }
            QHeaderView::section {
                background-color: #333;
                color: #ddd;
                border: 1px solid #444;
                padding: 4px;
            }
        """))

    # 注：_navigate_to 定义在文件末尾（使用正确的页面索引映射）

    def _open_pdf_manual(self):
        pdf_path, _ = QFileDialog.getOpenFileName(
            self,
            self, "选择PDF说明", "",
            "",
            "PDF文件 (*.pdf)"
        )
        if not pdf_path:
            return

        try:
            os.startfile(pdf_path)
            self.statusBar().showMessage(
                f"已打说明? {os.path.basename(pdf_path)}"
            )
        except Exception as e:
            QMessageBox.critical(self, "打开失败", f"无法打开PDF说明?\n{e}")

    # 注：_clear_current_log 定义在文件末尾（使用正确的页面索引映射）

    def _show_about(self):
        """显示关于对话框。"""
        QMessageBox.about(
            self, "关于",
            "三维多相流场测量软件 v1.0\n\n"
            "功能：相机标定 -> 气泡图像预处理 -> MART层析重建\n"
            "     -> 3D点云输出 -> 示踪粒子重建 -> 互相关速度场\n\n"
            "作者: OpenAI Codex"
        )

    def _create_calibration_page(self):
        """创建标定页：左侧参数 + 右侧预览。"""
        page = QWidget()
        layout = QHBoxLayout(page)

        # 左侧: 参数设置
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # 标定模式选择
        mode_group = QGroupBox("标定模式")
        mode_layout = QVBoxLayout()

        self.calib_mode_combo = QComboBox()
        self.calib_mode_combo.addItems([
            "多相机联合标定 (3+相机)",
            "单相机标定",
            "双目标定 (Stereo)"
        ])
        self.calib_mode_combo.currentIndexChanged.connect(
            self._on_calib_mode_changed
        )
        mode_layout.addWidget(self.calib_mode_combo)

        self.calib_mode_desc = QLabel(
            "使用多个相机同时标定，获取各相机内参和相对位姿。\n"
            "适用于 Tomographic PIV 多相机系统。"
        )
        self.calib_mode_desc.setStyleSheet("color: #666; font-size: 12px;")
        self.calib_mode_desc.setWordWrap(True)
        mode_layout.addWidget(self.calib_mode_desc)

        mode_group.setLayout(mode_layout)
        left_layout.addWidget(mode_group)

        # 单相机标定区域（默认隐藏）
        self.single_calib_group = QGroupBox("单相机标定")
        sc_layout = QVBoxLayout()
        sc_layout.addWidget(QLabel("使用单个相机拍摄棋盘格标定板图像序列，获取内参和畸变系数。"))
        btn_load_single_calib = QPushButton("加载标定图像...")
        btn_load_single_calib.clicked.connect(self._load_single_calib_images)
        sc_layout.addWidget(btn_load_single_calib)
        self.single_calib_label = QLabel("未加载图像")
        self.single_calib_label.setStyleSheet("color: gray;")
        sc_layout.addWidget(self.single_calib_label)
        self.single_calib_pixel_label = QLabel("标尺像素距离: 未测量")
        self.single_calib_pixel_label.setStyleSheet("color: #666;")
        sc_layout.addWidget(self.single_calib_pixel_label)

        actual_size_row = QHBoxLayout()
        actual_size_row.addWidget(QLabel("实际尺寸 (mm):"))
        self.single_actual_size_spin = QDoubleSpinBox()
        self.single_actual_size_spin.setRange(0.001, 100000.0)
        self.single_actual_size_spin.setDecimals(4)
        self.single_actual_size_spin.setValue(1.0)
        self.single_actual_size_spin.setSingleStep(0.1)
        actual_size_row.addWidget(self.single_actual_size_spin, stretch=1)
        sc_layout.addLayout(actual_size_row)

        btn_run_single_calib = QPushButton("开始单相机标定")
        btn_run_single_calib.setStyleSheet(
            "QPushButton { background-color: #FF9800; color: white; padding: 8px; border-radius: 5px; }"
            "QPushButton:hover { background-color: #F57C00; }"
        )
        btn_run_single_calib.clicked.connect(self._run_single_calibration)
        sc_layout.addWidget(btn_run_single_calib)
        self.single_calib_group.setLayout(sc_layout)
        self.single_calib_group.setVisible(False)
        left_layout.addWidget(self.single_calib_group)

        # 双目标定区域（默认隐藏）
        self.stereo_calib_group = QGroupBox("双目标定 (Stereo)")
        st_layout = QVBoxLayout()
        st_layout.addWidget(QLabel("加载左右相机拍摄的棋盘格图像对，进行立体标定。"))

        btn_load_stereo_left = QPushButton("加载左相机标定图像...")
        btn_load_stereo_left.clicked.connect(self._load_stereo_images_left)
        st_layout.addWidget(btn_load_stereo_left)

        btn_load_stereo_right = QPushButton("加载右相机标定图像...")
        btn_load_stereo_right.clicked.connect(self._load_stereo_images_right)
        st_layout.addWidget(btn_load_stereo_right)

        self.stereo_left_label = QLabel("左相机: 未加载")
        self.stereo_left_label.setStyleSheet("color: gray;")
        st_layout.addWidget(self.stereo_left_label)

        self.stereo_right_label = QLabel("右相机: 未加载")
        self.stereo_right_label.setStyleSheet("color: gray;")
        st_layout.addWidget(self.stereo_right_label)

        btn_run_stereo = QPushButton("开始双目标定")
        btn_run_stereo.setStyleSheet(
            "QPushButton { background-color: #9C27B0; color: white; padding: 8px; border-radius: 5px; }"
            "QPushButton:hover { background-color: #7B1FA2; }"
        )
        btn_run_stereo.clicked.connect(self._run_stereo_calibration)
        st_layout.addWidget(btn_run_stereo)
        self.stereo_calib_group.setLayout(st_layout)
        self.stereo_calib_group.setVisible(False)
        left_layout.addWidget(self.stereo_calib_group)

        # 标定板参数
        pattern_group = QGroupBox("标定板参数")
        grid = QGridLayout()

        grid.addWidget(QLabel("标定板类型:"), 0, 0)
        self.pattern_type_combo = QComboBox()
        self.pattern_type_combo.addItems([
            "checkerboard (棋盘格)",
            "circles (对称圆点阵)",
            "acircles (非对称圆点阵)",
            "volume_dots (体标定板点阵)"
        ])
        grid.addWidget(self.pattern_type_combo, 0, 1)

        grid.addWidget(QLabel("宽度方向角点数:"), 1, 0)
        self.pattern_w_spin = QSpinBox()
        self.pattern_w_spin.setRange(3, 50)
        self.pattern_w_spin.setValue(11)
        grid.addWidget(self.pattern_w_spin, 1, 1)

        grid.addWidget(QLabel("高度方向角点数:"), 2, 0)
        self.pattern_h_spin = QSpinBox()
        self.pattern_h_spin.setRange(3, 50)
        self.pattern_h_spin.setValue(8)
        grid.addWidget(self.pattern_h_spin, 2, 1)

        grid.addWidget(QLabel("方格/间距大小 (mm):"), 3, 0)
        self.square_size_spin = QDoubleSpinBox()
        self.square_size_spin.setRange(0.1, 100)
        self.square_size_spin.setValue(5.0)
        self.square_size_spin.setSingleStep(0.5)
        grid.addWidget(self.square_size_spin, 3, 1)

        grid.addWidget(QLabel("层间距 (mm):"), 4, 0)
        self.level_separation_spin = QDoubleSpinBox()
        self.level_separation_spin.setRange(0.01, 100)
        self.level_separation_spin.setValue(1.0)
        self.level_separation_spin.setSingleStep(0.1)
        grid.addWidget(self.level_separation_spin, 4, 1)

        self.pattern_type_combo.currentIndexChanged.connect(
            lambda _index: self._refresh_calibration_preview()
        )
        self.pattern_w_spin.valueChanged.connect(
            lambda _value: self._refresh_calibration_preview()
        )
        self.pattern_h_spin.valueChanged.connect(
            lambda _value: self._refresh_calibration_preview()
        )
        self.square_size_spin.valueChanged.connect(
            lambda _value: self._refresh_calibration_preview()
        )
        self.level_separation_spin.valueChanged.connect(
            lambda _value: self._refresh_calibration_preview()
        )

        pattern_group.setLayout(grid)
        left_layout.addWidget(pattern_group)

        # === 多相机标定区域（默?===
        self.multi_calib_container = QWidget()
        multi_layout = QVBoxLayout(self.multi_calib_container)
        multi_layout.setContentsMargins(0, 0, 0, 0)

        # 相机管理
        camera_group = QGroupBox("相机管理")
        cam_layout = QVBoxLayout()

        cam_btn_layout = QHBoxLayout()
        self.cam_id_input = QLineEdit()
        self.cam_id_input.setPlaceholderText("相机ID (如 cam1)")
        cam_btn_layout.addWidget(self.cam_id_input)

        btn_add_cam = QPushButton("添加相机")
        btn_add_cam.clicked.connect(self._add_camera)
        cam_btn_layout.addWidget(btn_add_cam)

        btn_remove_cam = QPushButton("移除选中")
        btn_remove_cam.clicked.connect(self._remove_camera)
        cam_btn_layout.addWidget(btn_remove_cam)

        cam_layout.addLayout(cam_btn_layout)

        self.camera_list = QListWidget()
        cam_layout.addWidget(self.camera_list)

        btn_load_calib_images = QPushButton("加载选中相机的标定图像...")
        btn_load_calib_images.clicked.connect(self._load_calibration_images)
        cam_layout.addWidget(btn_load_calib_images)

        self.calib_image_label = QLabel("未加载标定图像")
        self.calib_image_label.setStyleSheet("color: gray;")
        cam_layout.addWidget(self.calib_image_label)

        camera_group.setLayout(cam_layout)
        multi_layout.addWidget(camera_group)

        # 标定按钮
        btn_calibrate = QPushButton("开始标定")
        btn_calibrate.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; "
            "font-size: 14px; padding: 10px; border-radius: 5px; }"
            "QPushButton:hover { background-color: #45a049; }"
        )
        btn_calibrate.clicked.connect(self._run_calibration)
        multi_layout.addWidget(btn_calibrate)

        left_layout.addWidget(self.multi_calib_container)

        left_layout.addStretch()

        # 右侧: 标定结果 + 预览
        right_panel, self.calib_result_text, self.calib_preview_label = \
            self._make_right_panel_with_viz(
                log_placeholder="标定结果将在此处显示",
                viz_context="calibration"
            )

        layout.addWidget(left_panel, stretch=1)
        layout.addWidget(right_panel, stretch=2)

        self.content_stack.addWidget(page)

    def _make_right_panel_with_viz(self, log_placeholder="", viz_context=""):
        """创建右侧面板：日志 + 预览 + 可视化工具。"""
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(
            self._sp(6), self._sp(6), self._sp(6), self._sp(6)
        )

        # 上部: 日志 + 预览工具栏
        top_splitter = QSplitter(Qt.Horizontal)

        # 日志区
        log_text = QTextEdit()
        log_text.setReadOnly(True)
        log_text.setFontFamily("Consolas")
        log_text.setPlaceholderText(log_placeholder)
        top_splitter.addWidget(log_text)

        # 预览区（在日志旁边）
        if viz_context == "calibration":
            preview_widget = CalibrationPreviewWidget()
            preview_widget.rulerDistanceChanged.connect(self._on_calib_ruler_distance_changed)
            preview_widget.setStyleSheet(
                "border: 1px solid #ccc; background-color: #f8f8f8; padding: 4px;"
            )
            top_splitter.addWidget(preview_widget)
        else:
            preview_widget = QLabel()
            preview_widget.setAlignment(Qt.AlignCenter)
            preview_widget.setMinimumSize(self._sp(400), self._sp(300))
            preview_widget.setStyleSheet(
                "border: 1px solid #ccc; background-color: #f8f8f8; padding: 10px;"
            )
            preview_widget.setText("预览区域")
            top_splitter.addWidget(preview_widget)

        top_splitter.setSizes([self._sp(500), self._sp(500)])
        right_layout.addWidget(top_splitter, stretch=1)

        # 下部: 可视化工具栏
        viz_bar = self._make_viz_toolbar(viz_context)
        if viz_bar:
            right_layout.addWidget(viz_bar)

        return right_panel, log_text, preview_widget

    def _make_viz_toolbar(self, context=""):
        """创建可视化工具栏。"""
        toolbar_widget = QWidget()
        toolbar_layout = QHBoxLayout(toolbar_widget)
        toolbar_layout.setContentsMargins(
            self._sp(8), self._sp(4), self._sp(8), self._sp(4)
        )

        btn_style = (
            "QPushButton { padding: 5px 12px; border: 1px solid #ddd; "
            "border-radius: 4px; background-color: #f5f5f5; font-size: 12px; }"
            "QPushButton:hover { background-color: #e0e0e0; }"
        )

        if context == "calibration":
            title = "标定工具"
            label = QLabel(title)
            label.setStyleSheet("font-weight: bold; color: #555; padding-right: 8px;")
            toolbar_layout.addWidget(label)

            btn = QPushButton("📊 综合报告")
            btn.setStyleSheet(btn_style)
            btn.clicked.connect(self._show_report)
            toolbar_layout.addWidget(btn)

            toolbar_layout.addStretch()
            return toolbar_widget

        elif context == "reconstruction":
            title = "重建工具"
            label = QLabel(title)
            label.setStyleSheet("font-weight: bold; color: #555; padding-right: 8px;")
            toolbar_layout.addWidget(label)

            btn_pc = QPushButton("显示点云")
            btn_pc.setStyleSheet(btn_style)
            btn_pc.clicked.connect(self._show_point_cloud)
            toolbar_layout.addWidget(btn_pc)

            btn_slices = QPushButton("体素切片")
            btn_slices.setStyleSheet(btn_style)
            btn_slices.clicked.connect(self._show_volume_slices)
            toolbar_layout.addWidget(btn_slices)

            btn_proj = QPushButton("投影对比")
            btn_proj.setStyleSheet(btn_style)
            btn_proj.clicked.connect(self._show_projections)
            toolbar_layout.addWidget(btn_proj)

            btn_report = QPushButton("综合报告")
            btn_report.setStyleSheet(btn_style)
            btn_report.clicked.connect(self._show_report)
            toolbar_layout.addWidget(btn_report)

            btn_batch = QPushButton("批量概览")
            btn_batch.setStyleSheet(btn_style)
            btn_batch.clicked.connect(self._show_batch_summary)
            toolbar_layout.addWidget(btn_batch)

            toolbar_layout.addStretch()
            return toolbar_widget

        elif context == "raytrace":
            title = "光线追踪工具"
            label = QLabel(title)
            label.setStyleSheet("font-weight: bold; color: #555; padding-right: 8px;")
            toolbar_layout.addWidget(label)

            btn_3d = QPushButton("3D曲面")
            btn_3d.setStyleSheet(btn_style)
            btn_3d.clicked.connect(self._show_rt_3d_view)
            toolbar_layout.addWidget(btn_3d)

            btn_report = QPushButton("📊 综合报告")
            btn_report.setStyleSheet(btn_style)
            btn_report.clicked.connect(self._show_rt_report)
            toolbar_layout.addWidget(btn_report)

            btn_export = QPushButton("导出点云")
            btn_export.setStyleSheet(btn_style)
            btn_export.clicked.connect(self._export_rt_points)
            toolbar_layout.addWidget(btn_export)

            btn_export_fig = QPushButton("导出3D图")
            btn_export_fig.setStyleSheet(btn_style)
            btn_export_fig.clicked.connect(self._export_rt_figure)
            toolbar_layout.addWidget(btn_export_fig)

            toolbar_layout.addStretch()
            return toolbar_widget

        elif context == "particle":
            title = "粒子PIV工具"
            label = QLabel(title)
            label.setStyleSheet("font-weight: bold; color: #555; padding-right: 8px;")
            toolbar_layout.addWidget(label)

            btn_pc = QPushButton("粒子3D")
            btn_pc.setStyleSheet(btn_style)
            btn_pc.clicked.connect(self._show_particle_viz)
            toolbar_layout.addWidget(btn_pc)

            btn_vf = QPushButton("速度场")
            btn_vf.setStyleSheet(btn_style)
            btn_vf.clicked.connect(self._show_velocity_viz)
            toolbar_layout.addWidget(btn_vf)

            btn_batch = QPushButton("批量概览")
            btn_batch.setStyleSheet(btn_style)
            btn_batch.clicked.connect(self._show_batch_summary)
            toolbar_layout.addWidget(btn_batch)

            toolbar_layout.addStretch()
            return toolbar_widget

        return None

    def _create_reconstruction_page(self):
        """创建重建页面：左侧参数 + 右侧日志/可视化 + 时间点选择。"""
        page = QWidget()
        layout = QHBoxLayout(page)

        # 左侧: 重建参数
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # 图像加载（批量模式）
        image_group = QGroupBox("气泡图像（批量加载）")
        img_layout = QVBoxLayout()

        btn_load_bubble_batch = QPushButton("📁 批量加载气泡图像序列...")
        btn_load_bubble_batch.setStyleSheet(
            "QPushButton { font-weight: bold; padding: 6px; }"
        )
        btn_load_bubble_batch.clicked.connect(self._load_bubble_images_batch)
        img_layout.addWidget(btn_load_bubble_batch)

        self.bubble_batch_info = QLabel(
            "选择包含多时刻子文件夹的根目录\n"
            "子文件夹名作为时间点标识（如 t000, t001, ...）\n"
            "每个子文件夹内包含各相机的图像"
        )
        self.bubble_batch_info.setStyleSheet("color: gray; font-size: 11px;")
        self.bubble_batch_info.setWordWrap(True)
        img_layout.addWidget(self.bubble_batch_info)

        btn_load_reference = QPushButton("加载背景参考图（可选）...")
        btn_load_reference.clicked.connect(self._load_reference_images)
        img_layout.addWidget(btn_load_reference)

        self.bubble_status_label = QLabel("未加载气泡图像")
        self.bubble_status_label.setStyleSheet("color: gray;")
        img_layout.addWidget(self.bubble_status_label)

        image_group.setLayout(img_layout)
        left_layout.addWidget(image_group)

        # 重建参数
        recon_group = QGroupBox("层析重建参数")
        grid = QGridLayout()

        # --- 算法选择 ---
        grid.addWidget(QLabel("重建算法:"), 0, 0)
        self.recon_algo_combo = QComboBox()
        self.recon_algo_combo.addItems(["MART", "SMART", "Conv-SMART"])
        self.recon_algo_combo.setCurrentIndex(0)
        self.recon_algo_combo.setToolTip(
            "MART: 逐光线乘法重建(经典)\n"
            "SMART: 同步乘法重建(更稳定)\n"
            "Conv-SMART: 卷积加速SMART(最快)"
        )
        self.recon_algo_combo.currentIndexChanged.connect(
            self._on_recon_algo_changed
        )
        grid.addWidget(self.recon_algo_combo, 0, 1)

        grid.addWidget(QLabel("网格X:"), 1, 0)
        self.grid_x = QSpinBox()
        self.grid_x.setRange(16, 256)
        self.grid_x.setValue(64)
        grid.addWidget(self.grid_x, 1, 1)

        grid.addWidget(QLabel("网格Y:"), 2, 0)
        self.grid_y = QSpinBox()
        self.grid_y.setRange(16, 256)
        self.grid_y.setValue(64)
        grid.addWidget(self.grid_y, 2, 1)

        grid.addWidget(QLabel("网格Z:"), 3, 0)
        self.grid_z = QSpinBox()
        self.grid_z.setRange(16, 256)
        self.grid_z.setValue(64)
        grid.addWidget(self.grid_z, 3, 1)

        grid.addWidget(QLabel("域尺寸X (mm):"), 4, 0)
        self.domain_x = QDoubleSpinBox()
        self.domain_x.setRange(1, 200)
        self.domain_x.setValue(20.0)
        grid.addWidget(self.domain_x, 4, 1)

        grid.addWidget(QLabel("域尺寸Y (mm):"), 5, 0)
        self.domain_y = QDoubleSpinBox()
        self.domain_y.setRange(1, 200)
        self.domain_y.setValue(20.0)
        grid.addWidget(self.domain_y, 5, 1)

        grid.addWidget(QLabel("域尺寸Z (mm):"), 6, 0)
        self.domain_z = QDoubleSpinBox()
        self.domain_z.setRange(1, 200)
        self.domain_z.setValue(20.0)
        grid.addWidget(self.domain_z, 6, 1)

        grid.addWidget(QLabel("松弛因子:"), 7, 0)
        self.relax_spin = QDoubleSpinBox()
        self.relax_spin.setRange(0.01, 1.0)
        self.relax_spin.setValue(0.5)
        self.relax_spin.setSingleStep(0.05)
        grid.addWidget(self.relax_spin, 7, 1)

        grid.addWidget(QLabel("最大迭代次数:"), 8, 0)
        self.iter_spin = QSpinBox()
        self.iter_spin.setRange(1, 200)
        self.iter_spin.setValue(50)
        grid.addWidget(self.iter_spin, 8, 1)

        grid.addWidget(QLabel("体素阈值:"), 9, 0)
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.01, 0.5)
        self.threshold_spin.setValue(0.1)
        self.threshold_spin.setSingleStep(0.01)
        grid.addWidget(self.threshold_spin, 9, 1)

        # --- Conv-SMART 专属参数 ---
        self.convsmart_group = QGroupBox("Conv-SMART 参数")
        conv_grid = QGridLayout()

        conv_grid.addWidget(QLabel("卷积核尺寸:"), 0, 0)
        self.conv_kernel_spin = QSpinBox()
        self.conv_kernel_spin.setRange(3, 15)
        self.conv_kernel_spin.setValue(5)
        self.conv_kernel_spin.setSingleStep(2)
        self.conv_kernel_spin.setToolTip("PSF卷积核边长（奇数），越大越平滑")
        conv_grid.addWidget(self.conv_kernel_spin, 0, 1)

        conv_grid.addWidget(QLabel("PSF类型:"), 1, 0)
        self.psf_type_combo = QComboBox()
        self.psf_type_combo.addItems(["gaussian", "tophat", "empirical"])
        self.psf_type_combo.setToolTip(
            "gaussian: 高斯PSF（通用）\n"
            "tophat: 均匀圆盘PSF\n"
            "empirical: 经验PSF（需标定数据）"
        )
        conv_grid.addWidget(self.psf_type_combo, 1, 1)

        conv_grid.addWidget(QLabel("PSF sigma (体素):"), 2, 0)
        self.psf_sigma_spin = QDoubleSpinBox()
        self.psf_sigma_spin.setRange(0.3, 5.0)
        self.psf_sigma_spin.setValue(1.0)
        self.psf_sigma_spin.setSingleStep(0.1)
        self.psf_sigma_spin.setToolTip("高斯PSF的标准差（体素单位）")
        conv_grid.addWidget(self.psf_sigma_spin, 2, 1)

        conv_grid.addWidget(QLabel("FFT卷积:"), 3, 0)
        self.fft_conv_check = QCheckBox("启用FFT加速")
        self.fft_conv_check.setChecked(True)
        self.fft_conv_check.setToolTip("启用FFT卷积可大幅提升速度")
        conv_grid.addWidget(self.fft_conv_check, 3, 1)

        self.convsmart_group.setLayout(conv_grid)
        self.convsmart_group.setVisible(False)  # 默认隐藏
        grid.addWidget(self.convsmart_group, 10, 0, 1, 2)

        recon_group.setLayout(grid)
        left_layout.addWidget(recon_group)

        # 预处理参数
        preprocess_group = QGroupBox("图像预处理")
        pre_grid = QGridLayout()

        pre_grid.addWidget(QLabel("背景去除:"), 0, 0)
        self.bg_method_combo = QComboBox()
        self.bg_method_combo.addItems(["reference", "median", "mog"])
        pre_grid.addWidget(self.bg_method_combo, 0, 1)

        pre_grid.addWidget(QLabel("分割方法:"), 1, 0)
        self.threshold_method_combo = QComboBox()
        self.threshold_method_combo.addItems(["otsu", "adaptive", "li", "manual"])
        pre_grid.addWidget(self.threshold_method_combo, 1, 1)

        pre_grid.addWidget(QLabel("投影类型:"), 2, 0)
        self.proj_type_combo = QComboBox()
        self.proj_type_combo.addItems(["soft_edge", "silhouette", "distance"])
        pre_grid.addWidget(self.proj_type_combo, 2, 1)

        preprocess_group.setLayout(pre_grid)
        left_layout.addWidget(preprocess_group)

        # 操作按钮
        btn_layout = QHBoxLayout()

        btn_reconstruct = QPushButton("开始重建（当前时间点）")
        btn_reconstruct.setStyleSheet(
            "QPushButton { background-color: #2196F3; color: white; "
            "font-size: 13px; padding: 8px; border-radius: 5px; }"
            "QPushButton:hover { background-color: #1976D2; }"
        )
        btn_reconstruct.clicked.connect(self._run_reconstruction_single)
        btn_layout.addWidget(btn_reconstruct)

        btn_reconstruct_all = QPushButton("批量重建所有时间点")
        btn_reconstruct_all.setStyleSheet(
            "QPushButton { background-color: #1565C0; color: white; "
            "font-size: 13px; padding: 8px; border-radius: 5px; }"
            "QPushButton:hover { background-color: #0D47A1; }"
        )
        btn_reconstruct_all.clicked.connect(self._run_reconstruction_batch)
        btn_layout.addWidget(btn_reconstruct_all)

        left_layout.addLayout(btn_layout)

        left_layout.addStretch()

        # 右侧: 日志 + 可视化 + 时间点选择
        right_container = QWidget()
        right_container_layout = QVBoxLayout(right_container)
        right_container_layout.setContentsMargins(
            self._sp(6), self._sp(6), self._sp(6), self._sp(6)
        )

        # Log and preview area
        mid_splitter = QSplitter(Qt.Vertical)

        # 日志
        self.recon_log = QTextEdit()
        self.recon_log.setReadOnly(True)
        self.recon_log.setFontFamily("Consolas")
        self.recon_log.setPlaceholderText("重建日志")
        mid_splitter.addWidget(self.recon_log)

        # Batch/result preview
        self.proj_preview_label = QLabel()
        self.proj_preview_label.setAlignment(Qt.AlignCenter)
        self.proj_preview_label.setMinimumSize(self._sp(400), self._sp(300))
        self.proj_preview_label.setStyleSheet(
            "border: 1px solid #ccc; background-color: #f8f8f8; padding: 10px;"
        )
        self.proj_preview_label.setText("投影预览")
        mid_splitter.addWidget(self.proj_preview_label)

        mid_splitter.setSizes([self._sp(250), self._sp(400)])
        right_container_layout.addWidget(mid_splitter, stretch=1)

        # Visualization toolbar
        viz_toolbar = self._make_viz_toolbar("reconstruction")
        right_container_layout.addWidget(viz_toolbar)

        # === 时间点选择 ===
        timepoint_group = QGroupBox("时间点选择")
        tp_layout = QVBoxLayout()

        self.bubble_tp_info_label = QLabel("未加载时间序列")
        self.bubble_tp_info_label.setStyleSheet("color: gray;")
        tp_layout.addWidget(self.bubble_tp_info_label)

        tp_select_layout = QHBoxLayout()

        self.bubble_tp_slider = QSlider(Qt.Horizontal)
        self.bubble_tp_slider.setEnabled(False)
        self.bubble_tp_slider.valueChanged.connect(self._on_bubble_tp_changed)
        tp_select_layout.addWidget(self.bubble_tp_slider, stretch=3)

        self.bubble_tp_combo = QComboBox()
        self.bubble_tp_combo.setEnabled(False)
        self.bubble_tp_combo.setMinimumWidth(self._sp(180))
        self.bubble_tp_combo.currentIndexChanged.connect(
            self._on_bubble_tp_combo_changed
        )
        tp_select_layout.addWidget(self.bubble_tp_combo, stretch=1)

        tp_layout.addLayout(tp_select_layout)

        tp_nav_layout = QHBoxLayout()

        self.bubble_tp_prev_btn = QPushButton("上一时刻")
        self.bubble_tp_prev_btn.setEnabled(False)
        self.bubble_tp_prev_btn.clicked.connect(self._bubble_tp_prev)
        tp_nav_layout.addWidget(self.bubble_tp_prev_btn)

        self.bubble_tp_label = QLabel("-- / --")
        self.bubble_tp_label.setAlignment(Qt.AlignCenter)
        self.bubble_tp_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        tp_nav_layout.addWidget(self.bubble_tp_label)

        self.bubble_tp_next_btn = QPushButton("下一时刻")
        self.bubble_tp_next_btn.setEnabled(False)
        self.bubble_tp_next_btn.clicked.connect(self._bubble_tp_next)
        tp_nav_layout.addWidget(self.bubble_tp_next_btn)

        tp_layout.addLayout(tp_nav_layout)

        self.bubble_batch_progress = QProgressBar()
        self.bubble_batch_progress.setVisible(False)
        tp_layout.addWidget(self.bubble_batch_progress)

        timepoint_group.setLayout(tp_layout)
        right_container_layout.addWidget(timepoint_group)

        layout.addWidget(left_panel, stretch=1)
        layout.addWidget(right_container, stretch=2)

        self.content_stack.addWidget(page)

    def _create_particle_page(self):
        """创建粒子追踪/PIV 页面：左侧参数 + 右侧日志/可视化 + 时间点选择。"""
        page = QWidget()
        layout = QHBoxLayout(page)

        # 左侧参数
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # 粒子图像加载（批量模式）
        img_group = QGroupBox("示踪粒子图像（按相机批量加载）")
        img_layout = QVBoxLayout()

        cam_count_layout = QGridLayout()
        cam_count_layout.addWidget(QLabel("相机数目:"), 0, 0)
        self.particle_camera_count_spin = QSpinBox()
        self.particle_camera_count_spin.setRange(1, 16)
        self.particle_camera_count_spin.setValue(2)
        self.particle_camera_count_spin.valueChanged.connect(
            self._on_particle_camera_count_changed
        )
        cam_count_layout.addWidget(self.particle_camera_count_spin, 0, 1)
        img_layout.addLayout(cam_count_layout)

        btn_load_particles_batch = QPushButton("加载各相机粒子图像序列...")
        btn_load_particles_batch.setStyleSheet(
            "QPushButton { font-weight: bold; padding: 6px; }"
        )
        btn_load_particles_batch.clicked.connect(self._load_particle_images_batch)
        img_layout.addWidget(btn_load_particles_batch)

        self.particle_batch_info = QLabel(
            "每个相机选择一组按时间排序的粒子图像。\n"
            "系统会按时间顺序对齐各相机序列，并可分别指定第1帧和第2帧。"
        )
        self.particle_batch_info.setStyleSheet("color: gray; font-size: 11px;")
        self.particle_batch_info.setWordWrap(True)
        img_layout.addWidget(self.particle_batch_info)

        self.particle_camera_loader_container = QWidget()
        self.particle_camera_loader_layout = QVBoxLayout(
            self.particle_camera_loader_container
        )
        self.particle_camera_loader_layout.setContentsMargins(0, 0, 0, 0)
        self.particle_camera_loader_layout.setSpacing(self._sp(6))
        self.particle_camera_loader_toolbox = QToolBox()
        self.particle_camera_loader_toolbox.setStyleSheet(
            "QToolBox::tab { padding: 6px 10px; font-weight: bold; }"
        )
        self.particle_camera_loader_layout.addWidget(
            self.particle_camera_loader_toolbox
        )
        img_layout.addWidget(self.particle_camera_loader_container)

        frame_select_group = QGroupBox("第1帧 / 第2帧设置")
        frame_select_layout = QGridLayout(frame_select_group)
        frame_select_layout.addWidget(QLabel("第1帧:"), 0, 0)
        self.piv_frame1_combo = QComboBox()
        self.piv_frame1_combo.setEnabled(False)
        self.piv_frame1_combo.currentIndexChanged.connect(
            lambda idx: self._on_particle_frame_combo_changed(1, idx)
        )
        frame_select_layout.addWidget(self.piv_frame1_combo, 0, 1)
        frame_select_layout.addWidget(QLabel("第2帧:"), 1, 0)
        self.piv_frame2_combo = QComboBox()
        self.piv_frame2_combo.setEnabled(False)
        self.piv_frame2_combo.currentIndexChanged.connect(
            lambda idx: self._on_particle_frame_combo_changed(2, idx)
        )
        frame_select_layout.addWidget(self.piv_frame2_combo, 1, 1)
        self.piv_frame_select_info = QLabel("未加载时间序列")
        self.piv_frame_select_info.setStyleSheet("color: #666;")
        self.piv_frame_select_info.setWordWrap(True)
        frame_select_layout.addWidget(self.piv_frame_select_info, 2, 0, 1, 2)
        img_layout.addWidget(frame_select_group)

        # 单帧手动加载
        single_frame_layout = QHBoxLayout()
        btn_load_frame1 = QPushButton("手动加载第1帧...")
        btn_load_frame1.clicked.connect(self._load_particle_frame1)
        single_frame_layout.addWidget(btn_load_frame1)

        btn_load_frame2 = QPushButton("手动加载第2帧...")
        btn_load_frame2.clicked.connect(self._load_particle_frame2)
        single_frame_layout.addWidget(btn_load_frame2)
        img_layout.addLayout(single_frame_layout)

        self.particle_status = QLabel("尚未加载粒子图像")
        self.particle_status.setStyleSheet("color: gray;")
        img_layout.addWidget(self.particle_status)

        img_group.setLayout(img_layout)
        left_layout.addWidget(img_group)

        # 粒子检测参数
        det_group = QGroupBox("粒子检测参数")
        det_grid = QGridLayout()
        det_grid.addWidget(QLabel("最小面积 (px²):"), 0, 0)
        self.p_min_area = QDoubleSpinBox()
        self.p_min_area.setRange(0.5, 100)
        self.p_min_area.setValue(2.0)
        det_grid.addWidget(self.p_min_area, 0, 1)

        det_grid.addWidget(QLabel("最大面积 (px²):"), 1, 0)
        self.p_max_area = QDoubleSpinBox()
        self.p_max_area.setRange(10, 2000)
        self.p_max_area.setValue(200.0)
        det_grid.addWidget(self.p_max_area, 1, 1)

        det_grid.addWidget(QLabel("圆度阈值:"), 2, 0)
        self.p_circularity = QDoubleSpinBox()
        self.p_circularity.setRange(0.1, 1.0)
        self.p_circularity.setValue(0.5)
        self.p_circularity.setSingleStep(0.05)
        det_grid.addWidget(self.p_circularity, 2, 1)

        det_grid.addWidget(QLabel("极线阈值 (px):"), 3, 0)
        self.p_epipolar = QDoubleSpinBox()
        self.p_epipolar.setRange(0.5, 20)
        self.p_epipolar.setValue(3.0)
        det_grid.addWidget(self.p_epipolar, 3, 1)

        det_group.setLayout(det_grid)
        left_layout.addWidget(det_group)

        # 速度场参数
        vel_group = QGroupBox("速度场参数")
        vel_grid = QGridLayout()
        vel_grid.addWidget(QLabel("dt (s):"), 0, 0)
        self.piv_dt = QDoubleSpinBox()
        self.piv_dt.setRange(0.0001, 10)
        self.piv_dt.setValue(0.001)
        self.piv_dt.setDecimals(4)
        self.piv_dt.setSingleStep(0.0001)
        vel_grid.addWidget(self.piv_dt, 0, 1)

        vel_grid.addWidget(QLabel("查询窗口尺寸 (mm):"), 1, 0)
        self.piv_interrog = QDoubleSpinBox()
        self.piv_interrog.setRange(0.5, 20)
        self.piv_interrog.setValue(2.0)
        self.piv_interrog.setSingleStep(0.5)
        vel_grid.addWidget(self.piv_interrog, 1, 1)

        vel_grid.addWidget(QLabel("重叠率:"), 2, 0)
        self.piv_overlap = QDoubleSpinBox()
        self.piv_overlap.setRange(0.0, 0.75)
        self.piv_overlap.setValue(0.5)
        self.piv_overlap.setSingleStep(0.1)
        vel_grid.addWidget(self.piv_overlap, 2, 1)

        vel_grid.addWidget(QLabel("SNR 阈值:"), 3, 0)
        self.piv_snr = QDoubleSpinBox()
        self.piv_snr.setRange(0.5, 5.0)
        self.piv_snr.setValue(1.2)
        self.piv_snr.setSingleStep(0.1)
        vel_grid.addWidget(self.piv_snr, 3, 1)

        vel_group.setLayout(vel_grid)
        left_layout.addWidget(vel_group)

        # 操作按钮
        piv_btn_layout = QHBoxLayout()

        btn_reconstruct_p = QPushButton("粒子3D重建")
        btn_reconstruct_p.setStyleSheet(
            "QPushButton { background-color: #FF9800; color: white; "
            "font-size: 13px; padding: 8px; border-radius: 5px; }"
            "QPushButton:hover { background-color: #F57C00; }"
        )
        btn_reconstruct_p.clicked.connect(self._run_particle_reconstruction)
        piv_btn_layout.addWidget(btn_reconstruct_p)

        btn_piv_all = QPushButton("批量计算速度场")
        btn_piv_all.setStyleSheet(
            "QPushButton { background-color: #E65100; color: white; "
            "font-size: 13px; padding: 8px; border-radius: 5px; }"
            "QPushButton:hover { background-color: #BF360C; }"
        )
        btn_piv_all.clicked.connect(self._run_piv_batch)
        piv_btn_layout.addWidget(btn_piv_all)

        left_layout.addLayout(piv_btn_layout)

        btn_velocity = QPushButton("计算当前双帧速度场")
        btn_velocity.setStyleSheet(
            "QPushButton { background-color: #9C27B0; color: white; "
            "font-size: 13px; padding: 8px; border-radius: 5px; }"
            "QPushButton:hover { background-color: #7B1FA2; }"
        )
        btn_velocity.clicked.connect(self._run_velocity_computation)
        left_layout.addWidget(btn_velocity)

        left_layout.addStretch()

        # 右侧: 日志 + 化?+ 时间点择
        right_container = QWidget()
        right_container_layout = QVBoxLayout(right_container)
        right_container_layout.setContentsMargins(
            self._sp(6), self._sp(6), self._sp(6), self._sp(6)
        )

        # Log and preview area
        mid_splitter = QSplitter(Qt.Vertical)

        self.piv_log = QTextEdit()
        self.piv_log.setReadOnly(True)
        self.piv_log.setFontFamily("Consolas")
        self.piv_log.setPlaceholderText("PIV日志")
        mid_splitter.addWidget(self.piv_log)

        self.particle_camera_preview_scroll = QScrollArea()
        self.particle_camera_preview_scroll.setWidgetResizable(True)
        self.particle_camera_preview_content = QWidget()
        self.particle_camera_preview_layout = QGridLayout(
            self.particle_camera_preview_content
        )
        self.particle_camera_preview_layout.setContentsMargins(
            self._sp(6), self._sp(6), self._sp(6), self._sp(6)
        )
        self.particle_camera_preview_layout.setSpacing(self._sp(8))
        self.particle_camera_preview_scroll.setWidget(
            self.particle_camera_preview_content
        )
        mid_splitter.addWidget(self.particle_camera_preview_scroll)

        self.piv_preview = QLabel()
        self.piv_preview.setAlignment(Qt.AlignCenter)
        self.piv_preview.setMinimumSize(self._sp(400), self._sp(300))
        self.piv_preview.setStyleSheet(
            "border: 1px solid #ccc; background-color: #f8f8f8; padding: 10px;"
        )
        self.piv_preview.setText("粒子追踪与速度场结果")
        mid_splitter.addWidget(self.piv_preview)

        mid_splitter.setSizes([self._sp(140), self._sp(260), self._sp(360)])
        right_container_layout.addWidget(mid_splitter, stretch=1)

        # Visualization toolbar
        viz_toolbar = self._make_viz_toolbar("particle")
        right_container_layout.addWidget(viz_toolbar)

        # === 时间点选择 ===
        piv_timepoint_group = QGroupBox("时间点选择")
        piv_tp_layout = QVBoxLayout()

        self.piv_tp_info_label = QLabel("未加载时间序列")
        self.piv_tp_info_label.setStyleSheet("color: gray;")
        piv_tp_layout.addWidget(self.piv_tp_info_label)

        piv_tp_select_layout = QHBoxLayout()

        self.piv_tp_slider = QSlider(Qt.Horizontal)
        self.piv_tp_slider.setEnabled(False)
        self.piv_tp_slider.valueChanged.connect(self._on_piv_tp_changed)
        piv_tp_select_layout.addWidget(self.piv_tp_slider, stretch=3)

        self.piv_tp_combo = QComboBox()
        self.piv_tp_combo.setEnabled(False)
        self.piv_tp_combo.setMinimumWidth(self._sp(180))
        self.piv_tp_combo.currentIndexChanged.connect(
            self._on_piv_tp_combo_changed
        )
        piv_tp_select_layout.addWidget(self.piv_tp_combo, stretch=1)

        piv_tp_layout.addLayout(piv_tp_select_layout)

        piv_tp_nav_layout = QHBoxLayout()

        self.piv_tp_prev_btn = QPushButton("上一时刻")
        self.piv_tp_prev_btn.setEnabled(False)
        self.piv_tp_prev_btn.clicked.connect(self._piv_tp_prev)
        piv_tp_nav_layout.addWidget(self.piv_tp_prev_btn)

        self.piv_tp_label = QLabel("-- / --")
        self.piv_tp_label.setAlignment(Qt.AlignCenter)
        self.piv_tp_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        piv_tp_nav_layout.addWidget(self.piv_tp_label)

        self.piv_tp_next_btn = QPushButton("下一时刻")
        self.piv_tp_next_btn.setEnabled(False)
        self.piv_tp_next_btn.clicked.connect(self._piv_tp_next)
        piv_tp_nav_layout.addWidget(self.piv_tp_next_btn)

        piv_tp_layout.addLayout(piv_tp_nav_layout)

        self.piv_pair_label = QLabel("帧对: -- -> --")
        self.piv_pair_label.setAlignment(Qt.AlignCenter)
        self.piv_pair_label.setStyleSheet("color: #666; font-size: 12px;")
        piv_tp_layout.addWidget(self.piv_pair_label)

        self.piv_batch_progress = QProgressBar()
        self.piv_batch_progress.setVisible(False)
        piv_tp_layout.addWidget(self.piv_batch_progress)

        piv_timepoint_group.setLayout(piv_tp_layout)
        right_container_layout.addWidget(piv_timepoint_group)

        layout.addWidget(left_panel, stretch=1)
        layout.addWidget(right_container, stretch=2)

        self.content_stack.addWidget(page)

        # 粒子数据缓存
        self.particle_images_frame1: Dict[str, np.ndarray] = {}
        self.particle_images_frame2: Dict[str, np.ndarray] = {}
        self.particles_3d_frame1 = []
        self.particles_3d_frame2 = []
        self._velocity_result = None
        self._particle_sequence_info_labels: Dict[str, QLabel] = {}
        self._particle_frame_preview_labels: Dict[int, Dict[str, QLabel]] = {
            1: {},
            2: {},
        }
        self._on_particle_camera_count_changed(
            self.particle_camera_count_spin.value()
        )

    def _get_particle_camera_ids(self) -> List[str]:
        count = self.particle_camera_count_spin.value()
        if self.calibration_results:
            calib_ids = list(self.calibration_results.keys())
            if count <= len(calib_ids):
                return calib_ids[:count]
        return [f"cam{i + 1}" for i in range(count)]

    def _clear_layout_widgets(self, layout):
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            child_layout = item.layout()
            if widget is not None:
                widget.deleteLater()
            elif child_layout is not None:
                self._clear_layout_widgets(child_layout)

    def _on_particle_camera_count_changed(self, _value: int):
        self.particle_active_camera_ids = self._get_particle_camera_ids()
        self.particle_sequence_paths = {
            cam_id: self.particle_sequence_paths.get(cam_id, [])
            for cam_id in self.particle_active_camera_ids
        }
        self._build_particle_camera_loader_widgets()
        self._build_particle_camera_preview_widgets()
        self._rebuild_particle_timepoints_from_sequences(update_log=False)

    def _build_particle_camera_loader_widgets(self):
        while self.particle_camera_loader_toolbox.count():
            widget = self.particle_camera_loader_toolbox.widget(0)
            self.particle_camera_loader_toolbox.removeItem(0)
            if widget is not None:
                widget.deleteLater()
        self._particle_sequence_info_labels = {}

        for cam_id in self.particle_active_camera_ids:
            panel = QWidget()
            group_layout = QGridLayout(panel)

            btn = QPushButton("批量加载该相机序列...")
            btn.clicked.connect(
                lambda _checked=False, cid=cam_id: self._load_particle_sequence_for_camera(cid)
            )
            group_layout.addWidget(btn, 0, 0)

            info = QLabel("未加载")
            info.setWordWrap(True)
            info.setStyleSheet("color: gray;")
            group_layout.addWidget(info, 0, 1)
            self._particle_sequence_info_labels[cam_id] = info

            hint = QLabel("展开当前相机后选择按时间顺序排列的图像序列。")
            hint.setWordWrap(True)
            hint.setStyleSheet("color: #666; font-size: 11px;")
            group_layout.addWidget(hint, 1, 0, 1, 2)

            self.particle_camera_loader_toolbox.addItem(panel, f"{cam_id} 图像序列")

        if self.particle_camera_loader_toolbox.count() > 0:
            self.particle_camera_loader_toolbox.setCurrentIndex(0)
        self._refresh_particle_sequence_info_labels()

    def _build_particle_camera_preview_widgets(self):
        self._clear_layout_widgets(self.particle_camera_preview_layout)
        self._particle_frame_preview_labels = {1: {}, 2: {}}

        if not self.particle_active_camera_ids:
            empty_label = QLabel("请先设置相机数量并加载图像序列")
            empty_label.setAlignment(Qt.AlignCenter)
            empty_label.setStyleSheet("color: gray;")
            self.particle_camera_preview_layout.addWidget(empty_label, 0, 0)
            return

        for col, cam_id in enumerate(self.particle_active_camera_ids):
            title = QLabel(cam_id)
            title.setAlignment(Qt.AlignCenter)
            title.setStyleSheet("font-weight: bold; color: #E65100;")
            self.particle_camera_preview_layout.addWidget(title, 0, col)

            f1_label = QLabel("第1帧")
            f1_label.setAlignment(Qt.AlignCenter)
            f1_label.setMinimumSize(self._sp(180), self._sp(130))
            f1_label.setStyleSheet("border: 1px solid #ccc; background: #f8f8f8;")
            self.particle_camera_preview_layout.addWidget(f1_label, 1, col)
            self._particle_frame_preview_labels[1][cam_id] = f1_label

            f2_label = QLabel("第2帧")
            f2_label.setAlignment(Qt.AlignCenter)
            f2_label.setMinimumSize(self._sp(180), self._sp(130))
            f2_label.setStyleSheet("border: 1px solid #ccc; background: #f8f8f8;")
            self.particle_camera_preview_layout.addWidget(f2_label, 2, col)
            self._particle_frame_preview_labels[2][cam_id] = f2_label

    def _refresh_particle_sequence_info_labels(self):
        for cam_id, label in self._particle_sequence_info_labels.items():
            paths = self.particle_sequence_paths.get(cam_id, [])
            if paths:
                label.setText(f"已加载 {len(paths)} 张\n首张: {os.path.basename(paths[0])}")
                label.setStyleSheet("color: green;")
            else:
                label.setText("未加载")
                label.setStyleSheet("color: gray;")

    def _load_particle_sequence_for_camera(self, cam_id: str):
        import cv2

        files, _ = QFileDialog.getOpenFileNames(
            self,
            f"选择 {cam_id} 的粒子图像序列",
            "",
            "图像文件 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;所有文件 (*)"
        )
        if not files:
            return

        valid_files = []
        for path in files:
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is not None:
                valid_files.append(path)

        self.particle_sequence_paths[cam_id] = valid_files
        self._refresh_particle_sequence_info_labels()
        self._rebuild_particle_timepoints_from_sequences()

    def _rebuild_particle_timepoints_from_sequences(self, update_log: bool = True):
        import cv2

        active = self.particle_active_camera_ids
        self.particle_timepoint_images.clear()
        self.particle_timepoint_names.clear()
        self.piv_batch_results.clear()

        if not active or any(not self.particle_sequence_paths.get(cam_id) for cam_id in active):
            self._setup_piv_frame_selectors()
            self._refresh_particle_camera_previews()
            return

        seq_lengths = [len(self.particle_sequence_paths[cam_id]) for cam_id in active]
        common_len = min(seq_lengths) if seq_lengths else 0
        if common_len < 1:
            self._setup_piv_frame_selectors()
            self._refresh_particle_camera_previews()
            return

        for idx in range(common_len):
            cam_images = {}
            for cam_id in active:
                path = self.particle_sequence_paths[cam_id][idx]
                img = cv2.imread(path, cv2.IMREAD_COLOR)
                if img is not None:
                    cam_images[cam_id] = img
            if len(cam_images) == len(active):
                self.particle_timepoint_images[idx] = cam_images
                self.particle_timepoint_names[idx] = f"t{idx:03d}"

        if self.particle_timepoint_images:
            self._setup_piv_timepoint_selector()
            self._setup_piv_frame_selectors()
            default_f1 = 0
            default_f2 = min(1, len(self.particle_timepoint_images) - 1)
            self._set_particle_frame_by_index(1, default_f1)
            self._set_particle_frame_by_index(2, default_f2)
            self.particle_status.setText(
                f"已对齐 {len(self.particle_timepoint_images)} 个时间点，"
                f"{len(active)} 个相机"
            )
            self.particle_status.setStyleSheet("color: green; font-weight: bold;")
            if update_log:
                self.piv_log.append(
                    f"=== 粒子序列对齐完成 ===\n"
                    f"相机数量: {len(active)}\n"
                    f"时间点数: {len(self.particle_timepoint_images)}\n"
                    f"序列长度: {seq_lengths}\n"
                )
        else:
            self._setup_piv_frame_selectors()
            self._refresh_particle_camera_previews()

    def _setup_piv_frame_selectors(self):
        keys = sorted(self.particle_timepoint_images.keys())
        self.piv_frame1_combo.blockSignals(True)
        self.piv_frame2_combo.blockSignals(True)
        self.piv_frame1_combo.clear()
        self.piv_frame2_combo.clear()

        for idx in keys:
            name = self.particle_timepoint_names.get(idx, f"t{idx}")
            self.piv_frame1_combo.addItem(f"{name} (t{idx})", idx)
            self.piv_frame2_combo.addItem(f"{name} (t{idx})", idx)

        enabled = bool(keys)
        self.piv_frame1_combo.setEnabled(enabled)
        self.piv_frame2_combo.setEnabled(enabled)
        self.piv_frame1_combo.blockSignals(False)
        self.piv_frame2_combo.blockSignals(False)

        if not enabled:
            self.piv_frame_select_info.setText("未设置双帧")
        else:
            self._update_particle_frame_selection_info()

    def _set_particle_frame_by_index(self, frame_num: int, combo_idx: int):
        combo = self.piv_frame1_combo if frame_num == 1 else self.piv_frame2_combo
        if combo_idx < 0 or combo_idx >= combo.count():
            return
        combo.blockSignals(True)
        combo.setCurrentIndex(combo_idx)
        combo.blockSignals(False)
        self._on_particle_frame_combo_changed(frame_num, combo_idx)

    def _on_particle_frame_combo_changed(self, frame_num: int, combo_idx: int):
        combo = self.piv_frame1_combo if frame_num == 1 else self.piv_frame2_combo
        time_idx = combo.itemData(combo_idx)
        if time_idx is None or time_idx not in self.particle_timepoint_images:
            return
        if frame_num == 1:
            self.particle_images_frame1 = self.particle_timepoint_images[time_idx]
        else:
            self.particle_images_frame2 = self.particle_timepoint_images[time_idx]
        self._update_particle_frame_selection_info()
        self._update_piv_pair_label()
        self._refresh_particle_camera_previews()

    def _update_particle_frame_selection_info(self):
        def _name(combo: QComboBox) -> str:
            idx = combo.currentData()
            if idx is None:
                return "--"
            return self.particle_timepoint_names.get(idx, f"t{idx}")

        self.piv_frame_select_info.setText(
            f"当前双帧: 第1帧 = {_name(self.piv_frame1_combo)}\n"
            f"第2帧 = {_name(self.piv_frame2_combo)}"
        )

    def _refresh_particle_camera_previews(self):
        for frame_num, frame_dict in [
            (1, self.particle_images_frame1),
            (2, self.particle_images_frame2),
        ]:
            for cam_id, label in self._particle_frame_preview_labels.get(frame_num, {}).items():
                img = frame_dict.get(cam_id)
                if img is None:
                    label.setText(f"第{frame_num}帧\n{cam_id}\n未加载")
                    label.setPixmap(QPixmap())
                    continue
                pixmap = self._ie_ndarray_to_pixmap(img)
                if pixmap:
                    label.setPixmap(
                        pixmap.scaled(
                            label.size(),
                            Qt.KeepAspectRatio,
                            Qt.SmoothTransformation
                        )
                    )
                else:
                    label.setText(f"第{frame_num}帧\n{cam_id}\n预览失败")

    # ---- 标定模式切换 ----

    def _on_calib_mode_changed(self, idx):
        """标定模式切换"""
        if idx == 0:
            # 多相机联合标定
            self.multi_calib_container.setVisible(True)
            self.single_calib_group.setVisible(False)
            self.stereo_calib_group.setVisible(False)
            self.calib_mode_desc.setText(
                "使用多个相机同时标定，获取各相机内参和相对位姿。\n"
                "适用于 Tomographic PIV 多相机系统。"
            )
        elif idx == 1:
            # 单相机标定
            self.multi_calib_container.setVisible(False)
            self.single_calib_group.setVisible(True)
            self.stereo_calib_group.setVisible(False)
            self.calib_mode_desc.setText(
                "使用单个相机拍摄棋盘格标定板图像序列，获取内参和畸变系数。\n"
                "结果可用于单相机射线追踪三维重建。"
            )
            self.calib_mode_desc.setText(
                "使用单个相机拍摄棋盘格或圆点标定板图像序列，获取内参和畸变系数。\n"
                "结果可用于单相机射线追踪三维重建。"
            )
        elif idx == 2:
            # 双目标定
            self.multi_calib_container.setVisible(False)
            self.single_calib_group.setVisible(False)
            self.stereo_calib_group.setVisible(True)
            self.calib_mode_desc.setText(
                "加载左右相机拍摄的棋盘格图像对，进行立体标定。\n"
                "获取双目之间的旋转(R)、平移(T)、本质矩阵(E)和基础矩阵(F)。"
            )

        self._refresh_calibration_preview()

    def _pattern_type_items(self) -> Dict[str, str]:
        return {
            "checkerboard (棋盘格)": "checkerboard",
            "circles (对称圆点阵)": "circles",
            "acircles (非对称圆点阵)": "acircles",
            "volume_dots (体标定板点阵)": "volume_dots",
        }

    def _pattern_type_key_to_text(self, pattern_type: str) -> Optional[str]:
        for text, value in self._pattern_type_items().items():
            if value == pattern_type:
                return text
        return None

    def _current_calibration_detection_config(self) -> dict:
        pattern_type = self._pattern_type_items().get(
            self.pattern_type_combo.currentText(), "checkerboard"
        )
        config = {
            "pattern_type": pattern_type,
            "pattern_size": (
                self.pattern_w_spin.value(),
                self.pattern_h_spin.value(),
            ),
            "square_size": self.square_size_spin.value(),
        }
        if hasattr(self, "level_separation_spin"):
            config["level_separation"] = self.level_separation_spin.value()
        return config

    def _infer_and_apply_pattern_spec(self, image_paths: List[str]):
        if not image_paths:
            return

        current_type = self._pattern_type_items().get(
            self.pattern_type_combo.currentText(), "checkerboard"
        )
        candidate_types = [current_type] + [
            pattern_type
            for pattern_type in MultiCameraCalibrator.SUPPORTED_PATTERN_TYPES
            if pattern_type != current_type
        ]
        result = MultiCameraCalibrator.infer_pattern_spec_from_paths(
            image_paths,
            candidate_types=candidate_types,
            size_min=3,
            size_max=20,
            max_images=2,
        )
        if result is None:
            return

        pattern_text = self._pattern_type_key_to_text(result["pattern_type"])
        if pattern_text is not None:
            combo_index = self.pattern_type_combo.findText(pattern_text)
            if combo_index >= 0:
                self.pattern_type_combo.setCurrentIndex(combo_index)

        pattern_w, pattern_h = result["pattern_size"]
        self.pattern_w_spin.setValue(int(pattern_w))
        self.pattern_h_spin.setValue(int(pattern_h))
        self.statusBar().showMessage(
            f"检测到标定板: {result['pattern_type']} {pattern_w}x{pattern_h}"
        )

    def _refresh_calibration_preview(self, selected_key: Optional[str] = None):
        mode_index = self.calib_mode_combo.currentIndex()
        if hasattr(self.calib_preview_label, "set_detection_config"):
            self.calib_preview_label.set_detection_config(
                self._current_calibration_detection_config()
            )

        if mode_index == 0:
            datasets = {
                cam_id: files
                for cam_id, files in self.camera_calib_images.items()
                if files
            }
            self.calib_preview_label.set_image_sets(datasets, selected_key=selected_key)
            return

        if mode_index == 1:
            datasets = {}
            if hasattr(self, "_single_calib_files") and self._single_calib_files:
                datasets["single_camera"] = list(self._single_calib_files)
            self.calib_preview_label.set_image_sets(
                datasets, selected_key="single_camera"
            )
            return

        if mode_index == 2:
            datasets = {}
            if hasattr(self, "_stereo_left_files") and self._stereo_left_files:
                datasets["left_camera"] = list(self._stereo_left_files)
            if hasattr(self, "_stereo_right_files") and self._stereo_right_files:
                datasets["right_camera"] = list(self._stereo_right_files)
            if selected_key is None:
                selected_key = "left_camera" if "left_camera" in datasets else "right_camera"
            self.calib_preview_label.set_image_sets(datasets, selected_key=selected_key)

    def _on_calib_ruler_distance_changed(self, distance_px: float):
        if hasattr(self, "single_calib_pixel_label"):
            if distance_px > 0:
                self.single_calib_pixel_label.setText(
                    f"标尺像素距离: {distance_px:.2f} px"
                )
            else:
                self.single_calib_pixel_label.setText("标尺像素距离: 未测量")
    def _load_single_calib_images(self):
        """Load one single-camera calibration image without blocking on analysis."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择单相机标定图像",
            "",
            "图像文件 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;所有文件 (*)"
        )
        if not file_path:
            return

        self._single_calib_files = [file_path]
        self.single_camera_params = None
        self.single_calib_label.setText(f"已加载 1 张图像: {os.path.basename(file_path)}")
        self.single_calib_label.setStyleSheet("color: green; font-weight: bold;")
        if hasattr(self, "single_calib_pixel_label"):
            self.single_calib_pixel_label.setText("标尺像素距离: 未测量")
        if hasattr(self.calib_preview_label, "clear_ruler"):
            self.calib_preview_label.clear_ruler()
            self.calib_preview_label.enable_ruler(True)
            self.single_calib_label.setText(f"单相机标定: 已加载 {len(self._single_calib_files)} 张图像，可在预览区域右键打开标尺")
        self._refresh_calibration_preview(selected_key="single_camera")

    # 注：_run_single_calibration_legacy 的完整实现在本文件后面（使用 MultiCameraCalibrator）

    def _run_single_calibration_legacy(self):
        """执行单相机标定，支持棋盘格和圆点标定板。"""
        if not hasattr(self, '_single_calib_files') or not self._single_calib_files:
            QMessageBox.warning(self, "警告", "请先加载标定图像")
            return
        if len(self._single_calib_files) < 3:
            QMessageBox.warning(self, "警告", "至少需要 3 张标定图像")
            return

        pattern_map = {
            "checkerboard (棋盘格)": "checkerboard",
            "circles (对称圆点阵)": "circles",
            "acircles (非对称圆点阵)": "acircles",
            "volume_dots (体标定板点阵)": "volume_dots",
        }
        pattern_type = pattern_map.get(
            self.pattern_type_combo.currentText(), "checkerboard"
        )
        pattern_size = (
            self.pattern_w_spin.value(), self.pattern_h_spin.value()
        )
        square_size = self.square_size_spin.value()
        level_separation = self.level_separation_spin.value()

        calibrator = MultiCameraCalibrator(
            pattern_type=pattern_type,
            pattern_size=pattern_size,
            square_size=square_size,
            level_separation=level_separation
        )

        try:
            params = calibrator.calibrate_camera(
                "single_camera",
                self._single_calib_files
            )
        except Exception as e:
            QMessageBox.warning(self, "警告", str(e))
            return

        calib_data = calibrator._calib_data.get("single_camera", {})
        rvecs = calib_data.get("rvecs", [])
        tvecs = calib_data.get("tvecs", [])
        n_images = len(calib_data.get("img_points", []))

        camera_matrix = np.array(params.camera_matrix, dtype=np.float64)
        dist_coeffs = np.array(params.dist_coeffs, dtype=np.float64).reshape(1, -1)
        image_w, image_h = params.image_size

        self.single_camera_params = {
            'pattern_type': pattern_type,
            'pattern_size': list(pattern_size),
            'square_size': float(square_size),
            'camera_matrix': params.camera_matrix,
            'dist_coeffs': params.dist_coeffs,
            'rms': float(params.rms_error),
            'image_size': [image_w, image_h],
            'rvecs': [r.flatten().tolist() for r in rvecs],
            'tvecs': [t.flatten().tolist() for t in tvecs],
            'n_images': n_images
        }

        report = (
            f"=== 单相机标定完?===\n\n"
            f"标定板类? {pattern_type}\n"
            f"标定板尺? {pattern_size[0]} x {pattern_size[1]}\n"
            f"有效图像数量: {n_images}\n"
            f"图像尺寸: {image_w} x {image_h}\n"
            f"重投影?RMS): {params.rms_error:.4f} px\n\n"
            f"--- 内参矩阵 ---\n"
            f"fx={camera_matrix[0,0]:.2f}  fy={camera_matrix[1,1]:.2f}\n"
            f"cx={camera_matrix[0,2]:.2f}  cy={camera_matrix[1,2]:.2f}\n\n"
            f"--- 畸变系数 ---\n"
            f"k1={dist_coeffs[0,0]:.6f}\n"
            f"k2={dist_coeffs[0,1] if dist_coeffs.shape[1] > 1 else 0.0:.6f}\n"
            f"p1={dist_coeffs[0,2] if dist_coeffs.shape[1] > 2 else 0.0:.6f}\n"
            f"p2={dist_coeffs[0,3] if dist_coeffs.shape[1] > 3 else 0.0:.6f}\n"
            f"k3={dist_coeffs[0,4] if dist_coeffs.shape[1] > 4 else 0.0:.6f}\n\n"
            "标定结果已保存，可用于单相机射线追踪三维重建。"
        )
        self.calib_result_text.setPlainText(report)
        self.statusBar().showMessage(
            f"单相机标定完? RMS={params.rms_error:.4f}px"
        )
        QMessageBox.information(
            self,
            "完成",
            f"单相机标定完?\nRMS请: {params.rms_error:.4f} px\n"
            "可用于单相机 3D 重建模块。"
        )

    def _run_single_calibration(self):
        """Finish single-camera scale calibration from one image and a ruler distance."""
        if not hasattr(self, '_single_calib_files') or not self._single_calib_files:
            QMessageBox.warning(self, "警告", "请先加载 1 张单相机标定图像")
            return

        image_path = self._single_calib_files[0]
        image = robust_imread(image_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            QMessageBox.warning(self, "警告", "无法读取已加载的标定图像")
            return

        distance_px = 0.0
        if hasattr(self.calib_preview_label, "ruler_distance_px"):
            distance_px = self.calib_preview_label.ruler_distance_px()
        if distance_px <= 0:
            QMessageBox.warning(
                self,
                "警告",
                "请先在预览区域右键打开标尺，并拖动鼠标获得像素距离",
            )
            return

        actual_size = self.single_actual_size_spin.value()
        if actual_size <= 0:
            QMessageBox.warning(self, "警告", "请输入大于 0 的实际尺寸")
            return

        image_h, image_w = image.shape[:2]
        mm_per_px = float(actual_size / distance_px)
        px_per_mm = float(distance_px / actual_size)

        # This single-image workflow provides image scale, not lens distortion calibration.
        camera_matrix = [
            [px_per_mm, 0.0, image_w / 2.0],
            [0.0, px_per_mm, image_h / 2.0],
            [0.0, 0.0, 1.0],
        ]
        dist_coeffs = [0.0, 0.0, 0.0, 0.0, 0.0]

        self.single_camera_params = {
            'calibration_type': 'single_image_scale',
            'image_path': image_path,
            'image_size': [image_w, image_h],
            'ruler_distance_px': distance_px,
            'actual_size_mm': float(actual_size),
            'mm_per_px': mm_per_px,
            'px_per_mm': px_per_mm,
            'camera_matrix': camera_matrix,
            'dist_coeffs': dist_coeffs,
            'rms': 0.0,
            'n_images': 1,
        }

        report = (
            "=== 单图像标尺标定结果 ===\n\n"
            "标定类型: 单图像像素标尺\n"
            f"图像: {os.path.basename(image_path)}\n"
            f"图像尺寸: {image_w} x {image_h}\n"
            f"标尺像素距离: {distance_px:.4f} px\n"
            f"实际尺寸: {actual_size:.4f} mm\n"
            f"比例: {mm_per_px:.8f} mm/px ({px_per_mm:.4f} px/mm)\n\n"
            "注: 单图像无法估计镜头畸变，畸变参数设为 0。\n"
            "后续重建将使用该像素尺度。"
        )
        self.calib_result_text.setPlainText(report)
        self.statusBar().showMessage(
            f"单相机标定完? {mm_per_px:.6f} mm/px"
        )
        QMessageBox.information(
            self,
            "完成",
            f"单相机标定完成\n尺度: {mm_per_px:.6f} mm/px",
        )

    def _load_stereo_images_left(self):
        """加载双目标定左相机图像。"""
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择左相机标定图像", "",
            "图像文件 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;所有文件 (*)"
        )
        if files:
            self._stereo_left_files = files
            self.stereo_left_label.setText(
                f"左相? 已加?{len(files)} 张图?"
            )
            self.stereo_left_label.setStyleSheet("color: green;")
            self.statusBar().showMessage(
                f"左相? 已加?{len(files)} 张标定图?"
            )

            self._refresh_calibration_preview(selected_key="left_camera")

    def _load_stereo_images_right(self):
        """加载双目标定右相机图像。"""
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择右相机标定图像", "",
            "图像文件 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;所有文件 (*)"
        )
        if files:
            self._stereo_right_files = files
            self.stereo_right_label.setText(
                f"右相? 已加?{len(files)} 张图?"
            )
            self.stereo_right_label.setStyleSheet("color: green;")
            self.statusBar().showMessage(
                f"右相? 已加?{len(files)} 张标定图?"
            )

            self._refresh_calibration_preview(selected_key="right_camera")

    def _run_stereo_calibration_legacy(self):
        """执行双目标定。"""
        import cv2

        if (not hasattr(self, '_stereo_left_files') or
                not hasattr(self, '_stereo_right_files')):
            QMessageBox.warning(self, "警告", "请先加载左右相机的标定图像")
            return

        if (len(self._stereo_left_files) < 3 or
                len(self._stereo_right_files) < 3):
            QMessageBox.warning(self, "警告", "左右相机各至少需要 3 张图像")
            return

        pattern_map = {
            "checkerboard (棋盘格)": "checkerboard",
            "circles (对称圆点阵)": "circles",
            "acircles (非对称圆点阵)": "acircles",
            "volume_dots (体标定板点阵)": "volume_dots",
        }
        pattern_type = pattern_map.get(
            self.pattern_type_combo.currentText(), "checkerboard"
        )
        pattern_size = (
            self.pattern_w_spin.value(), self.pattern_h_spin.value()
        )
        square_size = self.square_size_spin.value()

        objp = np.zeros(
            (pattern_size[0] * pattern_size[1], 3), np.float32
        )
        objp[:, :2] = np.mgrid[
            0:pattern_size[0], 0:pattern_size[1]
        ].T.reshape(-1, 2) * square_size

        objpoints = []
        imgpoints_l = []
        imgpoints_r = []

        for fl, fr in zip(self._stereo_left_files, self._stereo_right_files):
            img_l = cv2.imread(fl)
            img_r = cv2.imread(fr)
            if img_l is None or img_r is None:
                continue
            gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

            if pattern_type == "checkerboard":
                ret_l, corners_l = cv2.findChessboardCorners(
                    gray_l, pattern_size, None)
                ret_r, corners_r = cv2.findChessboardCorners(
                    gray_r, pattern_size, None)
            else:
                flags = (
                    cv2.CALIB_CB_SYMMETRIC_GRID if pattern_type == "circles"
                    else cv2.CALIB_CB_ASYMMETRIC_GRID
                )
                ret_l, corners_l = cv2.findCirclesGrid(
                    gray_l, pattern_size, flags=flags)
                ret_r, corners_r = cv2.findCirclesGrid(
                    gray_r, pattern_size, flags=flags)

            if ret_l and ret_r:
                if pattern_type == "checkerboard":
                    corners_l = cv2.cornerSubPix(
                        gray_l, corners_l, (11, 11), (-1, -1),
                        (cv2.TERM_CRITERIA_EPS +
                         cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                    corners_r = cv2.cornerSubPix(
                        gray_r, corners_r, (11, 11), (-1, -1),
                        (cv2.TERM_CRITERIA_EPS +
                         cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                objpoints.append(objp)
                imgpoints_l.append(corners_l)
                imgpoints_r.append(corners_r)

        if len(objpoints) < 3:
            QMessageBox.warning(self, "警告",
                "能同时检测到点的图像对不足 3 对")
            return

        h, w = gray_l.shape[:2]

        # 先分删定左右相?
        rms_l, K_l, D_l, _, _ = cv2.calibrateCamera(
            objpoints, imgpoints_l, (w, h), None, None)
        rms_r, K_r, D_r, _, _ = cv2.calibrateCamera(
            objpoints, imgpoints_r, (w, h), None, None)

        # 双目标定
        rms_stereo, R, T, E, F = cv2.stereoCalibrate(
            objpoints, imgpoints_l, imgpoints_r,
            K_l, D_l, K_r, D_r,
            (w, h),
            flags=cv2.CALIB_FIX_INTRINSIC
        )

        self.stereo_params = {
            'K_left': K_l.tolist(),
            'D_left': D_l.flatten().tolist(),
            'K_right': K_r.tolist(),
            'D_right': D_r.flatten().tolist(),
            'R': R.tolist(),
            'T': T.flatten().tolist(),
            'E': E.tolist(),
            'F': F.tolist(),
            'rms_stereo': float(rms_stereo),
            'rms_left': float(rms_l),
            'rms_right': float(rms_r),
            'image_size': [w, h],
            'n_pairs': len(objpoints)
        }

        report = (
            f"=== 双目标定完成 ===\n\n"
            f"图像对数: {len(objpoints)}\n"
            f"图像尺寸: {w} x {h}\n\n"
            f"--- 左相?---\n"
            f"RMS: {rms_l:.4f} px\n"
            f"fx={K_l[0,0]:.2f}  fy={K_l[1,1]:.2f}\n\n"
            f"--- 右相?---\n"
            f"RMS: {rms_r:.4f} px\n"
            f"fx={K_r[0,0]:.2f}  fy={K_r[1,1]:.2f}\n\n"
            f"--- 立体参数 ---\n"
            f"立体RMS: {rms_stereo:.4f} px\n"
            f"平移 T: [{T[0]:.2f}, {T[1]:.2f}, {T[2]:.2f}] mm\n"
            f"旋转? {np.degrees(np.arccos(np.clip((np.trace(R)-1)/2, -1, 1))):.2f} deg"
        )
        self.calib_result_text.setPlainText(report)
        self.statusBar().showMessage(
            f"双目标定完成, 立体RMS={rms_stereo:.4f}px"
        )
        QMessageBox.information(self, "完成",
            f"双目标定完成!\n立体标定RMS误差: {rms_stereo:.4f} px")

    # ---- 相机管理 ----

    def _run_stereo_calibration(self):
        """执行双目标定，支持棋盘格、圆点阵和体标定板点阵。"""
        import cv2

        if (not hasattr(self, "_stereo_left_files") or
                not hasattr(self, "_stereo_right_files")):
            QMessageBox.warning(self, "警告", "请先加载左右相机的标定图像")
            return

        if (len(self._stereo_left_files) < 3 or
                len(self._stereo_right_files) < 3):
            QMessageBox.warning(self, "警告", "左右相机各至少需要 3 张图像")
            return

        pattern_type = self._pattern_type_items().get(
            self.pattern_type_combo.currentText(), "checkerboard"
        )
        pattern_size = (
            self.pattern_w_spin.value(), self.pattern_h_spin.value()
        )
        square_size = self.square_size_spin.value()
        level_separation = self.level_separation_spin.value()

        calibrator = MultiCameraCalibrator(
            pattern_type=pattern_type,
            pattern_size=pattern_size,
            square_size=square_size,
            level_separation=level_separation
        )

        objpoints = []
        imgpoints_l = []
        imgpoints_r = []
        image_size = None

        for fl, fr in zip(self._stereo_left_files, self._stereo_right_files):
            img_l = cv2.imread(fl)
            img_r = cv2.imread(fr)
            if img_l is None or img_r is None:
                continue

            if image_size is None:
                image_size = (img_l.shape[1], img_l.shape[0])

            obs_l = calibrator.detect_pattern_observation(img_l)
            obs_r = calibrator.detect_pattern_observation(img_r)
            if obs_l is None or obs_r is None:
                continue

            map_l = {
                point_id: point
                for point_id, point in zip(obs_l.point_ids, obs_l.image_points.reshape(-1, 2))
            }
            map_r = {
                point_id: point
                for point_id, point in zip(obs_r.point_ids, obs_r.image_points.reshape(-1, 2))
            }
            common_ids = sorted(
                set(map_l.keys()) & set(map_r.keys()),
                key=lambda item: (item[1], item[0])
            )
            if len(common_ids) < 5:
                continue

            objpoints.append(calibrator._object_points_from_ids(common_ids))
            imgpoints_l.append(
                np.array([map_l[point_id] for point_id in common_ids], dtype=np.float32).reshape(-1, 1, 2)
            )
            imgpoints_r.append(
                np.array([map_r[point_id] for point_id in common_ids], dtype=np.float32).reshape(-1, 1, 2)
            )

        if len(objpoints) < 3 or image_size is None:
            QMessageBox.warning(self, "警告", "能同时检测到有效标定点的图像对不足 3 对")
            return

        rms_l, K_l, D_l, _, _ = cv2.calibrateCamera(
            objpoints, imgpoints_l, image_size, None, None
        )
        rms_r, K_r, D_r, _, _ = cv2.calibrateCamera(
            objpoints, imgpoints_r, image_size, None, None
        )

        rms_stereo, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
            objpoints,
            imgpoints_l,
            imgpoints_r,
            K_l,
            D_l,
            K_r,
            D_r,
            image_size,
            flags=cv2.CALIB_FIX_INTRINSIC
        )

        self.stereo_params = {
            'pattern_type': pattern_type,
            'pattern_size': list(pattern_size),
            'square_size': float(square_size),
            'K_left': K_l.tolist(),
            'D_left': D_l.flatten().tolist(),
            'K_right': K_r.tolist(),
            'D_right': D_r.flatten().tolist(),
            'R': R.tolist(),
            'T': T.flatten().tolist(),
            'E': E.tolist(),
            'F': F.tolist(),
            'rms_stereo': float(rms_stereo),
            'rms_left': float(rms_l),
            'rms_right': float(rms_r),
            'image_size': list(image_size),
            'n_pairs': len(objpoints)
        }

        report = (
            f"=== 双目标定完成 ===\n\n"
            f"标定板类? {pattern_type}\n"
            f"标定板尺? {pattern_size[0]} x {pattern_size[1]}\n"
            f"有效图像? {len(objpoints)}\n"
            f"图像尺寸: {image_size[0]} x {image_size[1]}\n\n"
            f"--- 左相?---\n"
            f"RMS: {rms_l:.4f} px\n"
            f"fx={K_l[0,0]:.2f}  fy={K_l[1,1]:.2f}\n\n"
            f"--- 右相?---\n"
            f"RMS: {rms_r:.4f} px\n"
            f"fx={K_r[0,0]:.2f}  fy={K_r[1,1]:.2f}\n\n"
            f"--- 立体参数 ---\n"
            f"立体RMS: {rms_stereo:.4f} px\n"
            f"平移 T: [{T[0,0]:.2f}, {T[1,0]:.2f}, {T[2,0]:.2f}] mm\n"
            f"旋转? {np.degrees(np.arccos(np.clip((np.trace(R)-1)/2, -1, 1))):.2f} deg"
        )
        self.calib_result_text.setPlainText(report)
        self.statusBar().showMessage(
            f"双目标定完成, 立体RMS={rms_stereo:.4f}px"
        )
        QMessageBox.information(
            self,
            "完成",
            f"双目标定完成!\n立体标定RMS误差: {rms_stereo:.4f} px"
        )

    def _add_camera(self):
        cam_id = self.cam_id_input.text().strip()
        if not cam_id:
            QMessageBox.warning(self, "警告", "请输入相机ID")
            return
        if cam_id in self.camera_calib_images:
            QMessageBox.warning(self, "警告", f"相机 {cam_id} 已存?")
            return
        self.camera_calib_images[cam_id] = []
        self.camera_list.addItem(f"{cam_id} (0 张标定图?")
        self.cam_id_input.clear()
        self.statusBar().showMessage(f"已添加相? {cam_id}")

    def _remove_camera(self):
        current = self.camera_list.currentRow()
        if current < 0:
            return
        cam_ids = list(self.camera_calib_images.keys())
        cam_id = cam_ids[current]
        self.camera_calib_images.pop(cam_id, None)
        self.camera_list.takeItem(current)
        self.statusBar().showMessage(f"已移除相? {cam_id}")

    def _load_calibration_images(self):
        current = self.camera_list.currentRow()
        if current < 0:
            QMessageBox.warning(self, "警告", "请先选择相机")
            return

        cam_ids = list(self.camera_calib_images.keys())
        cam_id = cam_ids[current]

        files, _ = QFileDialog.getOpenFileNames(
            self, f"选择相机 {cam_id} 的标定图?, """,
            "图像文件 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;所有文件 (*)"
        )

        if files:
            self.camera_calib_images[cam_id] = files
            self.camera_list.item(current).setText(
                f"{cam_id} ({len(files)} 张标定图?"
            )
            self.calib_image_label.setText(
                f"相机 {cam_id}: 已加?{len(files)} 张标定图?"
            )
            self.calib_image_label.setStyleSheet("color: green;")
            self.statusBar().showMessage(
                f"相机 {cam_id}: 已加载 {len(files)} 张标定图像"
            )
            # 加载后自动在右侧显示各相机标定图像
            self._refresh_calibration_preview(selected_key=cam_id)

    def _show_calibration_images_preview_legacy(self):
        """在右侧预览区显示已加载的相机图像缩略图。"""
        import cv2

        # 收集已加载图像的相机
        cams_with_images = {
            cam_id: files
            for cam_id, files in self.camera_calib_images.items()
            if files
        }

        if not cams_with_images:
            return

        # 为每个相机取第一张图像，拼接为网格
        cam_ids = list(cams_with_images.keys())
        n_cams = len(cam_ids)

        # 计算网格布局
        cols = min(n_cams, 3)
        rows = (n_cams + cols - 1) // cols

        cell_h, cell_w = 300, 400
        padding = 6
        title_h = 30

        canvas_w = cols * (cell_w + padding) + padding
        canvas_h = rows * (cell_h + title_h + padding) + padding

        canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 240

        for i, cam_id in enumerate(cam_ids):
            img_path = cams_with_images[cam_id][0]
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is None:
                continue

            row = i // cols
            col = i % cols

            x0 = padding + col * (cell_w + padding)
            y0 = padding + row * (cell_h + title_h + padding)

            # 相机标签
            cv2.putText(canvas, cam_id, (x0 + 4, y0 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 1)

            # 缩放图像?cell 区域
            img_h, img_w = img.shape[:2]
            scale = min((cell_h - 4) / img_h, (cell_w - 4) / img_w)
            new_w = int(img_w * scale)
            new_h = int(img_h * scale)
            resized = cv2.resize(img, (new_w, new_h))

            # 居中放置
            ix = x0 + (cell_w - new_w) // 2
            iy = y0 + title_h + (cell_h - new_h) // 2
            canvas[iy:iy + new_h, ix:ix + new_w] = resized

            # 边框
            cv2.rectangle(canvas, (x0, y0 + title_h),
                          (x0 + cell_w, y0 + title_h + cell_h),
                          (180, 180, 180), 1)

        # 总标题
        cv2.putText(canvas, "Camera Calibration Images Preview",
                    (padding, canvas_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 1)

        preview_path = os.path.join(
            self.visualizer.output_dir, 'calib_images_preview.png'
        )
        cv2.imwrite(preview_path, canvas)
        pixmap = QPixmap(preview_path)
        self.calib_preview_label.setPixmap(pixmap.scaled(
            self.calib_preview_label.size(), Qt.KeepAspectRatio,
            Qt.SmoothTransformation))

    # ---- 标定 ----

    def _show_calibration_images_preview(self):
        """兼容旧调用，转发到新的交互式预览器。"""
        self._refresh_calibration_preview()

    def _run_calibration(self):
        if len(self.camera_calib_images) < 1:
            QMessageBox.warning(self, "警告", "至少要添加 1 个相机才能进行相机标定")
            return

        for cam_id, imgs in self.camera_calib_images.items():
            if len(imgs) < 3:
                QMessageBox.warning(self, "警告", f"相机 {cam_id} 至少需要 3 张标定图像")
                return

        pattern_map = {
            "checkerboard (棋盘格)": "checkerboard",
            "circles (对称圆点阵)": "circles",
            "acircles (非对称圆点阵)": "acircles",
            "volume_dots (体标定板点阵)": "volume_dots",
        }
        pattern_type = pattern_map[self.pattern_type_combo.currentText()]
        pattern_size = (self.pattern_w_spin.value(), self.pattern_h_spin.value())
        square_size = self.square_size_spin.value()
        level_separation = self.level_separation_spin.value()

        self.calibrator = MultiCameraCalibrator(
            pattern_type=pattern_type,
            pattern_size=pattern_size,
            square_size=square_size,
            level_separation=level_separation
        )

        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)

        self.worker = CalibrationWorker(self.calibrator, self.camera_calib_images)
        self.worker.progress.connect(self._on_calib_progress)
        self.worker.finished.connect(self._on_calib_finished)
        self.worker.error.connect(self._on_calib_error)
        self.worker.start()

    def _on_calib_progress(self, msg):
        self.statusBar().showMessage(msg)
        self.calib_result_text.append(msg)

    def _on_calib_finished(self, results):
        self.calibration_results = results
        self.progress_bar.setVisible(False)
        self.statusBar().showMessage("标定完成")

        report = self.calibrator.get_calibration_report()
        self.calib_result_text.setPlainText(report)

        # 保存标定结果预?
        try:
            self.visualizer.plot_projection_comparison(
                {cam_id: np.random.rand(100, 100) for cam_id in results},
                "标定完成 - 各相机已就绪",
                os.path.join(self.visualizer.output_dir, 'calib_complete.png')
            )
            self.calib_preview_label.show_static_image(
                os.path.join(self.visualizer.output_dir, 'calib_complete.png'),
                title="标定完成预览"
            )
        except Exception:
            pass

        QMessageBox.information(self, "完成",
                                f"成功标定 {len(results)} 个相机！\n"
                                "详细信息请查看标定结果面板")

    def _on_calib_error(self, msg):
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "标定错误", msg)

    def _save_calibration(self):
        if not self.calibrator or not self.calibrator.camera_params:
            QMessageBox.warning(self, "警告", "没有标定结果可保存")
            return
        dir_path = QFileDialog.getExistingDirectory(self, "选择保存目录")
        if dir_path:
            self.calibrator.save_results(dir_path)
            QMessageBox.information(self, "完成", f"标定结果已保存至:\n{dir_path}")

    def _load_calibration(self):
        dir_path = QFileDialog.getExistingDirectory(self, "选择标定结果目录")
        if dir_path:
            try:
                self.calibrator = MultiCameraCalibrator.load_results(dir_path)
                self.calibration_results = self.calibrator.camera_params
                report = self.calibrator.get_calibration_report()
                self.calib_result_text.setPlainText(report)
                QMessageBox.information(
                    self, "完成",
                    f"已加载 {len(self.calibration_results)} 个相机的标定结果"
                )
            except Exception as e:
                QMessageBox.critical(self, "错误", f"加载失败: {e}")

    # ---- 气泡重建 ----

    def _load_bubble_images_batch(self):
        """批量加载气泡图像序列"""
        import cv2
        if not self.calibration_results:
            QMessageBox.warning(self, "警告", "请先完成相机标定或加载标定结果")
            return

        root_dir = QFileDialog.getExistingDirectory(
            self, "选择气泡图像序列根目录"
        )
        if not root_dir:
            return

        # 扏子文件夹作为时间?
        subdirs = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])

        if not subdirs:
            QMessageBox.warning(self, "警告",
                "所选目录下没有子文件夹。\n"
                "请确保每个时间点的图像存放在单独的子文件夹中。")
            return

        cam_ids = list(self.calibration_results.keys())
        self.bubble_timepoint_images.clear()
        self.bubble_timepoint_names.clear()
        self.bubble_batch_results.clear()

        img_exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
        loaded_count = 0

        for idx, subdir in enumerate(subdirs):
            subdir_path = os.path.join(root_dir, subdir)
            cam_images = {}
            for cam_id in cam_ids:
                # 在子文件夹中查找包含相机ID的图像文?
                # 策略1: 文件名包含cam_id
                # Match image files by camera ID
                files_in_dir = [
                    f for f in os.listdir(subdir_path)
                    if os.path.splitext(f)[1].lower() in img_exts
                ]
                files_in_dir.sort()

                # 查找匹配当前相机的图?
                matched = None
                for f in files_in_dir:
                    if cam_id.lower() in f.lower():
                        matched = os.path.join(subdir_path, f)
                        break

                # 如果没有精确匹配，尝试按编号分配
                if matched is None and len(files_in_dir) == len(cam_ids):
                    cam_idx = cam_ids.index(cam_id)
                    if cam_idx < len(files_in_dir):
                        matched = os.path.join(subdir_path, files_in_dir[cam_idx])

                if matched and os.path.isfile(matched):
                    img = cv2.imread(matched, cv2.IMREAD_COLOR)
                    if img is not None:
                        cam_images[cam_id] = img

            if cam_images:
                self.bubble_timepoint_images[idx] = cam_images
                self.bubble_timepoint_names[idx] = subdir
                loaded_count += 1

        if loaded_count > 0:
            # 设置时间点择?
            self._setup_bubble_timepoint_selector()
            # 臊设置当前时间点的单帧数据（兼容旧方法?
            self._set_current_bubble_timepoint(0)
            self.bubble_status_label.setText(
                f"已加?{loaded_count} 丗间点，每?{len(cam_ids)} 相机"
            )
            self.bubble_status_label.setStyleSheet("color: green; font-weight: bold;")
            self.recon_log.append(
                f"=== 批量加载完成 ===\n"
                f"根目? {root_dir}\n"
                f"时间点数? {loaded_count}\n"
                f"相机数量: {len(cam_ids)}\n"
            )
            self.statusBar().showMessage(
                f"已批量加载 {loaded_count} 个时间点的气泡图像"
            )
        else:
            QMessageBox.warning(self, "警告",
                "未能从子文件夹中匹配到相机图像。\n"
                "请确保子文件夹内的图像文件名包含对应相机ID。")

    def _setup_bubble_timepoint_selector(self):
        """初始化气泡重建的时间点选择器。"""
        n = len(self.bubble_timepoint_images)
        if n == 0:
            return

        self.bubble_tp_combo.blockSignals(True)
        self.bubble_tp_slider.blockSignals(True)

        self.bubble_tp_combo.clear()
        for idx in sorted(self.bubble_timepoint_images.keys()):
            name = self.bubble_timepoint_names.get(idx, f"t{idx}")
            self.bubble_tp_combo.addItem(f"{name} (t{idx})", userData=idx)

        self.bubble_tp_slider.setRange(0, n - 1)
        self.bubble_tp_slider.setValue(0)

        self.bubble_tp_combo.setEnabled(True)
        self.bubble_tp_slider.setEnabled(True)
        self.bubble_tp_prev_btn.setEnabled(n > 1)
        self.bubble_tp_next_btn.setEnabled(n > 1)

        self._update_bubble_tp_label(0)

        self.bubble_tp_combo.blockSignals(False)
        self.bubble_tp_slider.blockSignals(False)

        self.bubble_tp_info_label.setText(
            f"?{n} 丗间点 | 当前: {self.bubble_timepoint_names.get(0, '')}"
        )
        self.bubble_tp_info_label.setStyleSheet("color: #1565C0; font-weight: bold;")

    def _update_bubble_tp_label(self, idx):
        n = len(self.bubble_timepoint_images)
        name = self.bubble_timepoint_names.get(idx, f"t{idx}")
        self.bubble_tp_label.setText(f"{name}  ({idx + 1} / {n})")

    def _set_current_bubble_timepoint(self, idx):
        """设置当前气泡重建时间点，更新单帧缓存"""
        if idx not in self.bubble_timepoint_images:
            return
        self.current_bubble_timepoint = idx
        self.camera_bubble_images = self.bubble_timepoint_images[idx]

        # 更新预?
        self._preview_bubble_timepoint(idx)

        # 如果已有重建结果，切换到对应结果
        if idx in self.bubble_batch_results:
            self._last_results = self.bubble_batch_results[idx]
            self._switch_bubble_viz(idx)

    def _preview_bubble_timepoint(self, idx):
        """预览指定时间点的气泡图像。"""
        import cv2
        if idx not in self.bubble_timepoint_images:
            return
        cam_imgs = self.bubble_timepoint_images[idx]
        if not cam_imgs:
            return

        # 拼接各相机图像为?
        imgs = list(cam_imgs.values())
        if not imgs:
            return

        # Compose multi-camera preview
        target_h = 300
        resized = []
        for img in imgs:
            h, w = img.shape[:2]
            scale = target_h / h
            new_w = int(w * scale)
            resized.append(cv2.resize(img, (new_w, target_h)))

        combined = np.hstack(resized)
        name = self.bubble_timepoint_names.get(idx, f"t{idx}")

        # 添加文字标注
        cv2.putText(combined, f"Time: {name}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        preview_path = os.path.join(
            self.visualizer.output_dir, f'bubble_preview_t{idx}.png'
        )
        cv2.imwrite(preview_path, combined)
        pixmap = QPixmap(preview_path)
        self.proj_preview_label.setPixmap(pixmap.scaled(
            self.proj_preview_label.size(), Qt.KeepAspectRatio,
            Qt.SmoothTransformation))

    def _on_bubble_tp_changed(self, idx):
        """滑块变更"""
        if idx < 0:
            return
        self.bubble_tp_combo.blockSignals(True)
        self.bubble_tp_combo.setCurrentIndex(idx)
        self.bubble_tp_combo.blockSignals(False)
        self._update_bubble_tp_label(idx)
        self._set_current_bubble_timepoint(idx)

    def _on_bubble_tp_combo_changed(self, idx):
        """组合框变更。"""
        if idx < 0:
            return
        self.bubble_tp_slider.blockSignals(True)
        self.bubble_tp_slider.setValue(idx)
        self.bubble_tp_slider.blockSignals(False)
        self._update_bubble_tp_label(idx)
        self._set_current_bubble_timepoint(idx)

    def _bubble_tp_prev(self):
        idx = self.bubble_tp_slider.value()
        if idx > 0:
            self.bubble_tp_slider.setValue(idx - 1)

    def _bubble_tp_next(self):
        idx = self.bubble_tp_slider.value()
        n = len(self.bubble_timepoint_images)
        if idx < n - 1:
            self.bubble_tp_slider.setValue(idx + 1)

    def _switch_bubble_viz(self, idx):
        """切换可视化标签页到指定时间点的结果。"""
        if idx not in self.bubble_batch_results:
            return
        try:
            result = self.bubble_batch_results[idx]
            name = self.bubble_timepoint_names.get(idx, f"t{idx}")
            self.visualizer.create_report_figure(
                result['volume'], result['points'],
                result.get('projections', {}),
                result['stats']
            )
            path = os.path.join(self.visualizer.output_dir,
                                'reconstruction_report.png')
            self._show_image(path)
            self.recon_log.append(f"已切换到时间?{name} 的重建结?")
        except Exception as e:
            self.recon_log.append(f"预览刷新失败: {e}")

    def _load_bubble_images(self):
        """单帧加载气泡图像（兼容旧接口）。"""
        import cv2
        if not self.calibration_results:
            QMessageBox.warning(self, "警告", "请先完成相机标定或加载标定结果")
            return

        cam_ids = list(self.calibration_results.keys())
        self.camera_bubble_images = {}

        for cam_id in cam_ids:
            file_path, _ = QFileDialog.getOpenFileName(
                self, f"选择相机 {cam_id} 的气泡图?, """,
                "图像文件 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
            )
            if file_path:
                img = cv2.imread(file_path, cv2.IMREAD_COLOR)
                if img is not None:
                    self.camera_bubble_images[cam_id] = img

        if self.camera_bubble_images:
            self.bubble_status_label.setText(
                f"已加?{len(self.camera_bubble_images)} 丛机的气泡图像（单帧）"
            )
            self.bubble_status_label.setStyleSheet("color: green;")
            self.statusBar().showMessage("气泡图像已加载")
        else:
            self.bubble_status_label.setText("未加载气泡图像")
            self.bubble_status_label.setStyleSheet("color: gray;")

    def _load_reference_images(self):
        """加载背景参考图。"""
        if not self.calibration_results:
            QMessageBox.warning(self, "警告", "请先完成相机标定")
            return

        cam_ids = list(self.calibration_results.keys())

        for cam_id in cam_ids:
            file_path, _ = QFileDialog.getOpenFileName(
                self, f"选择相机 {cam_id} 的背晏考图（可跳过?, """,
                "图像文件 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
            )
            if file_path:
                import cv2
                img = cv2.imread(file_path, cv2.IMREAD_COLOR)
                if img is not None:
                    self.camera_reference_images[cam_id] = img

        if self.camera_reference_images:
            self.statusBar().showMessage(
                f"已加?{len(self.camera_reference_images)} 张背晏考图"
            )

    def _on_recon_algo_changed(self, index):
        """重建算法切换时，显示/隐藏Conv-SMART专属参数"""
        algo = self.recon_algo_combo.currentText()
        is_conv = algo == "Conv-SMART"
        self.convsmart_group.setVisible(is_conv)

    def _run_reconstruction(self):
        """执行重建（自动判断单帧或批量）。"""
        if self.bubble_timepoint_images:
            self._run_reconstruction_batch()
        else:
            self._run_reconstruction_single()

    def _run_reconstruction_single(self):
        """单帧重建"""
        if not self.calibration_results:
            QMessageBox.warning(self, "警告", "请先完成相机标定或加载标定结果")
            return
        if not self.camera_bubble_images:
            QMessageBox.warning(self, "警告", "请先加载气泡图像")
            return

        # Preprocess by camera parameters
        self.image_processor = BubbleImageProcessor(
            background_method=self.bg_method_combo.currentText(),
            threshold_method=self.threshold_method_combo.currentText()
        )

        # Create projections
        camera_params_for_preprocess = {}
        camera_params_for_recon = {}

        for cam_id, params in self.calibration_results.items():
            camera_params_for_preprocess[cam_id] = {
                'camera_matrix': params.camera_matrix,
                'dist_coeffs': params.dist_coeffs
            }

            P = self.calibrator.compute_projection_matrix(cam_id)
            K = np.array(params.camera_matrix)
            K_inv = np.linalg.inv(K)

            camera_params_for_recon[cam_id] = {
                'P': P,
                'K_inv': K_inv
            }

        # 图像预?
            self.recon_log.append("=== 图像预处理 ===")
        self.projections = self.image_processor.prepare_projection_data(
            self.camera_bubble_images,
            camera_params_for_preprocess,
            self.camera_reference_images,
            projection_type=self.proj_type_combo.currentText()
        )

        # Image preview
        try:
            self.visualizer.plot_projection_comparison(
                self.projections, "投影预处理结果",
                os.path.join(self.visualizer.output_dir, 'proj_preview.png')
            )
            pixmap = QPixmap(os.path.join(
                self.visualizer.output_dir, 'proj_preview.png'))
            self.proj_preview_label.setPixmap(pixmap.scaled(
                self.proj_preview_label.size(), Qt.KeepAspectRatio,
                Qt.SmoothTransformation))
        except Exception:
            pass

        # 配置重建参数（根据算法类型）
        algo_name = self.recon_algo_combo.currentText()
        config = ReconstructionConfig(
            grid_size=(self.grid_x.value(), self.grid_y.value(),
                       self.grid_z.value()),
            domain_size=(self.domain_x.value(), self.domain_y.value(),
                         self.domain_z.value()),
            relaxation_factor=self.relax_spin.value(),
            max_iterations=self.iter_spin.value(),
            voxel_threshold=self.threshold_spin.value(),
            algorithm=algo_name,
            # Conv-SMART 专属参数
            conv_kernel_size=self.conv_kernel_spin.value(),
            psf_type=self.psf_type_combo.currentText(),
            psf_sigma=self.psf_sigma_spin.value(),
            use_fft_convolution=self.fft_conv_check.isChecked(),
        )

        self.reconstructor = create_reconstructor(config)

        # Start worker thread
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, self.iter_spin.value())
        self.progress_bar.setValue(0)

        self.recon_worker = ReconstructionWorker(
            self.reconstructor, self.projections, camera_params_for_recon
        )
        self.recon_worker.progress.connect(self.recon_log.append)
        self.recon_worker.iteration_done.connect(
            lambda i, e: self.progress_bar.setValue(i))
        self.recon_worker.finished.connect(self._on_recon_finished)
        self.recon_worker.error.connect(self._on_recon_error)
        self.recon_worker.start()

    def _run_reconstruction_batch(self):
        """批量重建所有时间点。"""
        if not self.calibration_results:
            QMessageBox.warning(self, "警告", "请先完成相机标定或加载标定结果")
            return
        if not self.bubble_timepoint_images:
            QMessageBox.warning(self, "警告", "请先批量加载气泡图像序列")
            return

        # Create projections
        camera_params_for_preprocess = {}
        camera_params_for_recon = {}

        for cam_id, params in self.calibration_results.items():
            camera_params_for_preprocess[cam_id] = {
                'camera_matrix': params.camera_matrix,
                'dist_coeffs': params.dist_coeffs
            }

            P = self.calibrator.compute_projection_matrix(cam_id)
            K = np.array(params.camera_matrix)
            K_inv = np.linalg.inv(K)

            camera_params_for_recon[cam_id] = {
                'P': P,
                'K_inv': K_inv,
                'camera_matrix': params.camera_matrix,
                'dist_coeffs': params.dist_coeffs
            }

        # 配置
        self.image_processor = BubbleImageProcessor(
            background_method=self.bg_method_combo.currentText(),
            threshold_method=self.threshold_method_combo.currentText()
        )

        config = ReconstructionConfig(
            grid_size=(self.grid_x.value(), self.grid_y.value(),
                       self.grid_z.value()),
            domain_size=(self.domain_x.value(), self.domain_y.value(),
                         self.domain_z.value()),
            relaxation_factor=self.relax_spin.value(),
            max_iterations=self.iter_spin.value(),
            voxel_threshold=self.threshold_spin.value(),
            algorithm=self.recon_algo_combo.currentText(),
            conv_kernel_size=self.conv_kernel_spin.value(),
            psf_type=self.psf_type_combo.currentText(),
            psf_sigma=self.psf_sigma_spin.value(),
            use_fft_convolution=self.fft_conv_check.isChecked(),
        )

        self.reconstructor = create_reconstructor(config)

        total = len(self.bubble_timepoint_images)
        self.bubble_batch_results.clear()
        self.bubble_batch_progress.setVisible(True)
        self.bubble_batch_progress.setRange(0, total)
        self.bubble_batch_progress.setValue(0)

        self.batch_worker = BatchReconstructionWorker(
            reconstructor=self.reconstructor,
            bubble_images_sequence=self.bubble_timepoint_images,
            camera_params=camera_params_for_recon,
            reference_images=self.camera_reference_images,
            image_processor=self.image_processor,
            projection_type=self.proj_type_combo.currentText()
        )
        self.batch_worker.progress.connect(self.recon_log.append)
        self.batch_worker.timepoint_done.connect(
            self._on_batch_tp_done)
        self.batch_worker.all_done.connect(self._on_batch_all_done)
        self.batch_worker.error.connect(self._on_batch_error)
        self.batch_worker.start()

        self.recon_log.append(f"=== 开始批量重建 ===\n共 {total} 个时间点")
        self.recon_log.append("批量重建进行中...")

    def _on_batch_tp_done(self, tp_idx, result):
        """单个时间点重建完成。"""
        self.bubble_batch_results[tp_idx] = result
        self._last_results = result
        name = self.bubble_timepoint_names.get(tp_idx, f"t{tp_idx}")
        self.recon_log.append(
            f"  时间点 t{tp_idx} ({name}) 完成 | "
            f"体素: {result['stats']['nonzero_voxels']} | "
            f"点云: {len(result['points'])} pts"
        )
        self.bubble_batch_progress.setValue(len(self.bubble_batch_results))

    def _on_batch_all_done(self, all_results):
        """批量重建全部完成。"""
        self.bubble_batch_progress.setVisible(False)
        self.progress_bar.setVisible(False)
        total = len(all_results)
        self.recon_log.append(
            f"\n=== 批量重建完成 ===\n"
            f"成功处理 {total} 个时间点\n"
            f"请使用底部时间点滑块切换查看各时刻结果"
        )
        self.statusBar().showMessage(f"批量重建完成: {total} 个时间点")
        QMessageBox.information(
            self, "完成",
            f"批量重建完成\n成功处理 {total} 个时间点\n"
            f"使用底部时间点滑块切换查看结果"
        )
        self.content_stack.setCurrentIndex(2)
        self._nav_buttons["reconstruction"].setChecked(True)
        QTimer.singleShot(500, self._show_report)

    def _on_batch_error(self, msg):
        self.bubble_batch_progress.setVisible(False)
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "批量重建错误", msg)

    def _on_recon_finished(self, results):
        self.progress_bar.setVisible(False)
        self.statusBar().showMessage("重建完成")

        self._last_results = results

        stats = results['stats']
        report = (
            f"=== 重建完成 ===\n\n"
            f"网格尺寸: {stats['grid_size']}\n"
            f"重建区域: {stats['domain_size_mm']} mm\n"
            f"有效体素: {stats['nonzero_voxels']} / {stats['total_voxels']}\n"
            f"填充率: {stats['fill_fraction']*100:.1f}%\n"
            f"点云点数: {len(results['points'])}\n"
            f"最终误差: {results['errors'][-1]:.4f}\n\n"
            f"请使用右侧可视化工具栏查看结果"
        )
        self.recon_log.append(report)

        self.content_stack.setCurrentIndex(2)
        self._nav_buttons["reconstruction"].setChecked(True)
        QTimer.singleShot(500, self._show_report)

    def _on_recon_error(self, msg):
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "重建错误", msg)

    # ---- 可视化 ----

    def _show_point_cloud(self):
        if not hasattr(self, '_last_results'):
            QMessageBox.warning(self, "警告", "请先完成重建")
            return
        try:
            self.content_stack.setCurrentIndex(2)
            self._nav_buttons["reconstruction"].setChecked(True)
            path = self.visualizer.plot_point_cloud(
                self._last_results['points'],
                self._last_results.get('normals')
            )
            self._show_image(path)
        except Exception as e:
            QMessageBox.critical(self, "错误", str(e))

    def _show_volume_slices(self):
        if not hasattr(self, '_last_results'):
            QMessageBox.warning(self, "警告", "请先完成重建")
            return
        try:
            # 确保显示在气泡重建页
            self.content_stack.setCurrentIndex(2)
            self._nav_buttons["reconstruction"].setChecked(True)
            path = self.visualizer.plot_volume_slices(
                self._last_results['volume']
            )
            self._show_image(path)
        except Exception as e:
            QMessageBox.critical(self, "错误", str(e))

    def _show_projections(self):
        if not self.projections:
            QMessageBox.warning(self, "警告", "没有投影数据")
            return
        try:
            # 确保显示在气泡重建页
            self.content_stack.setCurrentIndex(2)
            self._nav_buttons["reconstruction"].setChecked(True)
            path = self.visualizer.plot_projection_comparison(self.projections)
            self._show_image(path)
        except Exception as e:
            QMessageBox.critical(self, "错误", str(e))

    def _show_report(self):
        if not hasattr(self, '_last_results'):
            return
        try:
            # 确保显示在气泡重建页
            self.content_stack.setCurrentIndex(2)
            self._nav_buttons["reconstruction"].setChecked(True)
            self.visualizer.create_report_figure(
                self._last_results['volume'],
                self._last_results['points'],
                self.projections if not self.bubble_batch_results else
                    self._last_results.get('projections', self.projections),
                self._last_results['stats']
            )
            path = os.path.join(self.visualizer.output_dir,
                                'reconstruction_report.png')
            self._show_image(path)
        except Exception as e:
            QMessageBox.critical(self, "错误", str(e))

    def _show_batch_summary(self):
        """显示批量处理结果摘要。"""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        if not self.bubble_batch_results and not self.piv_batch_results:
            QMessageBox.warning(self, "警告", "没有批量处理结果。\n请先执行批量重建或批量PIV。")
            return

        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        fig.suptitle('Batch Processing Summary', fontsize=14, fontweight='bold')

        # 气泡重建摘要
        if self.bubble_batch_results:
            ax1 = axes[0]
            tps = sorted(self.bubble_batch_results.keys())
            names = [self.bubble_timepoint_names.get(t, f't{t}') for t in tps]
            voxel_counts = [
                self.bubble_batch_results[t]['stats']['nonzero_voxels']
                for t in tps
            ]
            point_counts = [
                len(self.bubble_batch_results[t]['points'])
                for t in tps
            ]
            errors = [
                self.bubble_batch_results[t]['errors'][-1]
                if self.bubble_batch_results[t]['errors'] else 0
                for t in tps
            ]

            ax1_twin = ax1.twinx()
            x_pos = range(len(names))
            bars = ax1.bar(x_pos, voxel_counts, alpha=0.6, color='steelblue',
                           label='Nonzero Voxels')
            ax1_twin.plot(list(x_pos), point_counts, 'ro-', label='Point Count')
            ax1_twin.plot(list(x_pos), errors, 'g^--', label='Final Error')

            ax1.set_xticks(list(x_pos))
            ax1.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
            ax1.set_ylabel('Voxels')
            ax1_twin.set_ylabel('Count / Error')
            ax1.set_title('Bubble Reconstruction Summary')
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax1_twin.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left',
                       fontsize=8)

        # PIV 结果摘要
        if self.piv_batch_results:
            ax2 = axes[1]
            tps = sorted(self.piv_batch_results.keys())
            names = [self.particle_timepoint_names.get(t, f't{t}') for t in tps]
            avg_speeds = []
            max_speeds = []
            particle_counts = []

            for t in tps:
                vf = self.piv_batch_results[t]['velocity_result']['velocity_field']
                speed = np.linalg.norm(vf, axis=-1)
                avg_speeds.append(speed.mean())
                max_speeds.append(speed.max())
                particle_counts.append(
                    len(self.piv_batch_results[t]['particles_3d_frame1'])
                )

            ax2_twin = ax2.twinx()
            x_pos = range(len(names))
            ax2.bar(x_pos, avg_speeds, alpha=0.6, color='coral',
                    label='Avg Speed (mm/s)')
            ax2.errorbar(list(x_pos), avg_speeds,
                         yerr=[max_speeds[i] - avg_speeds[i] for i in range(len(avg_speeds))],
                         fmt='none', ecolor='red', capsize=3, label='Max')
            ax2_twin.plot(list(x_pos), particle_counts, 'b^-',
                          label='Particle Count')

            ax2.set_xticks(list(x_pos))
            ax2.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
            ax2.set_ylabel('Speed (mm/s)')
            ax2_twin.set_ylabel('Particle Count')
            ax2.set_title('PIV Velocity Field Summary')
            lines1, labels1 = ax2.get_legend_handles_labels()
            lines2, labels2 = ax2_twin.get_legend_handles_labels()
            ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left',
                       fontsize=8)

        plt.tight_layout()
        summary_path = os.path.join(self.visualizer.output_dir,
                                     'batch_summary.png')
        plt.savefig(summary_path, dpi=150, bbox_inches='tight')
        plt.close()

        # 根据结果类型自动切换到对应页面
        if self.piv_batch_results and not self.bubble_batch_results:
            self.content_stack.setCurrentIndex(4)
            self._nav_buttons["particle"].setChecked(True)
        else:
            self.content_stack.setCurrentIndex(2)
            self._nav_buttons["reconstruction"].setChecked(True)
        self._show_image(summary_path)

    # 注：_show_image 定义在文件末尾（使用正确的页面索引映射）

    def _show_particle_viz(self):
        """粒子追踪页面：显示粒子 3D 位置。"""
        if not self.particles_3d_frame1:
            QMessageBox.warning(self, "警告", "请先完成粒子3D重建")
            return
        try:
            # 切换到PIV页面
            self.content_stack.setCurrentIndex(4)
            self._nav_buttons["particle"].setChecked(True)
            self.visualizer.plot_particle_positions_3d(
                self.particles_3d_frame1,
                save_path=os.path.join(self.visualizer.output_dir,
                                        'particles_3d_viz.png')
            )
            self._show_image(os.path.join(
                self.visualizer.output_dir, 'particles_3d_viz.png'))
        except Exception as e:
            QMessageBox.critical(self, "错误", str(e))

    def _show_velocity_viz(self):
        """粒子追踪页面: 显示速度场可视化"""
        if not self._velocity_result:
            QMessageBox.warning(self, "警告", "请先计算速度场")
            return
        try:
            # 切换到PIV页面
            self.content_stack.setCurrentIndex(4)
            self._nav_buttons["particle"].setChecked(True)
            self.visualizer.create_velocity_report(
                self._velocity_result,
                self.particles_3d_frame1,
                save_path=os.path.join(self.visualizer.output_dir,
                                        'velocity_viz.png')
            )
            self._show_image(os.path.join(
                self.visualizer.output_dir, 'velocity_viz.png'))
        except Exception as e:
            QMessageBox.critical(self, "错误", str(e))

    # ---- Raytrace 3D reconstruction (Silhouette + Snell correction) ----

    def _create_raytrace_page(self):
        """创建单相机 3D 重建页面：Silhouette + 光线追踪。"""
        page = QWidget()
        layout = QHBoxLayout(page)

        # 左侧: 控制面板
        left_panel = QWidget()
        left_panel.setMaximumWidth(self._sp(340))
        left_panel.setMinimumWidth(self._sp(300))
        left_layout = QVBoxLayout(left_panel)

        # 图像加载
        img_group = QGroupBox("📷 图像加载")
        img_layout = QVBoxLayout()

        btn_load_rt = QPushButton("选择气泡图像")
        btn_load_rt.setStyleSheet("QPushButton { font-weight: bold; padding: 6px; }")
        btn_load_rt.clicked.connect(self._load_rt_image)
        img_layout.addWidget(btn_load_rt)

        self.rt_img_path_label = QLabel("未加载图像")
        self.rt_img_path_label.setWordWrap(True)
        self.rt_img_path_label.setStyleSheet("color: gray;")
        img_layout.addWidget(self.rt_img_path_label)

        self.rt_img_preview = QLabel()
        self.rt_img_preview.setMaximumHeight(self._sp(150))
        self.rt_img_preview.setAlignment(Qt.AlignCenter)
        self.rt_img_preview.setStyleSheet("border: 1px dashed #ccc;")
        img_layout.addWidget(self.rt_img_preview)

        img_group.setLayout(img_layout)
        left_layout.addWidget(img_group)

        # 参数设置
        param_group = QGroupBox("⚙️ 参数设置")
        param_form = QGridLayout(param_group)

        param_form.addWidget(QLabel("相机夹角 θ:"), 0, 0)
        self.rt_spin_theta = QDoubleSpinBox()
        self.rt_spin_theta.setRange(0, 180)
        self.rt_spin_theta.setValue(90)
        self.rt_spin_theta.setSuffix("°")
        self.rt_spin_theta.setToolTip("相机2与相机1的夹角")
        param_form.addWidget(self.rt_spin_theta, 0, 1)

        param_form.addWidget(QLabel("相机1角度:"), 1, 0)
        self.rt_spin_cam1 = QDoubleSpinBox()
        self.rt_spin_cam1.setRange(0, 90)
        self.rt_spin_cam1.setValue(45)
        self.rt_spin_cam1.setSuffix("°")
        self.rt_spin_cam1.setToolTip("相机1的角度")
        param_form.addWidget(self.rt_spin_cam1, 1, 1)

        param_form.addWidget(QLabel("物距 (px):"), 2, 0)
        self.rt_spin_length = QDoubleSpinBox()
        self.rt_spin_length.setRange(-5000, 0)
        self.rt_spin_length.setValue(-1000)
        self.rt_spin_length.setSingleStep(100)
        self.rt_spin_length.setToolTip("物体到相机的像素距离")
        param_form.addWidget(self.rt_spin_length, 2, 1)

        param_form.addWidget(QLabel("气泡直径 (px):"), 3, 0)
        self.rt_spin_diameter = QDoubleSpinBox()
        self.rt_spin_diameter.setRange(10, 2000)
        self.rt_spin_diameter.setValue(100)
        self.rt_spin_diameter.setSingleStep(10)
        self.rt_spin_diameter.setToolTip("气泡等价直径（像素）")
        param_form.addWidget(self.rt_spin_diameter, 3, 1)

        param_form.addWidget(QLabel("重建面:"), 4, 0)
        self.rt_spin_face = QComboBox()
        self.rt_spin_face.addItems(["前后双面", "仅前面"])
        self.rt_spin_face.setCurrentIndex(0)
        param_form.addWidget(self.rt_spin_face, 4, 1)

        param_form.addWidget(QLabel("折射率 μ:"), 5, 0)
        self.rt_spin_miu2 = QDoubleSpinBox()
        self.rt_spin_miu2.setRange(1.0, 3.0)
        self.rt_spin_miu2.setValue(1.3)
        self.rt_spin_miu2.setSingleStep(0.05)
        self.rt_spin_miu2.setToolTip("气泡折射率")
        param_form.addWidget(self.rt_spin_miu2, 5, 1)

        left_layout.addWidget(param_group)

        # 计算控制
        calc_group = QGroupBox("🚀 计算")
        calc_layout = QVBoxLayout(calc_group)

        btn_run_all_rt = QPushButton("运行全部 (Step 1-5)")
        btn_run_all_rt.setStyleSheet(
            "QPushButton { background-color: #00796B; color: white; "
            "font-size: 13px; padding: 10px; border-radius: 5px; font-weight: bold; }"
            "QPushButton:hover { background-color: #00897B; }"
            "QPushButton:disabled { background-color: #666; color: #aaa; }"
        )
        btn_run_all_rt.clicked.connect(self._run_rt_all)
        calc_layout.addWidget(btn_run_all_rt)

        # Step buttons
        steps_layout = QGridLayout()
        self.rt_step_buttons = []
        step_names = [
            ("Step 1: 二值化", 1),
            ("Step 2: Silhouette重建", 2),
            ("Step 3: 面法向初始化", 3),
            ("Step 4: 光线追踪修正", 4),
            ("Step 5: 曲面构建", 5),
        ]
        for i, (name, step) in enumerate(step_names):
            btn = QPushButton(name)
            btn.setStyleSheet("QPushButton { padding: 5px; font-size: 11px; }")
            btn.clicked.connect(lambda checked, s=step: self._run_rt_step(s))
            steps_layout.addWidget(btn, i // 2, i % 2)
            self.rt_step_buttons.append(btn)
        calc_layout.addLayout(steps_layout)

        self.rt_progress_bar = QProgressBar()
        self.rt_progress_bar.setValue(0)
        calc_layout.addWidget(self.rt_progress_bar)

        self.rt_status_label = QLabel("就绪")
        self.rt_status_label.setStyleSheet("color: #00897B; font-weight: bold;")
        calc_layout.addWidget(self.rt_status_label)

        calc_group.setLayout(calc_layout)
        left_layout.addWidget(calc_group)

        # 导出
        export_group = QGroupBox("💾 导出")
        export_layout = QVBoxLayout(export_group)

        btn_export_pts = QPushButton("导出点云 (.csv)")
        btn_export_pts.clicked.connect(self._export_rt_points)
        export_layout.addWidget(btn_export_pts)

        btn_export_fig = QPushButton("导出3D图 (.png)")
        btn_export_fig.clicked.connect(self._export_rt_figure)
        export_layout.addWidget(btn_export_fig)

        left_layout.addWidget(export_group)
        left_layout.addStretch()

        # ??: ?? + ??
        right_container = QWidget()
        right_container_layout = QVBoxLayout(right_container)
        right_container_layout.setContentsMargins(
            self._sp(6), self._sp(6), self._sp(6), self._sp(6)
        )

        # Log and preview area
        mid_splitter = QSplitter(Qt.Vertical)

        self.rt_log = QTextEdit()
        self.rt_log.setReadOnly(True)
        self.rt_log.setFontFamily("Consolas")
        self.rt_log.setPlaceholderText("光线追踪重建日志")
        mid_splitter.addWidget(self.rt_log)

        self.rt_preview_label = QLabel()
        self.rt_preview_label.setAlignment(Qt.AlignCenter)
        self.rt_preview_label.setMinimumSize(self._sp(400), self._sp(300))
        self.rt_preview_label.setStyleSheet(
            "border: 1px solid #ccc; background-color: #f8f8f8; padding: 10px;"
        )
        self.rt_preview_label.setText("光线追踪结果预览")
        mid_splitter.addWidget(self.rt_preview_label)

        mid_splitter.setSizes([self._sp(250), self._sp(400)])
        right_container_layout.addWidget(mid_splitter, stretch=1)

        # Visualization toolbar
        viz_toolbar = self._make_viz_toolbar("raytrace")
        right_container_layout.addWidget(viz_toolbar)

        layout.addWidget(left_panel, stretch=1)
        layout.addWidget(right_container, stretch=2)

        self.content_stack.addWidget(page)

    # --- Raytrace helper methods ---

    def _rt_log_msg(self, msg):
        """追加日志消息"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.rt_log.append(f"[{timestamp}] {msg}")

    def _set_rt_buttons_enabled(self, enabled):
        """启用或禁用计算按钮。"""
        for btn in self.rt_step_buttons:
            btn.setEnabled(enabled)

    # --- 计算线程 ---

    class _RTComputeThread(QThread):
        """光线追踪后台计算线程"""
        progress = pyqtSignal(int, str)   # percent, message
        finished = pyqtSignal(str)        # result message
        error = pyqtSignal(str)           # error message

        def __init__(self, processor, step, **kwargs):
            super().__init__()
            self.processor = processor
            self.step = step
            self.kwargs = kwargs

        def run(self):
            try:
                if self.step == 1:
                    self.processor.run_step1_binary()
                    self.finished.emit("步骤1完成：二值化处理")
                elif self.step == 2:
                    self.processor.run_step2_silhouette(**self.kwargs)
                    self.finished.emit("步骤2完成：Silhouette 三维重建")
                elif self.step == 3:
                    self.processor.run_step3_raytrace_init()
                    self.finished.emit("步骤3完成：面法向初始化")
                elif self.step == 4:
                    self.processor.run_step4_raytrace_adjust(
                        progress_callback=lambda p, m: self.progress.emit(p, m)
                    )
                    self.finished.emit("步骤4完成：光线追踪修正")
                elif self.step == 5:
                    self.processor.run_step5_visualize()
                    self.finished.emit("步骤5完成：曲面构建")
                elif self.step == 'all':
                    self.progress.emit(5, "步骤1：二值化处理...")
                    self.processor.run_step1_binary()
                    self.progress.emit(15, "步骤2：Silhouette 三维重建...")
                    self.processor.run_step2_silhouette(**self.kwargs)
                    self.progress.emit(30, "步骤3：面法向初始化...")
                    self.processor.run_step3_raytrace_init()
                    self.progress.emit(40, "步骤4：光线追踪修正...")
                    self.processor.run_step4_raytrace_adjust(
                        progress_callback=lambda p, m: self.progress.emit(
                            40 + int(50 * p / 100), m)
                    )
                    self.progress.emit(95, "步骤5：曲面构建...")
                    self.processor.run_step5_visualize()
                    self.finished.emit("全部计算完成")
            except Exception as e:
                self.error.emit(str(e))

    # --- 图像加载 ---

    def _load_rt_image(self):
        """加载气泡图像用于光线追踪"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择气泡图像", "",
            "图像文件 (*.bmp *.png *.jpg *.tif *.tiff);;所有文件 (*)"
        )
        if not file_path:
            return

        try:
            self.rt_processor = RaytraceProcessor()
            img = self.rt_processor.load_image(file_path)
            self.rt_img_path_label.setText(
                os.path.basename(file_path)
            )
            self.rt_img_path_label.setStyleSheet("color: #00897B;")

            # 缩略图?
            h, w = img.shape[:2]
            scale = min(280 / w, 140 / h)
            new_w, new_h = int(w * scale), int(h * scale)
            if len(img.shape) == 2:
                arr = (img * 255).astype(np.uint8)
                # 使用 .tobytes() 创建数据副本，避免 numpy 数组被 GC 后 QImage 引用悬垂指针
                qimg = QImage(arr.data.tobytes(), w, h, w, QImage.Format_Grayscale8)
            else:
                arr = (img * 255).astype(np.uint8)
                qimg = QImage(arr.data.tobytes(), w, h, 3 * w, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg).scaled(
                new_w, new_h, Qt.KeepAspectRatio)
            self.rt_img_preview.setPixmap(pixmap)

            self._rt_log_msg(f"图像加载成功: {os.path.basename(file_path)} ({w}x{h})")
            self.rt_status_label.setText("图像已加载，可以开始计算")
            self.statusBar().showMessage(
                f"光线追踪: 已加载图像 {os.path.basename(file_path)}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载图像失败: {e}")
            self._rt_log_msg(f"??: {e}")

    # --- Run computation ---

    def _run_rt_step(self, step):
        """执行单步光线追踪计算。"""
        if self.rt_processor is None or self.rt_processor.image_camera1 is None:
            QMessageBox.warning(self, "提示", "请先加载气泡图像")
            return

        kwargs = {}
        if step == 2 or step == 'all':
            kwargs = {
                'theta': self.rt_spin_theta.value(),
                'camera1_angle': self.rt_spin_cam1.value(),
                'lengthtocamera1': self.rt_spin_length.value(),
                'face': 1 if self.rt_spin_face.currentIndex() == 1 else 2,
                'bubble_equivalent_diameter': self.rt_spin_diameter.value(),
            }

        self._set_rt_buttons_enabled(False)
        self.rt_compute_thread = self._RTComputeThread(
            self.rt_processor, step, **kwargs
        )
        self.rt_compute_thread.progress.connect(self._on_rt_progress)
        self.rt_compute_thread.finished.connect(self._on_rt_finished)
        self.rt_compute_thread.error.connect(self._on_rt_error)
        self.rt_compute_thread.start()

    def _run_rt_all(self):
        """运行全部步骤。"""
        if self.rt_processor is None or self.rt_processor.image_camera1 is None:
            QMessageBox.warning(self, "提示", "请先加载气泡图像")
            return
        self._run_rt_step('all')

    def _on_rt_progress(self, percent, msg):
        self.rt_progress_bar.setValue(percent)
        self.rt_status_label.setText(msg)

    def _on_rt_finished(self, msg):
        self._set_rt_buttons_enabled(True)
        self.rt_progress_bar.setValue(100)
        self.rt_status_label.setText("计算完成")
        self._rt_log_msg(msg)
        self.statusBar().showMessage(f"光线追踪: {msg}")
        # Refresh 3D preview automatically
        QTimer.singleShot(300, self._show_rt_3d_view)

    def _on_rt_error(self, err_msg):
        self._set_rt_buttons_enabled(True)
        self.rt_progress_bar.setValue(0)
        self.rt_status_label.setText("计算出错")
        self._rt_log_msg(f"??: {err_msg}")
        QMessageBox.critical(self, "计算错误", err_msg)

    # --- 3D??? ---

    def _show_rt_3d_view(self):
        """更新光线追踪3D视图。"""
        if self.rt_processor is None:
            return

        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D

            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')

            has_content = False

            # Silhouette点云
            if self.rt_processor.bubble_silhouette is not None:
                pts = self.rt_processor.bubble_silhouette
                ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                          c='cyan', s=1, alpha=0.3, label='Silhouette点云')
                has_content = True

            # 正面曲面
            if self.rt_processor.bubble_front_surf is not None:
                X, Y, Z = self.rt_processor.bubble_front_surf
                ax.plot_surface(X, Y, Z, alpha=0.7, cmap='coolwarm',
                              edgecolor='none', label='正面曲面')
                has_content = True

            # 背面点云
            if self.rt_processor.bubble_back_points is not None and \
               len(self.rt_processor.bubble_back_points) > 0:
                back = self.rt_processor.bubble_back_points
                ax.scatter(back[:, 0], back[:, 1], back[:, 2],
                          c='yellow', s=1, alpha=0.3, label='背面点云')
                has_content = True

            # 生成速度场报告
            if self.rt_processor.bubble_all_face_direction is not None and \
               self.rt_processor.bubble_all_position_face_world is not None:
                pos = self.rt_processor.bubble_all_position_face_world[::10, :3]
                normals = self.rt_processor.bubble_all_face_direction[::10]
                if len(pos) > 0 and len(normals) > 0:
                    ax.quiver(pos[:, 0], pos[:, 1], pos[:, 2],
                             normals[:, 0], normals[:, 1], normals[:, 2],
                             length=20, color='lime', alpha=0.5, label='法向量')

            if has_content:
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                ax.legend(fontsize=8)
                ax.view_init(elev=80, azim=10)
            else:
                ax.text2D(0.5, 0.5, "运行计算后显示3D结果",
                         ha='center', va='center', color='gray',
                         transform=ax.transAxes, fontsize=14)

            fig.tight_layout()
            viz_path = os.path.join(
                self.visualizer.output_dir, 'rt_3d_view.png')
            plt.savefig(viz_path, dpi=150, bbox_inches='tight')
            plt.close()

            self._show_image(viz_path)

        except Exception as e:
            self._rt_log_msg(f"3D可视化失败: {e}")

    # --- 综合报告 ---

    def _show_rt_report(self):
        """光线追踪: 综合报告"""
        if self.rt_processor is None:
            QMessageBox.warning(self, "警告", "请先完成光线追踪计算")
            return
        if self.rt_processor.image_camera1 is None:
            return

        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D

            fig = plt.figure(figsize=(16, 10))
            fig.suptitle('Raytrace 3D Reconstruction Report',
                         fontsize=14, fontweight='bold')

            # 1. Original grayscale image
            ax1 = fig.add_subplot(2, 3, 1)
            ax1.imshow(self.rt_processor.image_camera1, cmap='gray')
            ax1.set_title('原始灰度图', fontsize=10)
            ax1.axis('off')

            # 2. Binary result
            ax2 = fig.add_subplot(2, 3, 2)
            if self.rt_processor.imfill_img_camera1 is not None:
                ax2.imshow(self.rt_processor.imfill_img_camera1, cmap='gray')
            else:
                ax2.text(0.5, 0.5, "未运行", ha='center', va='center',
                         color='gray', transform=ax2.transAxes)
            ax2.set_title('二值化+填充', fontsize=10)
            ax2.axis('off')

            # 3. 3D点云/曲面
            ax3 = fig.add_subplot(2, 3, 3, projection='3d')
            if self.rt_processor.bubble_silhouette is not None:
                pts = self.rt_processor.bubble_silhouette
                ax3.scatter(pts[::5, 0], pts[::5, 1], pts[::5, 2],
                           s=0.5, c='cyan', alpha=0.3)
            if self.rt_processor.bubble_front_surf is not None:
                X, Y, Z = self.rt_processor.bubble_front_surf
                ax3.plot_surface(X, Y, Z, alpha=0.6, cmap='coolwarm',
                               edgecolor='none')
            ax3.set_title('3D重建结果', fontsize=10)
            ax3.view_init(elev=80, azim=10)

            # 4. 法向量可视化
            ax4 = fig.add_subplot(2, 3, 4, projection='3d')
            if self.rt_processor.bubble_all_face_direction is not None:
                pos = self.rt_processor.bubble_all_position_face_world[::20, :3]
                normals = self.rt_processor.bubble_all_face_direction[::20]
                if len(pos) > 0 and len(normals) > 0:
                    ax4.quiver(pos[:, 0], pos[:, 1], pos[:, 2],
                             normals[:, 0], normals[:, 1], normals[:, 2],
                             length=20, color='lime', alpha=0.5)
                    ax4.scatter(pos[:, 0], pos[:, 1], pos[:, 2],
                              c='blue', s=1, alpha=0.3)
            ax4.set_title('法向量分布', fontsize=10)
            ax4.view_init(elev=80, azim=10)

            # 5. 前后曲面对比
            ax5 = fig.add_subplot(2, 3, 5, projection='3d')
            if self.rt_processor.bubble_front_points is not None:
                fp = self.rt_processor.bubble_front_points
                ax5.scatter(fp[:, 0], fp[:, 1], fp[:, 2],
                          c='red', s=1, alpha=0.3, label='正面')
            if self.rt_processor.bubble_back_points is not None:
                bp = self.rt_processor.bubble_back_points
                ax5.scatter(bp[:, 0], bp[:, 1], bp[:, 2],
                          c='blue', s=1, alpha=0.3, label='背面')
            ax5.set_title('前后表面', fontsize=10)
            ax5.legend(fontsize=8)
            ax5.view_init(elev=80, azim=10)

            # 6. Statistics
            ax6 = fig.add_subplot(2, 3, 6)
            ax6.axis('off')

            info_lines = ["重建参数:\n"]
            info_lines.append(f"  ? = {self.rt_spin_theta.value()}?")
            info_lines.append(f"  相机1角度 = {self.rt_spin_cam1.value()}°")
            info_lines.append(f"  物距 = {self.rt_spin_length.value()} px")
            info_lines.append(f"  气泡直径 = {self.rt_spin_diameter.value()} px")
            info_lines.append(f"  ??? ? = {self.rt_spin_miu2.value()}")
            info_lines.append(f"  重建面 = {'双面' if self.rt_spin_face.currentIndex() == 0 else '仅前面'}")

            if self.rt_processor.bubble_silhouette is not None:
                info_lines.append(f"\n结果统计:")
                info_lines.append(f"  Silhouette点数: {len(self.rt_processor.bubble_silhouette)}")
                if self.rt_processor.tempab is not None:
                    info_lines.append(f"  ???: {len(self.rt_processor.tempab)}")
                if self.rt_processor.bubble_front_points is not None:
                    info_lines.append(f"  正面点数: {len(self.rt_processor.bubble_front_points)}")
                if self.rt_processor.bubble_back_points is not None:
                    info_lines.append(f"  背面点数: {len(self.rt_processor.bubble_back_points)}")

            info_text = "\n".join(info_lines)
            ax6.text(0.05, 0.95, info_text, transform=ax6.transAxes,
                     fontsize=10, verticalalignment='top',
                     fontfamily='monospace',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            plt.tight_layout()
            report_path = os.path.join(
                self.visualizer.output_dir, 'rt_report.png')
            plt.savefig(report_path, dpi=150, bbox_inches='tight')
            plt.close()

            self._show_image(report_path)

        except Exception as e:
            self._rt_log_msg(f"报告生成失败: {e}")
            QMessageBox.critical(self, "错误", str(e))

    # --- 导出 ---

    def _export_rt_points(self):
        """导出点云CSV"""
        if self.rt_processor is None or \
           self.rt_processor.bubble_all_position_face_world is None:
            QMessageBox.warning(self, "提示", "请先完成计算")
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self, "导出点云", "", "CSV文件 (*.csv);;所有文件 (*)"
        )
        if not filepath:
            return

        data = self.rt_processor.bubble_all_position_face_world
        header = "X, Y, Z, nx, ny, nz"
        np.savetxt(filepath, data, delimiter=',', header=header, comments='')
        self._rt_log_msg(f"点云已导出: {filepath}")
        QMessageBox.information(self, "完成", f"点云已导出到:\n{filepath}")

    def _export_rt_figure(self):
        """导出3D图PNG"""
        if self.rt_processor is None or \
           self.rt_processor.bubble_front_surf is None:
            QMessageBox.warning(self, "提示", "请先完成计算")
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self, "导出3D图", "", "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)"
        )
        if not filepath:
            return

        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D

            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')

            if self.rt_processor.bubble_silhouette is not None:
                pts = self.rt_processor.bubble_silhouette
                ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                          c='cyan', s=1, alpha=0.3)

            if self.rt_processor.bubble_front_surf is not None:
                X, Y, Z = self.rt_processor.bubble_front_surf
                ax.plot_surface(X, Y, Z, alpha=0.7, cmap='coolwarm',
                              edgecolor='none')

            if self.rt_processor.bubble_back_points is not None:
                back = self.rt_processor.bubble_back_points
                ax.scatter(back[:, 0], back[:, 1], back[:, 2],
                          c='yellow', s=1, alpha=0.3)

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            plt.savefig(filepath, dpi=300, facecolor='white',
                       bbox_inches='tight')
            plt.close()

            self._rt_log_msg(f"3D图已导出: {filepath}")
            QMessageBox.information(self, "完成", f"3D图已导出:\n{filepath}")
        except Exception as e:
            QMessageBox.critical(self, "错误", str(e))

    # ---- 粒子追踪 ----

    def _load_particle_images_batch(self):
        """批量加载粒子图像序列"""
        if not self.calibration_results:
            QMessageBox.warning(self, "警告", "请先完成相机标定")
            return
        calib_count = len(self.calibration_results)
        selected_count = self.particle_camera_count_spin.value()
        if selected_count > calib_count:
            QMessageBox.warning(
                self,
                "警告",
                f"当前标定了 {calib_count} 个相机，不能选择 {selected_count} 个相机"
            )
            self.particle_camera_count_spin.setValue(calib_count)
            return

        self.particle_active_camera_ids = self._get_particle_camera_ids()
        for cam_id in self.particle_active_camera_ids:
            self._load_particle_sequence_for_camera(cam_id)

        self.statusBar().showMessage("粒子图像序列加载完成")

    def _setup_piv_timepoint_selector(self):
        """初始化 PIV 的时间点选择器。"""
        n = len(self.particle_timepoint_images)
        if n == 0:
            return

        self.piv_tp_combo.blockSignals(True)
        self.piv_tp_slider.blockSignals(True)

        self.piv_tp_combo.clear()
        for idx in sorted(self.particle_timepoint_images.keys()):
            name = self.particle_timepoint_names.get(idx, f"t{idx}")
            self.piv_tp_combo.addItem(f"{name} (t{idx})", userData=idx)

        self.piv_tp_slider.setRange(0, n - 1)
        self.piv_tp_slider.setValue(0)

        self.piv_tp_combo.setEnabled(True)
        self.piv_tp_slider.setEnabled(True)
        self.piv_tp_prev_btn.setEnabled(n > 1)
        self.piv_tp_next_btn.setEnabled(n > 1)

        self._update_piv_tp_label(0)
        self._update_piv_pair_label()

        self.piv_tp_combo.blockSignals(False)
        self.piv_tp_slider.blockSignals(False)

        self.piv_tp_info_label.setText(
            f"共 {n} 个时间点 | 预测时间对: t0"
        )
        self.piv_tp_info_label.setStyleSheet("color: #E65100; font-weight: bold;")

    def _update_piv_tp_label(self, idx):
        n = len(self.particle_timepoint_images)
        name = self.particle_timepoint_names.get(idx, f"t{idx}")
        self.piv_tp_label.setText(f"{name}  ({idx + 1} / {n})")

    def _update_piv_pair_label(self, idx=None):
        """更新帧对显示。"""
        idx1 = self.piv_frame1_combo.currentData()
        idx2 = self.piv_frame2_combo.currentData()
        if idx1 is None or idx2 is None:
            self.piv_pair_label.setText("帧对: -- -> --")
            return
        n1 = self.particle_timepoint_names.get(idx1, f"t{idx1}")
        n2 = self.particle_timepoint_names.get(idx2, f"t{idx2}")
        self.piv_pair_label.setText(
            f"帧对: {n1} -> {n2}  (t{idx1} -> t{idx2})"
        )

    def _set_current_piv_timepoint(self, idx):
        """设置当前 PIV 时间点。"""
        if idx not in self.particle_timepoint_images:
            return
        self.current_piv_timepoint = idx
        self._preview_particle_timepoint(idx)
        self._update_piv_pair_label()

    def _preview_particle_timepoint(self, idx):
        """预览指定时间点的粒子图像。"""
        import cv2
        if idx not in self.particle_timepoint_images:
            return
        cam_imgs = self.particle_timepoint_images[idx]
        if not cam_imgs:
            return

        imgs = list(cam_imgs.values())
        if not imgs:
            return

        target_h = 300
        resized = []
        for img in imgs:
            h, w = img.shape[:2]
            scale = target_h / h
            new_w = int(w * scale)
            resized.append(cv2.resize(img, (new_w, target_h)))

        combined = np.hstack(resized)
        name = self.particle_timepoint_names.get(idx, f"t{idx}")
        cv2.putText(combined, f"Particle: {name}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 165, 0), 2)

        preview_path = os.path.join(
            self.visualizer.output_dir, f'particle_preview_t{idx}.png'
        )
        cv2.imwrite(preview_path, combined)
        pixmap = QPixmap(preview_path)
        self.piv_preview.setPixmap(pixmap.scaled(
            self.piv_preview.size(), Qt.KeepAspectRatio,
            Qt.SmoothTransformation))

    def _on_piv_tp_changed(self, idx):
        if idx < 0:
            return
        self.piv_tp_combo.blockSignals(True)
        self.piv_tp_combo.setCurrentIndex(idx)
        self.piv_tp_combo.blockSignals(False)
        self._update_piv_tp_label(idx)
        self._set_current_piv_timepoint(idx)

    def _on_piv_tp_combo_changed(self, idx):
        if idx < 0:
            return
        self.piv_tp_slider.blockSignals(True)
        self.piv_tp_slider.setValue(idx)
        self.piv_tp_slider.blockSignals(False)
        self._update_piv_tp_label(idx)
        self._set_current_piv_timepoint(idx)

    def _piv_tp_prev(self):
        idx = self.piv_tp_slider.value()
        if idx > 0:
            self.piv_tp_slider.setValue(idx - 1)

    def _piv_tp_next(self):
        idx = self.piv_tp_slider.value()
        n = len(self.particle_timepoint_images)
        if idx < n - 1:
            self.piv_tp_slider.setValue(idx + 1)

    def _load_particle_frame1(self):
        if not self.calibration_results:
            QMessageBox.warning(self, "警告", "请先完成相机标定")
            return
        self._load_particle_images(self.particle_images_frame1, 1)

    def _load_particle_frame2(self):
        if not self.calibration_results:
            QMessageBox.warning(self, "警告", "请先完成相机标定")
            return
        self._load_particle_images(self.particle_images_frame2, 2)

    def _load_particle_images(self, target_dict: dict, frame_num: int):
        import cv2
        cam_ids = self.particle_active_camera_ids or self._get_particle_camera_ids()
        target_dict.clear()
        for cam_id in cam_ids:
            file_path, _ = QFileDialog.getOpenFileName(
                self, f"相机 {cam_id} - 第{frame_num}帧粒子图像", "",
                "图像文件 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
            )
            if file_path:
                img = cv2.imread(file_path, cv2.IMREAD_COLOR)
                if img is not None:
                    target_dict[cam_id] = img
        if target_dict:
            self.particle_status.setText(
                f"帧 {frame_num}: 已加载 {len(target_dict)} 个相机图像"
            )
            self.particle_status.setStyleSheet("color: green;")
            self.statusBar().showMessage(f"第{frame_num}帧粒子图像已加载")
            self._refresh_particle_camera_previews()
            self._update_piv_pair_label()

    def _run_particle_reconstruction(self):
        if not self.calibration_results:
            QMessageBox.warning(self, "警告", "请先完成标定")
            return
        if not self.particle_images_frame1:
            QMessageBox.warning(self, "警告", "请先加载两帧粒子图像")
            return

        config = TriangulationConfig(
            blob_min_area=self.p_min_area.value(),
            blob_max_area=self.p_max_area.value(),
            circularity_threshold=self.p_circularity.value(),
            epipolar_threshold=self.p_epipolar.value()
        )

        reconstructor = Particle3DReconstructor(config)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.statusBar().showMessage("粒子3D重建中...")

        try:
            self.particles_3d_frame1 = reconstructor.reconstruct_particles(
                self.particle_images_frame1, self.calibrator
            )
            self.piv_log.append(
                f"第一帧重建 {len(self.particles_3d_frame1)} 个粒子"
            )

            if self.particle_images_frame2:
                self.particles_3d_frame2 = reconstructor.reconstruct_particles(
                    self.particle_images_frame2, self.calibrator
                )
                self.piv_log.append(
                    f"第二帧重建 {len(self.particles_3d_frame2)} 个粒子"
                )

            # 显示粒子三维点云预览
            if self.particles_3d_frame1:
                self.visualizer.plot_particle_positions_3d(
                    self.particles_3d_frame1,
                    save_path=os.path.join(self.visualizer.output_dir,
                                            'particles_3d.png')
                )
                pixmap = QPixmap(os.path.join(
                    self.visualizer.output_dir, 'particles_3d.png'))
                self.piv_preview.setPixmap(pixmap.scaled(
                    self.piv_preview.size(), Qt.KeepAspectRatio,
                    Qt.SmoothTransformation))

            QMessageBox.information(self, "完成",
                f"粒子重建完成!\n"
                f"第一帧: {len(self.particles_3d_frame1)} 个粒子\n"
                f"第二帧: {len(self.particles_3d_frame2)} 个粒子")
        except Exception as e:
            QMessageBox.critical(self, "错误", str(e))
            self.piv_log.append(f"错误: {e}")
        finally:
            self.progress_bar.setVisible(False)

    def _run_piv_batch(self):
        """计算 PIV 速度场。"""
        if not self.calibration_results:
            QMessageBox.warning(self, "警告", "请先完成标定")
            return
        if not self.particle_timepoint_images:
            QMessageBox.warning(self, "警告", "请先批量加载粒子图像序列")
            return

        sorted_keys = sorted(self.particle_timepoint_images.keys())
        total_pairs = len(sorted_keys) - 1
        if total_pairs < 1:
            QMessageBox.warning(self, "警告", "至少需要 2 个时间点才能计算速度场")
            return

        dt = self.piv_dt.value()
        interrog_size = (
            self.piv_interrog.value(),
            self.piv_interrog.value(),
            self.piv_interrog.value()
        )

        triang_config = TriangulationConfig(
            blob_min_area=self.p_min_area.value(),
            blob_max_area=self.p_max_area.value(),
            circularity_threshold=self.p_circularity.value(),
            epipolar_threshold=self.p_epipolar.value()
        )

        vel_config = CorrelationConfig(
            interrogation_size=interrog_size,
            overlap_ratio=self.piv_overlap.value(),
            peak_threshold=self.piv_snr.value()
        )

        domain_size = (20, 20, 20)
        try:
            domain_size = self.calibrator.domain_size if hasattr(self.calibrator, 'domain_size') else (20, 20, 20)
        except Exception:
            pass

        self.piv_batch_results.clear()
        self.piv_batch_progress.setVisible(True)
        self.piv_batch_progress.setRange(0, total_pairs)
        self.piv_batch_progress.setValue(0)

        self.piv_batch_worker = BatchPIVWorker(
            particle_sequence=self.particle_timepoint_images,
            calibrator=self.calibrator,
            triang_config=triang_config,
            vel_config=vel_config,
            dt=dt,
            domain_size=domain_size
        )
        self.piv_batch_worker.progress.connect(self.piv_log.append)
        self.piv_batch_worker.timepoint_done.connect(
            self._on_piv_batch_tp_done)
        self.piv_batch_worker.all_done.connect(self._on_piv_batch_all_done)
        self.piv_batch_worker.error.connect(self._on_piv_batch_error)
        self.piv_batch_worker.start()

        self.piv_log.append(
            f"=== 开始批量PIV处理 ===\n共 {total_pairs} 组"
        )
        self.statusBar().showMessage("批量PIV处理中...")

    def _on_piv_batch_tp_done(self, tp_idx, result):
        """单帧对PIV处理完成"""
        self.piv_batch_results[tp_idx] = result
        name = self.particle_timepoint_names.get(tp_idx, f"t{tp_idx}")
        sorted_keys = sorted(self.particle_timepoint_images.keys())
        if tp_idx in sorted_keys:
            pos = sorted_keys.index(tp_idx)
            if pos < len(sorted_keys) - 1:
                next_idx = sorted_keys[pos + 1]
                n2 = self.particle_timepoint_names.get(next_idx, f"t{next_idx}")
                vf = result['velocity_result']['velocity_field']
                speed = np.linalg.norm(vf, axis=-1)
                self.piv_log.append(
                    f"  帧对 {name} -> {n2} | "
                    f"粒子: {len(result['particles_3d_frame1'])} / "
                    f"{len(result['particles_3d_frame2'])} | "
                    f"平均速度: {speed.mean():.2f} mm/s"
                )
        self.piv_batch_progress.setValue(len(self.piv_batch_results))

    def _on_piv_batch_all_done(self, all_results):
        """批量PIV全部完成"""
        self.piv_batch_progress.setVisible(False)
        self.progress_bar.setVisible(False)
        total = len(all_results)
        self.piv_log.append(
            f"\n=== 批量PIV处理完成 ===\n"
            f"成功处理 {total} 组\n"
            f"请使用底部时间点滑块切换查看各时刻结果"
        )
        self.statusBar().showMessage(f"批量PIV完成: {total} 组")
        QMessageBox.information(self, "完成",
            f"批量PIV处理完成!\n成功处理 {total} 组\n"
            f"请使用底部时间点滑块切换查看结果")

        # Switch to particle PIV page
        if all_results:
            first_idx = sorted(all_results.keys())[0]
            self._show_piv_result_for_timepoint(first_idx)

    def _on_piv_batch_error(self, msg):
        self.piv_batch_progress.setVisible(False)
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "PIV错误", msg)

    def _show_piv_result_for_timepoint(self, tp_idx):
        """显示指定时间点的PIV结果"""
        if tp_idx not in self.piv_batch_results:
            return
        try:
            result = self.piv_batch_results[tp_idx]
            name = self.particle_timepoint_names.get(tp_idx, f"t{tp_idx}")
            self.visualizer.create_velocity_report(
                result['velocity_result'],
                result['particles_3d_frame1'],
                save_path=os.path.join(self.visualizer.output_dir,
                                        f'piv_report_t{tp_idx}.png')
            )
            pixmap = QPixmap(os.path.join(
                self.visualizer.output_dir, f'piv_report_t{tp_idx}.png'))
            self.piv_preview.setPixmap(pixmap.scaled(
                self.piv_preview.size(), Qt.KeepAspectRatio,
                Qt.SmoothTransformation))
            self.visualizer.save_velocity_field_vtk(
                result['velocity_result']['velocity_field'],
                result['velocity_result']['grid_positions'],
                save_path=os.path.join(self.visualizer.output_dir,
                                       f'velocity_t{tp_idx}.vtk')
            )
        except Exception as e:
            self.piv_log.append(f"显示结果失败: {e}")

    def _run_velocity_computation(self):
        if not self.particles_3d_frame1 or not self.particles_3d_frame2:
            QMessageBox.warning(self, "警告",
                "请先完成两帧的粒子3D重建")
            return

        dt = self.piv_dt.value()
        interrog_size = (self.piv_interrog.value(),
                         self.piv_interrog.value(),
                         self.piv_interrog.value())

        config = CorrelationConfig(
            interrogation_size=interrog_size,
            overlap_ratio=self.piv_overlap.value(),
            peak_threshold=self.piv_snr.value()
        )

        domain_size = self.calibrator.config_data if hasattr(self.calibrator, 'config_data') else (20, 20, 20)
        # Use reconstruction domain size from calibrator
        try:
            domain_size = self.calibrator.domain_size if hasattr(self.calibrator, 'domain_size') else (20, 20, 20)
        except:
            domain_size = (20, 20, 20)

        calculator = VelocityFieldCalculator(
            config=config, domain_size=domain_size, dt=dt
        )

        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.piv_log.append("速度场正在计算...")

        try:
            self._velocity_result = calculator.compute_velocity_field(
                self.particles_3d_frame1,
                self.particles_3d_frame2
            )

            self.piv_log.append("速度场计算完成")
            vf = self._velocity_result['velocity_field']
            speed = np.linalg.norm(vf, axis=-1)
            self.piv_log.append(f"网格: {vf.shape[:3]}")
            self.piv_log.append(f"平均速度: {speed.mean():.2f} mm/s")
            self.piv_log.append(f"最大速度: {speed.max():.2f} mm/s")

            # 生成速度场报告
            self.visualizer.create_velocity_report(
                self._velocity_result,
                self.particles_3d_frame1,
                save_path=os.path.join(self.visualizer.output_dir,
                                        'piv_report.png')
            )
            pixmap = QPixmap(os.path.join(
                self.visualizer.output_dir, 'piv_report.png'))
            self.piv_preview.setPixmap(pixmap.scaled(
                self.piv_preview.size(), Qt.KeepAspectRatio,
                Qt.SmoothTransformation))

            # 导出VTK
            self.visualizer.save_velocity_field_vtk(
                self._velocity_result['velocity_field'],
                self._velocity_result['grid_positions']
            )

            QMessageBox.information(self, "完成",
                f"速度场计算完成\n"
                f"平均速度: {speed.mean():.2f} mm/s\n"
                f"最大速度: {speed.max():.2f} mm/s\n"
                f"VTK文件已保存")

        except Exception as e:
            QMessageBox.critical(self, "错误", str(e))
            self.piv_log.append(f"错误: {e}")
        finally:
            self.progress_bar.setVisible(False)

    def _export_point_cloud(self, fmt):
        if not hasattr(self, '_last_results'):
            QMessageBox.warning(self, "警告", "请先完成重建")
            return

        points = self._last_results['points']
        normals = self._last_results.get('normals')

        if fmt == 'ply':
            path = self.visualizer.save_point_cloud_ply(points, normals)
        elif fmt == 'pcd':
            path = self.visualizer.save_point_cloud_pcd(points)
        else:
            return

        QMessageBox.information(self, "完成", f"点云已导出至:\n{path}")

    # ------------------------------------------------------------
    #  二维PIV页面（Page 4）
    # ------------------------------------------------------------

    def _create_piv2d_page(self):
        page = QWidget()
        layout = QHBoxLayout(page)

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFixedWidth(self._sp(380))
        left_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(self._sp(6), self._sp(6), self._sp(6), self._sp(6))
        left_layout.setSpacing(self._sp(6))

        mode_group = QGroupBox("二维PIV输入")
        mode_layout = QGridLayout(mode_group)
        mode_layout.addWidget(QLabel("处理模式:"), 0, 0)
        self.piv2d_mode_combo = QComboBox()
        self.piv2d_mode_combo.addItems(["单组图像", "批量目录"])
        self.piv2d_mode_combo.currentIndexChanged.connect(self._piv2d_on_mode_changed)
        mode_layout.addWidget(self.piv2d_mode_combo, 0, 1)

        self.piv2d_single_widget = QWidget()
        single_layout = QGridLayout(self.piv2d_single_widget)
        btn_f1 = QPushButton("选择第1帧...")
        btn_f1.clicked.connect(self._piv2d_open_frame1)
        single_layout.addWidget(btn_f1, 0, 0)
        self.piv2d_frame1_label = QLabel("未选择")
        self.piv2d_frame1_label.setWordWrap(True)
        self.piv2d_frame1_label.setStyleSheet("color: gray;")
        single_layout.addWidget(self.piv2d_frame1_label, 1, 0)
        btn_f2 = QPushButton("选择第2帧...")
        btn_f2.clicked.connect(self._piv2d_open_frame2)
        single_layout.addWidget(btn_f2, 2, 0)
        self.piv2d_frame2_label = QLabel("未选择")
        self.piv2d_frame2_label.setWordWrap(True)
        self.piv2d_frame2_label.setStyleSheet("color: gray;")
        single_layout.addWidget(self.piv2d_frame2_label, 3, 0)
        mode_layout.addWidget(self.piv2d_single_widget, 1, 0, 1, 2)

        self.piv2d_batch_widget = QWidget()
        batch_layout = QGridLayout(self.piv2d_batch_widget)
        btn_src = QPushButton("输入目录...")
        btn_src.clicked.connect(self._piv2d_pick_src_dir)
        batch_layout.addWidget(btn_src, 0, 0)
        self.piv2d_src_label = QLabel("未选择")
        self.piv2d_src_label.setWordWrap(True)
        self.piv2d_src_label.setStyleSheet("color: gray;")
        batch_layout.addWidget(self.piv2d_src_label, 1, 0)
        btn_dst = QPushButton("输出目录...")
        btn_dst.clicked.connect(self._piv2d_pick_dst_dir)
        batch_layout.addWidget(btn_dst, 2, 0)
        self.piv2d_dst_label = QLabel("未选择")
        self.piv2d_dst_label.setWordWrap(True)
        self.piv2d_dst_label.setStyleSheet("color: gray;")
        batch_layout.addWidget(self.piv2d_dst_label, 3, 0)
        self.piv2d_batch_widget.hide()
        mode_layout.addWidget(self.piv2d_batch_widget, 1, 0, 1, 2)

        timeline_hint = QLabel(
            "如果输入目录包含时间组子文件夹，右侧时间轴可用于挑选不同时间组的图片组成双帧。"
        )
        timeline_hint.setWordWrap(True)
        timeline_hint.setStyleSheet("color: gray; font-size: 11px;")
        mode_layout.addWidget(timeline_hint, 2, 0, 1, 2)
        left_layout.addWidget(mode_group)

        batch_pair_group = QGroupBox("批量帧对")
        batch_pair_layout = QGridLayout(batch_pair_group)
        batch_pair_layout.addWidget(QLabel("组合方式:"), 0, 0)
        self.piv2d_batch_pair_mode_combo = QComboBox()
        self.piv2d_batch_pair_mode_combo.addItems([
            "连续相邻 (1-2, 2-3, 3-4)",
            "两两分组 (1-2, 3-4, 5-6)",
            "自定义帧对",
        ])
        self.piv2d_batch_pair_mode_combo.currentIndexChanged.connect(self._piv2d_on_batch_pair_mode_changed)
        batch_pair_layout.addWidget(self.piv2d_batch_pair_mode_combo, 0, 1)
        batch_pair_layout.addWidget(QLabel("自定义:"), 1, 0)
        self.piv2d_batch_pairs_edit = QLineEdit()
        self.piv2d_batch_pairs_edit.setPlaceholderText("例如: 1-2, 3-4, 5-6 或 1,2,3,4")
        self.piv2d_batch_pairs_edit.setEnabled(False)
        batch_pair_layout.addWidget(self.piv2d_batch_pairs_edit, 1, 1)
        self.piv2d_batch_pair_hint = QLabel("编号按输入目录中排序后的图像序号，从 1 开始。")
        self.piv2d_batch_pair_hint.setWordWrap(True)
        self.piv2d_batch_pair_hint.setStyleSheet("color: gray; font-size: 11px;")
        batch_pair_layout.addWidget(self.piv2d_batch_pair_hint, 2, 0, 1, 2)
        left_layout.addWidget(batch_pair_group)

        cfg_group = QGroupBox("互相关参数")
        cfg_layout = QGridLayout(cfg_group)
        cfg_layout.addWidget(QLabel("窗口尺寸:"), 0, 0)
        self.piv2d_win_spin = QSpinBox()
        self.piv2d_win_spin.setRange(8, 256)
        self.piv2d_win_spin.setSingleStep(4)
        self.piv2d_win_spin.setValue(32)
        self.piv2d_win_spin.valueChanged.connect(self._piv2d_sync_correlation_window_size)
        cfg_layout.addWidget(self.piv2d_win_spin, 0, 1)
        cfg_layout.addWidget(QLabel("重叠率:"), 1, 0)
        self.piv2d_overlap_spin = QDoubleSpinBox()
        self.piv2d_overlap_spin.setRange(0.0, 0.9)
        self.piv2d_overlap_spin.setSingleStep(0.1)
        self.piv2d_overlap_spin.setValue(0.5)
        cfg_layout.addWidget(self.piv2d_overlap_spin, 1, 1)
        cfg_layout.addWidget(QLabel("搜索半径(px):"), 2, 0)
        self.piv2d_search_spin = QSpinBox()
        self.piv2d_search_spin.setRange(2, 128)
        self.piv2d_search_spin.setValue(16)
        cfg_layout.addWidget(self.piv2d_search_spin, 2, 1)
        cfg_layout.addWidget(QLabel("时间间隔 dt:"), 3, 0)
        self.piv2d_dt_spin = QDoubleSpinBox()
        self.piv2d_dt_spin.setRange(1e-6, 1e6)
        self.piv2d_dt_spin.setDecimals(6)
        self.piv2d_dt_spin.setValue(1.0)
        cfg_layout.addWidget(self.piv2d_dt_spin, 3, 1)
        cfg_layout.addWidget(QLabel("像素尺度:"), 4, 0)
        self.piv2d_scale_spin = QDoubleSpinBox()
        self.piv2d_scale_spin.setRange(1e-6, 1e6)
        self.piv2d_scale_spin.setDecimals(6)
        self.piv2d_scale_spin.setValue(1.0)
        cfg_layout.addWidget(self.piv2d_scale_spin, 4, 1)
        cfg_layout.addWidget(QLabel("SNR阈值:"), 5, 0)
        self.piv2d_snr_spin = QDoubleSpinBox()
        self.piv2d_snr_spin.setRange(0.1, 50.0)
        self.piv2d_snr_spin.setSingleStep(0.1)
        self.piv2d_snr_spin.setValue(1.2)
        cfg_layout.addWidget(self.piv2d_snr_spin, 5, 1)
        cfg_layout.addWidget(QLabel("最大位移:"), 6, 0)
        self.piv2d_maxdisp_spin = QDoubleSpinBox()
        self.piv2d_maxdisp_spin.setRange(1.0, 500.0)
        self.piv2d_maxdisp_spin.setValue(32.0)
        cfg_layout.addWidget(self.piv2d_maxdisp_spin, 6, 1)
        self.piv2d_adaptive_check = QCheckBox("自适应多重网格 / 窗口变形")
        self.piv2d_adaptive_check.toggled.connect(self._piv2d_on_adaptive_toggled)
        cfg_layout.addWidget(self.piv2d_adaptive_check, 7, 0, 1, 2)
        cfg_layout.addWidget(QLabel("窗口序列:"), 8, 0)
        self.piv2d_adaptive_windows_edit = QLineEdit("64,32,16")
        self.piv2d_adaptive_windows_edit.setEnabled(False)
        cfg_layout.addWidget(self.piv2d_adaptive_windows_edit, 8, 1)
        cfg_layout.addWidget(QLabel("残差搜索半径:"), 9, 0)
        self.piv2d_adaptive_residual_spin = QSpinBox()
        self.piv2d_adaptive_residual_spin.setRange(1, 64)
        self.piv2d_adaptive_residual_spin.setValue(6)
        self.piv2d_adaptive_residual_spin.setEnabled(False)
        cfg_layout.addWidget(self.piv2d_adaptive_residual_spin, 9, 1)
        self.piv2d_flow_check = QCheckBox("光流像素级细化")
        self.piv2d_flow_check.toggled.connect(self._piv2d_on_flow_toggled)
        cfg_layout.addWidget(self.piv2d_flow_check, 10, 0, 1, 2)
        cfg_layout.addWidget(QLabel("光流层数:"), 11, 0)
        self.piv2d_flow_levels_spin = QSpinBox()
        self.piv2d_flow_levels_spin.setRange(1, 8)
        self.piv2d_flow_levels_spin.setValue(3)
        self.piv2d_flow_levels_spin.setEnabled(False)
        cfg_layout.addWidget(self.piv2d_flow_levels_spin, 11, 1)
        cfg_layout.addWidget(QLabel("光流窗口:"), 12, 0)
        self.piv2d_flow_winsize_spin = QSpinBox()
        self.piv2d_flow_winsize_spin.setRange(5, 99)
        self.piv2d_flow_winsize_spin.setSingleStep(2)
        self.piv2d_flow_winsize_spin.setValue(15)
        self.piv2d_flow_winsize_spin.setEnabled(False)
        cfg_layout.addWidget(self.piv2d_flow_winsize_spin, 12, 1)
        cfg_layout.addWidget(QLabel("光流迭代:"), 13, 0)
        self.piv2d_flow_iterations_spin = QSpinBox()
        self.piv2d_flow_iterations_spin.setRange(1, 20)
        self.piv2d_flow_iterations_spin.setValue(3)
        self.piv2d_flow_iterations_spin.setEnabled(False)
        cfg_layout.addWidget(self.piv2d_flow_iterations_spin, 13, 1)
        self.piv2d_outlier_check = QCheckBox("剔除不合理矢量")
        self.piv2d_outlier_check.toggled.connect(self._piv2d_on_outlier_toggled)
        cfg_layout.addWidget(self.piv2d_outlier_check, 14, 0, 1, 2)
        self.piv2d_replace_outlier_check = QCheckBox("对剔除矢量进行插值替换")
        self.piv2d_replace_outlier_check.setEnabled(False)
        cfg_layout.addWidget(self.piv2d_replace_outlier_check, 15, 0, 1, 2)
        cfg_layout.addWidget(QLabel("局部残差阈值:"), 16, 0)
        self.piv2d_outlier_median_spin = QDoubleSpinBox()
        self.piv2d_outlier_median_spin.setRange(0.0, 1e6)
        self.piv2d_outlier_median_spin.setDecimals(3)
        self.piv2d_outlier_median_spin.setValue(3.0)
        self.piv2d_outlier_median_spin.setEnabled(False)
        cfg_layout.addWidget(self.piv2d_outlier_median_spin, 16, 1)
        cfg_layout.addWidget(QLabel("速度上限:"), 17, 0)
        self.piv2d_outlier_speed_spin = QDoubleSpinBox()
        self.piv2d_outlier_speed_spin.setRange(0.0, 1e9)
        self.piv2d_outlier_speed_spin.setDecimals(3)
        self.piv2d_outlier_speed_spin.setValue(0.0)
        self.piv2d_outlier_speed_spin.setEnabled(False)
        cfg_layout.addWidget(self.piv2d_outlier_speed_spin, 17, 1)
        cfg_layout.addWidget(QLabel("插值迭代:"), 18, 0)
        self.piv2d_outlier_interp_spin = QSpinBox()
        self.piv2d_outlier_interp_spin.setRange(1, 50)
        self.piv2d_outlier_interp_spin.setValue(5)
        self.piv2d_outlier_interp_spin.setEnabled(False)
        cfg_layout.addWidget(self.piv2d_outlier_interp_spin, 18, 1)
        left_layout.addWidget(cfg_group)

        viz_group = QGroupBox("矢量显示")
        viz_layout = QGridLayout(viz_group)
        viz_layout.addWidget(QLabel("长度倍率:"), 0, 0)
        self.piv2d_vector_scale_spin = QDoubleSpinBox()
        self.piv2d_vector_scale_spin.setRange(0.01, 20.0)
        self.piv2d_vector_scale_spin.setSingleStep(0.05)
        self.piv2d_vector_scale_spin.setValue(0.15)
        self.piv2d_vector_scale_spin.valueChanged.connect(self._piv2d_refresh_vector_preview)
        viz_layout.addWidget(self.piv2d_vector_scale_spin, 0, 1)
        viz_layout.addWidget(QLabel("颜色组:"), 1, 0)
        self.piv2d_vector_color_combo = QComboBox()
        self.piv2d_vector_color_combo.addItems(["按速度彩色", "红色", "绿色", "蓝色", "黄色", "白色"])
        self.piv2d_vector_color_combo.currentIndexChanged.connect(self._piv2d_refresh_vector_preview)
        viz_layout.addWidget(self.piv2d_vector_color_combo, 1, 1)
        left_layout.addWidget(viz_group)

        mask_group = QGroupBox("无粒子区域")
        mask_layout = QVBoxLayout(mask_group)
        shape_row = QHBoxLayout()
        shape_row.addWidget(QLabel("几何形状:"))
        self.piv2d_mask_shape_combo = QComboBox()
        self.piv2d_mask_shape_combo.addItems(["矩形", "圆形", "三角形", "多边形"])
        shape_row.addWidget(self.piv2d_mask_shape_combo, stretch=1)
        mask_layout.addLayout(shape_row)
        self.piv2d_mask_info_label = QLabel("未设置")
        self.piv2d_mask_info_label.setWordWrap(True)
        self.piv2d_mask_info_label.setStyleSheet("color: gray;")
        mask_layout.addWidget(self.piv2d_mask_info_label)
        btn_add_mask = QPushButton("添加无粒子区域")
        btn_add_mask.clicked.connect(self._piv2d_begin_mask_selection)
        mask_layout.addWidget(btn_add_mask)
        btn_clear_mask = QPushButton("清除无粒子区域")
        btn_clear_mask.clicked.connect(self._piv2d_clear_mask)
        mask_layout.addWidget(btn_clear_mask)
        left_layout.addWidget(mask_group)

        action_group = QGroupBox("执行")
        action_layout = QVBoxLayout(action_group)
        btn_single = QPushButton("运行单组二维PIV")
        btn_single.clicked.connect(self._piv2d_run_single)
        action_layout.addWidget(btn_single)
        self.piv2d_batch_run_btn = QPushButton("批量运行二维PIV")
        self.piv2d_batch_run_btn.clicked.connect(self._piv2d_run_batch)
        action_layout.addWidget(self.piv2d_batch_run_btn)
        self.piv2d_stop_btn = QPushButton("停止批量")
        self.piv2d_stop_btn.setEnabled(False)
        self.piv2d_stop_btn.clicked.connect(self._piv2d_stop_batch)
        action_layout.addWidget(self.piv2d_stop_btn)
        self.piv2d_progress = QProgressBar()
        self.piv2d_progress.setVisible(False)
        action_layout.addWidget(self.piv2d_progress)
        left_layout.addWidget(action_group)
        left_layout.addStretch()
        left_scroll.setWidget(left_panel)

        right_panel = QWidget()
        right_layout = QHBoxLayout(right_panel)
        preview_splitter = QSplitter(Qt.Vertical)

        top_widget = QWidget()
        top_layout = QHBoxLayout(top_widget)
        self.piv2d_frame1_preview = PIV2DPreviewWidget("第1帧预览")
        self.piv2d_frame1_preview.setMinimumSize(self._sp(260), self._sp(220))
        self.piv2d_frame1_preview.mask_changed.connect(self._piv2d_on_mask_changed)
        self.piv2d_frame1_preview.set_correlation_window_size(self.piv2d_win_spin.value())
        top_layout.addWidget(self.piv2d_frame1_preview)
        self.piv2d_frame2_preview = PIV2DPreviewWidget("第2帧预览")
        self.piv2d_frame2_preview.setMinimumSize(self._sp(260), self._sp(220))
        self.piv2d_frame2_preview.set_correlation_window_size(self.piv2d_win_spin.value())
        top_layout.addWidget(self.piv2d_frame2_preview)
        preview_splitter.addWidget(top_widget)

        self.piv2d_result_preview = PIV2DPreviewWidget("速度场预览")
        self.piv2d_result_preview.setMinimumSize(self._sp(500), self._sp(280))
        self.piv2d_result_preview.set_correlation_window_size(self.piv2d_win_spin.value())
        preview_splitter.addWidget(self.piv2d_result_preview)

        self.piv2d_log = QTextEdit()
        self.piv2d_log.setReadOnly(True)
        self.piv2d_log.setPlaceholderText("二维PIV日志")
        preview_splitter.addWidget(self.piv2d_log)
        preview_splitter.setSizes([220, 320, 180])

        timeline_panel = QGroupBox("图片时间轴")
        timeline_layout = QVBoxLayout(timeline_panel)

        self.piv2d_timeline_info = QLabel("未加载时间组")
        self.piv2d_timeline_info.setWordWrap(True)
        self.piv2d_timeline_info.setStyleSheet("color: #666;")
        timeline_layout.addWidget(self.piv2d_timeline_info)

        self.piv2d_timeline_list = QListWidget()
        self.piv2d_timeline_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.piv2d_timeline_list.currentRowChanged.connect(self._piv2d_on_timeline_selected)
        timeline_layout.addWidget(self.piv2d_timeline_list, stretch=1)

        self.piv2d_timeline_preview_label = QLabel("时间组选中预览")
        self.piv2d_timeline_preview_label.setAlignment(Qt.AlignCenter)
        self.piv2d_timeline_preview_label.setMinimumSize(self._sp(220), self._sp(150))
        self.piv2d_timeline_preview_label.setStyleSheet(
            "border: 1px solid #ccc; background: #f8f8f8;"
        )
        timeline_layout.addWidget(self.piv2d_timeline_preview_label)

        btn_assign_f1 = QPushButton("设为第1帧")
        btn_assign_f1.clicked.connect(lambda: self._piv2d_assign_selected_group(1))
        timeline_layout.addWidget(btn_assign_f1)

        btn_assign_f2 = QPushButton("设为第2帧")
        btn_assign_f2.clicked.connect(lambda: self._piv2d_assign_selected_group(2))
        timeline_layout.addWidget(btn_assign_f2)

        frame_pick_group = QGroupBox("双帧组合")
        frame_pick_layout = QGridLayout(frame_pick_group)
        frame_pick_layout.addWidget(QLabel("第1帧时间组:"), 0, 0)
        self.piv2d_frame1_group_combo = QComboBox()
        self.piv2d_frame1_group_combo.currentIndexChanged.connect(
            lambda idx: self._piv2d_on_group_combo_changed(1, idx)
        )
        frame_pick_layout.addWidget(self.piv2d_frame1_group_combo, 0, 1)
        frame_pick_layout.addWidget(QLabel("第1帧图片:"), 1, 0)
        self.piv2d_frame1_image_combo = QComboBox()
        self.piv2d_frame1_image_combo.currentIndexChanged.connect(
            lambda idx: self._piv2d_on_image_combo_changed(1, idx)
        )
        frame_pick_layout.addWidget(self.piv2d_frame1_image_combo, 1, 1)
        frame_pick_layout.addWidget(QLabel("第2帧时间组:"), 2, 0)
        self.piv2d_frame2_group_combo = QComboBox()
        self.piv2d_frame2_group_combo.currentIndexChanged.connect(
            lambda idx: self._piv2d_on_group_combo_changed(2, idx)
        )
        frame_pick_layout.addWidget(self.piv2d_frame2_group_combo, 2, 1)
        frame_pick_layout.addWidget(QLabel("第2帧图片:"), 3, 0)
        self.piv2d_frame2_image_combo = QComboBox()
        self.piv2d_frame2_image_combo.currentIndexChanged.connect(
            lambda idx: self._piv2d_on_image_combo_changed(2, idx)
        )
        frame_pick_layout.addWidget(self.piv2d_frame2_image_combo, 3, 1)
        timeline_layout.addWidget(frame_pick_group)

        self.piv2d_pair_info_label = QLabel("当前双帧: 未设置")
        self.piv2d_pair_info_label.setWordWrap(True)
        self.piv2d_pair_info_label.setStyleSheet("color: #666;")
        timeline_layout.addWidget(self.piv2d_pair_info_label)

        right_layout.addWidget(preview_splitter, stretch=1)
        right_layout.addWidget(timeline_panel)

        layout.addWidget(left_scroll)
        layout.addWidget(right_panel, stretch=1)
        self.content_stack.addWidget(page)

    def _piv2d_on_mode_changed(self, idx):
        single = (idx == 0)
        self.piv2d_single_widget.setVisible(single)
        self.piv2d_batch_widget.setVisible(not single)

    def _piv2d_on_batch_pair_mode_changed(self, idx):
        if hasattr(self, "piv2d_batch_pairs_edit"):
            self.piv2d_batch_pairs_edit.setEnabled(idx == 2)

    def _piv2d_open_frame1(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择第1帧图像", "", "图像文件 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
        )
        if path:
            self.piv2d_frame1_path = path
            self.piv2d_frame1_label.setText(os.path.basename(path))
            self.piv2d_frame1_label.setStyleSheet("")
            self._piv2d_set_preview(path, self.piv2d_frame1_preview)
            self._piv2d_update_pair_info()

    def _piv2d_open_frame2(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择第2帧图像", "", "图像文件 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
        )
        if path:
            self.piv2d_frame2_path = path
            self.piv2d_frame2_label.setText(os.path.basename(path))
            self.piv2d_frame2_label.setStyleSheet("")
            self._piv2d_set_preview(path, self.piv2d_frame2_preview)
            self._piv2d_update_pair_info()

    def _piv2d_pick_src_dir(self):
        d = QFileDialog.getExistingDirectory(self, "选择二维PIV输入目录")
        if d:
            self.piv2d_src_dir = d
            self.piv2d_src_label.setText(d)
            self.piv2d_src_label.setStyleSheet("")
            groups = self._piv2d_scan_time_groups(d)
            self.piv2d_time_groups = groups
            self._piv2d_refresh_timeline()
            if groups:
                image_count = sum(len(group["images"]) for group in groups)
                self.piv2d_log.append(
                    f"输入目录: {d}  (时间组 {len(groups)} 个, 图像 {image_count} 张)"
                )
            else:
                self.piv2d_log.append(f"输入目录: {d}  (未找到可用图像)")

    def _piv2d_pick_dst_dir(self):
        d = QFileDialog.getExistingDirectory(self, "选择二维PIV输出目录")
        if d:
            self.piv2d_dst_dir = d
            self.piv2d_dst_label.setText(d)
            self.piv2d_dst_label.setStyleSheet("")
            self.piv2d_log.append(f"输出目录: {d}")

    def _piv2d_get_config(self) -> PIV2DConfig:
        return PIV2DConfig(
            window_size=self.piv2d_win_spin.value(),
            overlap_ratio=self.piv2d_overlap_spin.value(),
            search_radius=self.piv2d_search_spin.value(),
            dt=self.piv2d_dt_spin.value(),
            pixel_scale=self.piv2d_scale_spin.value(),
            snr_threshold=self.piv2d_snr_spin.value(),
            max_displacement=self.piv2d_maxdisp_spin.value(),
            adaptive_enabled=self.piv2d_adaptive_check.isChecked() if hasattr(self, "piv2d_adaptive_check") else False,
            adaptive_window_sizes=self._piv2d_adaptive_window_sizes(),
            adaptive_residual_search_radius=(
                self.piv2d_adaptive_residual_spin.value()
                if hasattr(self, "piv2d_adaptive_residual_spin")
                else 6
            ),
            optical_flow_enabled=self.piv2d_flow_check.isChecked() if hasattr(self, "piv2d_flow_check") else False,
            optical_flow_levels=(
                self.piv2d_flow_levels_spin.value()
                if hasattr(self, "piv2d_flow_levels_spin")
                else 3
            ),
            optical_flow_winsize=(
                self._piv2d_odd_value(self.piv2d_flow_winsize_spin.value())
                if hasattr(self, "piv2d_flow_winsize_spin")
                else 15
            ),
            optical_flow_iterations=(
                self.piv2d_flow_iterations_spin.value()
                if hasattr(self, "piv2d_flow_iterations_spin")
                else 3
            ),
            outlier_filter_enabled=(
                self.piv2d_outlier_check.isChecked()
                if hasattr(self, "piv2d_outlier_check")
                else False
            ),
            outlier_replace_enabled=(
                self.piv2d_replace_outlier_check.isChecked()
                if hasattr(self, "piv2d_replace_outlier_check")
                else False
            ),
            outlier_median_threshold=(
                self.piv2d_outlier_median_spin.value()
                if hasattr(self, "piv2d_outlier_median_spin")
                else 3.0
            ),
            outlier_max_speed=(
                self.piv2d_outlier_speed_spin.value()
                if hasattr(self, "piv2d_outlier_speed_spin")
                else 0.0
            ),
            outlier_interp_iterations=(
                self.piv2d_outlier_interp_spin.value()
                if hasattr(self, "piv2d_outlier_interp_spin")
                else 5
            ),
        )

    def _piv2d_adaptive_window_sizes(self) -> tuple:
        if not hasattr(self, "piv2d_adaptive_windows_edit"):
            return (self.piv2d_win_spin.value(),)
        values = [int(x) for x in re.findall(r"\d+", self.piv2d_adaptive_windows_edit.text())]
        values = [v for v in values if v >= 8]
        if not values:
            values = [self.piv2d_win_spin.value()]
        return tuple(values)

    def _piv2d_on_adaptive_toggled(self, checked):
        if hasattr(self, "piv2d_adaptive_windows_edit"):
            self.piv2d_adaptive_windows_edit.setEnabled(checked)
        if hasattr(self, "piv2d_adaptive_residual_spin"):
            self.piv2d_adaptive_residual_spin.setEnabled(checked)

    def _piv2d_on_flow_toggled(self, checked):
        if checked and hasattr(self, "piv2d_adaptive_check"):
            self.piv2d_adaptive_check.setChecked(True)
        for name in ("piv2d_flow_levels_spin", "piv2d_flow_winsize_spin", "piv2d_flow_iterations_spin"):
            widget = getattr(self, name, None)
            if widget is not None:
                widget.setEnabled(checked)

    def _piv2d_on_outlier_toggled(self, checked):
        for name in (
            "piv2d_replace_outlier_check",
            "piv2d_outlier_median_spin",
            "piv2d_outlier_speed_spin",
            "piv2d_outlier_interp_spin",
        ):
            widget = getattr(self, name, None)
            if widget is not None:
                widget.setEnabled(checked)

    @staticmethod
    def _piv2d_odd_value(value: int) -> int:
        value = int(value)
        return value if value % 2 == 1 else value + 1

    def _piv2d_mask_shape(self) -> str:
        values = ["rectangle", "circle", "triangle", "polygon"]
        if not hasattr(self, "piv2d_mask_shape_combo"):
            return "rectangle"
        return values[self.piv2d_mask_shape_combo.currentIndex()]

    def _piv2d_sync_correlation_window_size(self, *_):
        size = self.piv2d_win_spin.value() if hasattr(self, "piv2d_win_spin") else None
        for preview_name in ("piv2d_frame1_preview", "piv2d_frame2_preview", "piv2d_result_preview"):
            preview = getattr(self, preview_name, None)
            if hasattr(preview, "set_correlation_window_size"):
                preview.set_correlation_window_size(size)

    def _piv2d_begin_mask_selection(self):
        if not hasattr(self, "piv2d_frame1_preview"):
            return
        shape = self._piv2d_mask_shape()
        if not self.piv2d_frame1_preview.begin_mask_selection(shape):
            QMessageBox.warning(self, "提示", "请先选择并预览第1帧图像")
            return
        if shape == "polygon":
            self.piv2d_log.append("请在第1帧预览图中左键添加多边形顶点，双击结束")
        else:
            self.piv2d_log.append("请在第1帧预览图中拖拽选择无粒子区域")

    def _piv2d_on_mask_changed(self, mask):
        self._piv2d_exclusion_mask = None if mask is None else np.asarray(mask, dtype=bool).copy()
        self._piv2d_update_mask_info()
        if hasattr(self, "piv2d_result_preview"):
            self.piv2d_result_preview.set_exclusion_mask(self._piv2d_exclusion_mask)

    def _piv2d_clear_mask(self):
        self._piv2d_exclusion_mask = None
        if hasattr(self, "piv2d_frame1_preview"):
            self.piv2d_frame1_preview.clear_exclusion_mask()
        if hasattr(self, "piv2d_result_preview"):
            self.piv2d_result_preview.set_exclusion_mask(None)
        self._piv2d_update_mask_info()
        self.piv2d_log.append("已清除无粒子区域")

    def _piv2d_update_mask_info(self):
        if not hasattr(self, "piv2d_mask_info_label"):
            return
        if self._piv2d_exclusion_mask is None or not np.any(self._piv2d_exclusion_mask):
            self.piv2d_mask_info_label.setText("未设置")
            return
        count = int(np.sum(self._piv2d_exclusion_mask))
        h, w = self._piv2d_exclusion_mask.shape
        self.piv2d_mask_info_label.setText(f"已屏蔽 {count} 像素 ({count / max(1, h * w) * 100:.1f}%)")

    def _piv2d_mask_for_image(self, image: np.ndarray) -> Optional[np.ndarray]:
        if self._piv2d_exclusion_mask is None:
            return None
        if self._piv2d_exclusion_mask.shape != image.shape[:2]:
            self.piv2d_log.append("无粒子区域尺寸与当前图像不一致，已忽略")
            return None
        return self._piv2d_exclusion_mask

    def _piv2d_batch_files(self) -> List[Path]:
        if not self.piv2d_src_dir or not os.path.isdir(self.piv2d_src_dir):
            return []
        return sorted(
            p for p in Path(self.piv2d_src_dir).iterdir()
            if p.is_file() and p.suffix.lower() in PIV2D_EXTS
        )

    def _piv2d_build_batch_pairs(self, files: List[Path]) -> Optional[List[tuple]]:
        mode = self.piv2d_batch_pair_mode_combo.currentIndex() if hasattr(self, "piv2d_batch_pair_mode_combo") else 0
        n = len(files)
        if mode == 0:
            pairs = [(i, i + 1) for i in range(max(0, n - 1))]
        elif mode == 1:
            pairs = [(i, i + 1) for i in range(0, n - 1, 2)]
        else:
            pairs = self._piv2d_parse_custom_batch_pairs(
                self.piv2d_batch_pairs_edit.text() if hasattr(self, "piv2d_batch_pairs_edit") else "",
                n,
            )
        if not pairs:
            return None
        return pairs

    def _piv2d_parse_custom_batch_pairs(self, text: str, image_count: int) -> List[tuple]:
        text = text.strip()
        if not text:
            raise ValueError("请输入自定义帧对，例如: 1-2, 3-4, 5-6")

        pair_matches = re.findall(r"(\d+)\s*[-:~>]\s*(\d+)", text)
        if pair_matches:
            pairs = [(int(a) - 1, int(b) - 1) for a, b in pair_matches]
        else:
            nums = [int(x) for x in re.findall(r"\d+", text)]
            if len(nums) < 2 or len(nums) % 2 != 0:
                raise ValueError("自定义数列需要成对输入，例如: 1,2,3,4 或 1-2,3-4")
            pairs = [(nums[i] - 1, nums[i + 1] - 1) for i in range(0, len(nums), 2)]

        invalid = [
            (a + 1, b + 1)
            for a, b in pairs
            if a < 0 or b < 0 or a >= image_count or b >= image_count or a == b
        ]
        if invalid:
            raise ValueError(f"自定义帧对超出范围或重复同一帧: {invalid[0][0]}-{invalid[0][1]}")
        return pairs

    def _piv2d_start_processing_timer(self, prefix: str, progress_max: int = 0):
        self._piv2d_processing_prefix = prefix
        self._piv2d_processing_start = time.perf_counter()
        self.processing_time_label.setText(f"{prefix} 处理时间: 0.0 s")
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, progress_max)
        self.progress_bar.setValue(0)
        if progress_max == 0:
            self.progress_bar.setFormat(f"{prefix}处理中")
        else:
            self.progress_bar.setFormat(f"{prefix}: %p%")
        self._piv2d_elapsed_timer.start(200)

    def _piv2d_update_elapsed_time(self):
        if self._piv2d_processing_start <= 0:
            return
        elapsed = time.perf_counter() - self._piv2d_processing_start
        self.processing_time_label.setText(f"{self._piv2d_processing_prefix} 处理时间: {elapsed:.1f} s")

    def _piv2d_finish_processing_timer(self, progress_value: Optional[int] = None):
        self._piv2d_elapsed_timer.stop()
        self._piv2d_update_elapsed_time()
        if progress_value is not None:
            self.progress_bar.setValue(progress_value)
        self.progress_bar.setVisible(False)

    def _piv2d_run_single(self):
        import cv2 as _cv2
        if not self.piv2d_frame1_path or not self.piv2d_frame2_path:
            QMessageBox.warning(self, "提示", "请先选择两帧图像")
            return

        img1 = robust_imread(self.piv2d_frame1_path, _cv2.IMREAD_UNCHANGED)
        img2 = robust_imread(self.piv2d_frame2_path, _cv2.IMREAD_UNCHANGED)
        if img1 is None or img2 is None:
            QMessageBox.warning(self, "提示", "图像读取失败")
            return

        self._piv2d_start_processing_timer("单组二维PIV", 0)
        try:
            calculator = PIV2DCalculator(self._piv2d_get_config())
            exclusion_mask = self._piv2d_mask_for_image(img1)
            result = calculator.compute_velocity_field(img1, img2, exclusion_mask=exclusion_mask)
            summary = calculator.summarize_result(result)
            self._piv2d_last_result = result
            self._piv2d_last_image = img1
            self._piv2d_show_vector_result(img1, result, "单组速度矢量")

            self.piv2d_log.append("=== 二维PIV单组计算完成 ===")
            self.piv2d_log.append(
                f"算法: {result.get('algorithm', 'fixed')}  "
                f"窗口序列: {list(result.get('window_sizes', [self.piv2d_win_spin.value()]))}"
            )
            outlier_count = int(np.sum(result.get("outlier", np.zeros_like(result["valid"], dtype=bool))))
            replaced_count = int(np.sum(result.get("replaced", np.zeros_like(result["valid"], dtype=bool))))
            if outlier_count or replaced_count:
                self.piv2d_log.append(f"矢量后处理: 剔除 {outlier_count} 个, 插值替换 {replaced_count} 个")
            self.piv2d_log.append(
                f"有效矢量: {summary['valid_count']}/{summary['total_count']}  "
                f"平均速度: {summary['mean_speed']:.3f}  最大速度: {summary['max_speed']:.3f}"
            )
            self.piv2d_log.append(
                f"平均 U: {summary['mean_u']:.3f}  平均 V: {summary['mean_v']:.3f}  平均SNR: {summary['mean_snr']:.3f}"
            )
        except Exception as e:
            self.piv2d_log.append(f"二维PIV失败: {e}")
            QMessageBox.critical(self, "二维PIV错误", str(e))
        finally:
            self._piv2d_finish_processing_timer()

    def _piv2d_run_batch(self):
        if not self.piv2d_src_dir or not os.path.isdir(self.piv2d_src_dir):
            QMessageBox.warning(self, "提示", "请先选择输入目录")
            return
        if not self.piv2d_dst_dir:
            QMessageBox.warning(self, "提示", "请先选择输出目录")
            return
        if any(group.get("source_type") == "subdir" for group in self.piv2d_time_groups):
            QMessageBox.information(
                self,
                "提示",
                "当前输入目录包含时间组子文件夹。\n"
                "批量二维PIV仍按根目录中的连续图像处理；\n"
                "跨时间组双帧请在右侧时间轴中选择后使用“运行单组二维PIV”。"
            )
            return
        os.makedirs(self.piv2d_dst_dir, exist_ok=True)

        batch_files = self._piv2d_batch_files()
        if len(batch_files) < 2:
            QMessageBox.warning(self, "提示", "输入目录中至少需要2张可用图像")
            return
        try:
            batch_pairs = self._piv2d_build_batch_pairs(batch_files)
        except ValueError as exc:
            QMessageBox.warning(self, "提示", str(exc))
            return
        if not batch_pairs:
            QMessageBox.warning(self, "提示", "没有可计算的双帧组合")
            return

        batch_mask = self._piv2d_exclusion_mask
        if batch_mask is not None:
            first_file = batch_files[batch_pairs[0][0]]
            if first_file is not None:
                first_image = robust_imread(str(first_file), cv2.IMREAD_UNCHANGED)
                if first_image is not None and batch_mask.shape != first_image.shape[:2]:
                    QMessageBox.warning(self, "提示", "无粒子区域尺寸与批量图像不一致，请重新选择或清除该区域")
                    return

        self.piv2d_progress.setValue(0)
        self.piv2d_progress.setVisible(True)
        self.piv2d_batch_run_btn.setEnabled(False)
        self.piv2d_stop_btn.setEnabled(True)
        self._piv2d_start_processing_timer("批量二维PIV", 100)
        pair_preview = ", ".join(f"{a + 1}-{b + 1}" for a, b in batch_pairs[:8])
        if len(batch_pairs) > 8:
            pair_preview += ", ..."
        config = self._piv2d_get_config()
        algo_text = (
            f"自适应多重网格 {list(config.adaptive_window_sizes)}"
            if config.adaptive_enabled
            else f"固定窗口 {config.window_size}"
        )
        self.piv2d_log.append(
            f"=== 开始批量二维PIV ===\n"
            f"输入: {self.piv2d_src_dir}\n"
            f"输出: {self.piv2d_dst_dir}\n"
            f"算法: {algo_text}\n"
            f"帧对: {pair_preview}  (共 {len(batch_pairs)} 组)"
        )

        self.piv2d_worker_thread = BatchPIV2DWorker(
            self.piv2d_src_dir,
            self.piv2d_dst_dir,
            config,
            exclusion_mask=batch_mask,
            frame_pairs=batch_pairs,
        )
        self.piv2d_worker_thread.progress.connect(self._piv2d_on_batch_progress)
        self.piv2d_worker_thread.finished.connect(self._piv2d_on_batch_finished)
        self.piv2d_worker_thread.error.connect(self._piv2d_on_batch_error)
        self.piv2d_worker_thread.start()

    def _piv2d_stop_batch(self):
        if self.piv2d_worker_thread and self.piv2d_worker_thread.isRunning():
            self.piv2d_worker_thread.stop()
            self.piv2d_log.append("已发送停止信号...")

    def _piv2d_on_batch_progress(self, done, total, name):
        pct = int(done / total * 100) if total > 0 else 0
        self.piv2d_progress.setValue(pct)
        self.progress_bar.setValue(pct)
        self.progress_bar.setFormat(f"批量二维PIV: {pct}%")
        self.piv2d_log.append(f"[{done}/{total}] {name}")

    def _piv2d_on_batch_finished(self, success, total, outputs):
        self.piv2d_progress.setVisible(False)
        self._piv2d_finish_processing_timer(100)
        self.piv2d_batch_run_btn.setEnabled(True)
        self.piv2d_stop_btn.setEnabled(False)
        self.piv2d_log.append(f"批量二维PIV完成: {success}/{total}")
        if outputs:
            self._piv2d_load_batch_vector_preview(outputs[0])
        QMessageBox.information(
            self,
            "完成",
            f"批量二维PIV完成!\n成功: {success}/{total}\n输出目录: {self.piv2d_dst_dir}",
        )

    def _piv2d_on_batch_error(self, msg):
        self.piv2d_progress.setVisible(False)
        self._piv2d_finish_processing_timer()
        self.piv2d_batch_run_btn.setEnabled(True)
        self.piv2d_stop_btn.setEnabled(False)
        self.piv2d_log.append(f"批量二维PIV出错: {msg}")
        QMessageBox.critical(self, "二维PIV错误", msg)

    def _piv2d_set_preview(self, path: str, label: QLabel):
        import cv2 as _cv2
        img = robust_imread(path, _cv2.IMREAD_UNCHANGED)
        if img is None:
            return
        if hasattr(label, "set_image_array"):
            label.set_image_array(img, os.path.basename(path))
            if label is getattr(self, "piv2d_frame1_preview", None):
                label.set_exclusion_mask(self._piv2d_exclusion_mask)
            return
        pixmap = self._ie_ndarray_to_pixmap(img)
        if pixmap:
            label.setPixmap(pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _piv2d_vector_color_mode(self) -> str:
        values = ["speed", "red", "green", "blue", "yellow", "white"]
        if not hasattr(self, "piv2d_vector_color_combo"):
            return "speed"
        return values[self.piv2d_vector_color_combo.currentIndex()]

    def _piv2d_show_vector_result(self, image: np.ndarray, result: dict, title: str):
        if not hasattr(self.piv2d_result_preview, "set_vector_result"):
            return
        self.piv2d_result_preview.set_vector_style(
            self.piv2d_vector_scale_spin.value(),
            self._piv2d_vector_color_mode(),
        )
        self.piv2d_result_preview.set_vector_result(image, result, title)
        self.piv2d_result_preview.set_exclusion_mask(self._piv2d_mask_for_image(image))

    def _piv2d_refresh_vector_preview(self, *_):
        if self._piv2d_last_image is None or self._piv2d_last_result is None:
            return
        self._piv2d_show_vector_result(
            self._piv2d_last_image,
            self._piv2d_last_result,
            "速度矢量",
        )

    def _piv2d_load_batch_vector_preview(self, overlay_path: str):
        overlay = Path(overlay_path)
        data_path = overlay.with_name(overlay.name.replace("_overlay.png", "_field.npz"))
        files = sorted(
            p for p in Path(self.piv2d_src_dir).iterdir()
            if p.is_file() and p.suffix.lower() in PIV2D_EXTS
        )
        if not data_path.exists() or not files:
            self._piv2d_set_preview(overlay_path, self.piv2d_result_preview)
            return

        source_stem = overlay.name.split("_to_", 1)[0]
        source_file = next((p for p in files if p.stem == source_stem), files[0])
        image = robust_imread(str(source_file), cv2.IMREAD_UNCHANGED)
        if image is None:
            self._piv2d_set_preview(overlay_path, self.piv2d_result_preview)
            return

        data = np.load(str(data_path))
        result = {
            "x": data["x"],
            "y": data["y"],
            "u": data["u"],
            "v": data["v"],
            "speed": data["speed"],
            "snr": data["snr"],
            "valid": data["valid"],
            "u_px": data["u_px"] if "u_px" in data.files else data["u"],
            "v_px": data["v_px"] if "v_px" in data.files else data["v"],
            "valid_original": data["valid_original"] if "valid_original" in data.files else data["valid"],
            "outlier": data["outlier"] if "outlier" in data.files else np.zeros_like(data["valid"], dtype=bool),
            "replaced": data["replaced"] if "replaced" in data.files else np.zeros_like(data["valid"], dtype=bool),
            "excluded": data["excluded"] if "excluded" in data.files else np.zeros_like(data["valid"], dtype=bool),
            "algorithm": str(data["algorithm"]) if "algorithm" in data.files else "fixed",
            "window_sizes": data["window_sizes"] if "window_sizes" in data.files else np.array([self.piv2d_win_spin.value()]),
            "optical_flow_residual_u_px": (
                data["optical_flow_residual_u_px"]
                if "optical_flow_residual_u_px" in data.files
                else np.zeros_like(data["valid"], dtype=np.float32)
            ),
            "optical_flow_residual_v_px": (
                data["optical_flow_residual_v_px"]
                if "optical_flow_residual_v_px" in data.files
                else np.zeros_like(data["valid"], dtype=np.float32)
            ),
        }
        self._piv2d_last_image = image
        self._piv2d_last_result = result
        self._piv2d_show_vector_result(image, result, "批量速度矢量")

    def _piv2d_scan_time_groups(self, src_dir: str) -> List[dict]:
        src_path = Path(src_dir)
        if not src_path.exists():
            return []

        groups: List[dict] = []
        subdirs = sorted([p for p in src_path.iterdir() if p.is_dir()])
        for subdir in subdirs:
            images = sorted(
                str(p) for p in subdir.iterdir()
                if p.is_file() and p.suffix.lower() in PIV2D_EXTS
            )
            if images:
                groups.append({
                    "name": subdir.name,
                    "images": images,
                    "source_type": "subdir",
                })

        if groups:
            return groups

        files = sorted(
            str(p) for p in src_path.iterdir()
            if p.is_file() and p.suffix.lower() in PIV2D_EXTS
        )
        for idx, path in enumerate(files):
            groups.append({
                "name": f"t{idx:03d}",
                "images": [path],
                "source_type": "file",
            })
        return groups

    def _piv2d_refresh_timeline(self):
        self.piv2d_timeline_list.clear()
        self.piv2d_frame1_group_combo.clear()
        self.piv2d_frame2_group_combo.clear()
        self.piv2d_frame1_image_combo.clear()
        self.piv2d_frame2_image_combo.clear()

        if not self.piv2d_time_groups:
            self.piv2d_timeline_info.setText("未加载时间组")
            self.piv2d_pair_info_label.setText("当前双帧: 未设置")
            self.piv2d_timeline_preview_label.setText("时间组选中预览")
            self.piv2d_timeline_preview_label.setPixmap(QPixmap())
            return

        total_images = sum(len(group["images"]) for group in self.piv2d_time_groups)
        self.piv2d_timeline_info.setText(
            f"共 {len(self.piv2d_time_groups)} 个时间组，{total_images} 张图片"
        )

        for idx, group in enumerate(self.piv2d_time_groups):
            item = QListWidgetItem(f"{idx:02d} | {group['name']} ({len(group['images'])} 张)")
            item.setData(Qt.UserRole, idx)
            self.piv2d_timeline_list.addItem(item)
            self.piv2d_frame1_group_combo.addItem(group["name"], idx)
            self.piv2d_frame2_group_combo.addItem(group["name"], idx)

        self.piv2d_timeline_list.setCurrentRow(0)
        self.piv2d_frame1_group_combo.setCurrentIndex(0)
        self.piv2d_frame2_group_combo.setCurrentIndex(min(1, len(self.piv2d_time_groups) - 1))
        self._piv2d_rebuild_image_combo(1)
        self._piv2d_rebuild_image_combo(2)
        self._piv2d_apply_combo_selection(1)
        self._piv2d_apply_combo_selection(2)

    def _piv2d_on_timeline_selected(self, row: int):
        if row < 0 or row >= len(self.piv2d_time_groups):
            return
        group = self.piv2d_time_groups[row]
        first_image = group["images"][0] if group["images"] else ""
        if first_image:
            self._piv2d_set_preview(first_image, self.piv2d_timeline_preview_label)
        self.piv2d_timeline_info.setText(
            f"当前时间组: {group['name']} | 图片数: {len(group['images'])}"
        )

    def _piv2d_assign_selected_group(self, frame_num: int):
        row = self.piv2d_timeline_list.currentRow()
        if row < 0 or row >= len(self.piv2d_time_groups):
            QMessageBox.warning(self, "提示", "请先在时间轴中选择一个时间组")
            return
        combo = self.piv2d_frame1_group_combo if frame_num == 1 else self.piv2d_frame2_group_combo
        combo.setCurrentIndex(row)
        self._piv2d_rebuild_image_combo(frame_num)
        self._piv2d_apply_combo_selection(frame_num)

    def _piv2d_on_group_combo_changed(self, frame_num: int, _: int):
        self._piv2d_rebuild_image_combo(frame_num)
        self._piv2d_apply_combo_selection(frame_num)

    def _piv2d_rebuild_image_combo(self, frame_num: int):
        group_combo = self.piv2d_frame1_group_combo if frame_num == 1 else self.piv2d_frame2_group_combo
        image_combo = self.piv2d_frame1_image_combo if frame_num == 1 else self.piv2d_frame2_image_combo
        image_combo.blockSignals(True)
        image_combo.clear()

        group_idx = group_combo.currentData()
        if group_idx is None or group_idx < 0 or group_idx >= len(self.piv2d_time_groups):
            image_combo.blockSignals(False)
            return

        group = self.piv2d_time_groups[group_idx]
        for img_idx, path in enumerate(group["images"]):
            image_combo.addItem(f"{img_idx:02d} | {os.path.basename(path)}", path)
        image_combo.blockSignals(False)
        if image_combo.count() > 0:
            image_combo.setCurrentIndex(0)

    def _piv2d_on_image_combo_changed(self, frame_num: int, _: int):
        self._piv2d_apply_combo_selection(frame_num)

    def _piv2d_apply_combo_selection(self, frame_num: int):
        image_combo = self.piv2d_frame1_image_combo if frame_num == 1 else self.piv2d_frame2_image_combo
        path = image_combo.currentData()
        if not path:
            return

        if frame_num == 1:
            self.piv2d_frame1_path = path
            self.piv2d_frame1_label.setText(os.path.basename(path))
            self.piv2d_frame1_label.setStyleSheet("")
            self._piv2d_set_preview(path, self.piv2d_frame1_preview)
        else:
            self.piv2d_frame2_path = path
            self.piv2d_frame2_label.setText(os.path.basename(path))
            self.piv2d_frame2_label.setStyleSheet("")
            self._piv2d_set_preview(path, self.piv2d_frame2_preview)

        self._piv2d_update_pair_info()

    def _piv2d_update_pair_info(self):
        def _group_name(combo: QComboBox) -> str:
            idx = combo.currentData()
            if idx is None or idx < 0 or idx >= len(self.piv2d_time_groups):
                return "--"
            return self.piv2d_time_groups[idx]["name"]

        f1_name = os.path.basename(self.piv2d_frame1_path) if self.piv2d_frame1_path else "--"
        f2_name = os.path.basename(self.piv2d_frame2_path) if self.piv2d_frame2_path else "--"
        self.piv2d_pair_info_label.setText(
            f"关于:\n"
            f"第1帧 = {_group_name(self.piv2d_frame1_group_combo)} / {f1_name}\n"
            f"第2帧 = {_group_name(self.piv2d_frame2_group_combo)} / {f2_name}"
        )

    # ------------------------------------------------------------
    #  通用图像处理页面（Page 5）— DaVis 10 风格三栏布局
    # ------------------------------------------------------------

    def _create_image_editor_page(self):
        """创建图像处理页面：DaVis 10 风格三栏布局。"""
        from utils.image_editor import ImageEditor as _IE

        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ===== 顶部工具栏 =====
        toolbar = QFrame()
        toolbar.setFixedHeight(self._sp(42))
        toolbar.setStyleSheet(
            "QFrame { background: #2b2b2b; border-bottom: 1px solid #444; }"
            "QPushButton { background: #3c3c3c; color: #ddd; border: 1px solid #555; "
            "  border-radius: 3px; padding: 4px 10px; font-size: 12px; }"
            "QPushButton:hover { background: #505050; }"
            "QPushButton:pressed { background: #606060; }"
            "QPushButton:disabled { background: #333; color: #666; }"
            "QLabel { color: #ccc; font-size: 12px; }"
        )
        tb_lay = QHBoxLayout(toolbar)
        tb_lay.setContentsMargins(self._sp(8), self._sp(4), self._sp(8), self._sp(4))

        self.ie_run_btn = QPushButton("处理")
        self.ie_run_btn.setFixedWidth(self._sp(80))
        self.ie_run_btn.setStyleSheet(
            "QPushButton { background: #2196F3; color: white; border: none; "
            "  border-radius: 3px; padding: 4px 12px; font-weight: bold; }"
            "QPushButton:hover { background: #1976D2; }"
            "QPushButton:disabled { background: #555; color: #888; }"
        )
        self.ie_run_btn.clicked.connect(self._ie_do_preview)
        tb_lay.addWidget(self.ie_run_btn)

        self.ie_stop_btn = QPushButton("■ 停止")
        self.ie_stop_btn.setFixedWidth(self._sp(80))
        self.ie_stop_btn.setEnabled(False)
        self.ie_stop_btn.clicked.connect(self._ie_stop_batch)
        tb_lay.addWidget(self.ie_stop_btn)

        self.ie_reset_btn = QPushButton("↺ 重置")
        self.ie_reset_btn.setFixedWidth(self._sp(80))
        self.ie_reset_btn.clicked.connect(self._ie_reset_workflow)
        tb_lay.addWidget(self.ie_reset_btn)

        tb_lay.addSpacing(self._sp(16))

        # 缩放控件
        tb_lay.addWidget(QLabel("缩放:"))
        self.ie_zoom_out_btn = QPushButton("−")
        self.ie_zoom_out_btn.setFixedSize(self._sp(28), self._sp(28))
        self.ie_zoom_out_btn.clicked.connect(lambda: self._ie_zoom(-1))
        tb_lay.addWidget(self.ie_zoom_out_btn)

        self.ie_zoom_label = QLabel("100%")
        self.ie_zoom_label.setFixedWidth(self._sp(50))
        self.ie_zoom_label.setAlignment(Qt.AlignCenter)
        tb_lay.addWidget(self.ie_zoom_label)

        self.ie_zoom_in_btn = QPushButton("+")
        self.ie_zoom_in_btn.setFixedSize(self._sp(28), self._sp(28))
        self.ie_zoom_in_btn.clicked.connect(lambda: self._ie_zoom(1))
        tb_lay.addWidget(self.ie_zoom_in_btn)

        self.ie_fit_btn = QPushButton("适应窗口")
        self.ie_fit_btn.clicked.connect(lambda: self._ie_zoom(0))
        tb_lay.addWidget(self.ie_fit_btn)

        tb_lay.addStretch()

        # 输入模式切换
        self.ie_mode_combo = QComboBox()
        self.ie_mode_combo.addItems(["单张图像", "批量目录"])
        self.ie_mode_combo.setStyleSheet(
            "QComboBox { background: #3c3c3c; color: #ddd; border: 1px solid #555; "
            "  border-radius: 3px; padding: 3px 8px; }"
            "QComboBox::drop-down { border: none; }"
            "QComboBox QAbstractItemView { background: #3c3c3c; color: #ddd; "
            "  selection-background-color: #505050; }"
        )
        self.ie_mode_combo.currentIndexChanged.connect(self._ie_on_mode_changed)
        tb_lay.addWidget(self.ie_mode_combo)

        outer.addWidget(toolbar)

        # ===== 三栏主体 =====
        body_splitter = QSplitter(Qt.Horizontal)
        body_splitter.setStyleSheet("QSplitter::handle { background: #444; width: 2px; }")

        # --- 左栏：操作面板 ---
        ops_panel = self._ie_create_ops_panel(_IE)
        body_splitter.addWidget(ops_panel)

        # --- 中栏：工作流画布 ---
        canvas_widget = self._ie_create_workflow_canvas(_IE)
        body_splitter.addWidget(canvas_widget)

        # --- 右栏：查看器区域 ---
        viewer_widget = self._ie_create_viewer_area()
        body_splitter.addWidget(viewer_widget)

        body_splitter.setSizes([self._sp(220), self._sp(320), self._sp(520)])
        outer.addWidget(body_splitter, stretch=1)

        self.content_stack.addWidget(page)

        # 实时预览定时器（防抖）
        self._ie_preview_timer = QTimer(self)
        self._ie_preview_timer.setSingleShot(True)
        self._ie_preview_timer.setInterval(300)
        self._ie_preview_timer.timeout.connect(self._ie_do_preview)

        # 初始化工作流节点
        self._ie_init_workflow_nodes(_IE)

        # 缩放状态
        self._ie_zoom_level = 100

    # ===== DaVis 风格子组件 =====

    def _ie_create_ops_panel(self, _IE):
        """左栏：操作面板 — 搜索框 + 可折叠算法分组。"""
        panel = QWidget()
        panel.setStyleSheet(
            "QWidget { background: #2b2b2b; }"
            "QLabel { color: #ccc; }"
            "QLineEdit { background: #3c3c3c; color: #eee; border: 1px solid #555; "
            "  border-radius: 3px; padding: 5px 8px; }"
            "QLineEdit::placeholder { color: #888; }"
            "QGroupBox { color: #aaa; border: none; border-top: 1px solid #444; "
            "  margin-top: 12px; padding-top: 16px; font-weight: bold; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }"
        )
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(self._sp(6), self._sp(6), self._sp(6), self._sp(6))
        lay.setSpacing(self._sp(4))

        # 搜索框
        self.ie_ops_search = QLineEdit()
        self.ie_ops_search.setPlaceholderText("🔍 搜索算法...")
        self.ie_ops_search.textChanged.connect(self._ie_filter_ops)
        lay.addWidget(self.ie_ops_search)

        # 可滚动区域
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(
            "QScrollArea { border: none; background: #2b2b2b; }"
            "QScrollBar:vertical { background: #2b2b2b; width: 6px; }"
            "QScrollBar::handle:vertical { background: #555; border-radius: 3px; }"
        )
        scroll_content = QWidget()
        scroll_lay = QVBoxLayout(scroll_content)
        scroll_lay.setContentsMargins(0, 0, 0, 0)
        scroll_lay.setSpacing(self._sp(2))

        # 算法分组定义
        algo_groups = {
            "📐 几何变换": [
                ("crop", "裁剪 (ROI)", "设置感兴趣区域"),
                ("mirror", "镜像翻转", "水平/垂直镜像"),
                ("rotate", "旋转", "90°/180°/自定义角度"),
            ],
            "🎨 灰度处理": [
                ("gray", "灰度转换", "彩色→8-bit灰度"),
                ("bit_depth", "位深转换", "24/16/12位→8位"),
                ("gray_math", "灰度运算", "平均/log/exp/sqrt/sqr"),
            ],
            "☀️ 图像增强": [
                ("bc", "亮度/对比度", "调整α/β参数"),
                ("threshold", "阈值化", "全局/大津/自适应"),
            ],
            "🔢 运算处理": [
                ("arithmetic", "图像加减", "标量或双图运算"),
            ],
        }

        self._ie_algo_cards = {}  # key -> card widget
        for group_name, algos in algo_groups.items():
            group_box = QGroupBox(group_name)
            group_box.setCheckable(True)
            group_box.setChecked(True)
            g_lay = QVBoxLayout(group_box)
            g_lay.setContentsMargins(self._sp(4), self._sp(4), self._sp(4), self._sp(4))
            g_lay.setSpacing(self._sp(2))

            for key, label, desc in algos:
                card = self._ie_make_algo_card(key, label, desc)
                g_lay.addWidget(card)
                self._ie_algo_cards[key] = card

            scroll_lay.addWidget(group_box)

        scroll_lay.addStretch()
        scroll.setWidget(scroll_content)
        lay.addWidget(scroll, stretch=1)

        panel.setFixedWidth(self._sp(220))
        return panel

    def _ie_make_algo_card(self, key, label, desc):
        """创建单个算法卡片（可点击添加 / 可拖拽到画布）。"""
        card = _IEAlgoCard(key, label, desc, self._sp)
        card.add_requested.connect(self._ie_add_node)
        return card

    def _ie_create_workflow_canvas(self, _IE):
        """中栏：工作流画布 — 垂直节点卡片流，支持拖拽添加和节点排序。"""
        canvas = _IEWorkflowCanvas(self._sp)
        canvas.setStyleSheet("background: #1e1e1e;")
        canvas.node_drop_requested.connect(self._ie_add_node)
        canvas.node_move_requested.connect(self._ie_move_node)
        c_lay = QVBoxLayout(canvas)
        c_lay.setContentsMargins(self._sp(6), self._sp(6), self._sp(6), self._sp(6))
        c_lay.setSpacing(0)

        # 标题
        title = QLabel("⚙ 工作流")
        title.setStyleSheet(
            "color: #ccc; font-size: 14px; font-weight: bold; "
            "padding: 4px; border: none;"
        )
        c_lay.addWidget(title)

        # 可滚动节点区域
        self.ie_canvas_scroll = QScrollArea()
        self.ie_canvas_scroll.setWidgetResizable(True)
        self.ie_canvas_scroll.setStyleSheet(
            "QScrollArea { border: none; background: #1e1e1e; }"
            "QScrollBar:vertical { background: #1e1e1e; width: 6px; }"
            "QScrollBar::handle:vertical { background: #555; border-radius: 3px; }"
        )

        self.ie_canvas_content = QWidget()
        self.ie_canvas_layout = QVBoxLayout(self.ie_canvas_content)
        self.ie_canvas_layout.setContentsMargins(0, 0, 0, 0)
        self.ie_canvas_layout.setSpacing(0)
        self.ie_canvas_layout.addStretch()

        self.ie_canvas_scroll.setWidget(self.ie_canvas_content)
        c_lay.addWidget(self.ie_canvas_scroll, stretch=1)

        # 节点引用列表
        self._ie_workflow_nodes = {}  # key -> node widget

        return canvas

    def _ie_create_viewer_area(self):
        """右栏：查看器区域 — 上下分割源图/结果。"""
        viewer = QWidget()
        viewer.setStyleSheet("background: #1e1e1e;")
        v_lay = QVBoxLayout(viewer)
        v_lay.setContentsMargins(self._sp(4), self._sp(4), self._sp(4), self._sp(4))
        v_lay.setSpacing(0)

        # 输入源选择区域
        source_bar = QFrame()
        source_bar.setStyleSheet(
            "QFrame { background: #2b2b2b; border-radius: 4px; border: none; }"
            "QPushButton { background: #3c3c3c; color: #ddd; border: 1px solid #555; "
            "  border-radius: 3px; padding: 3px 10px; font-size: 11px; }"
            "QPushButton:hover { background: #505050; }"
            "QLabel { color: #aaa; font-size: 11px; border: none; }"
        )
        sb_lay = QHBoxLayout(source_bar)
        sb_lay.setContentsMargins(self._sp(6), self._sp(4), self._sp(6), self._sp(4))

        sb_lay.addWidget(QLabel("数据源:"))

        # 单张模式控件
        self.ie_single_widget = QWidget()
        sg_lay = QHBoxLayout(self.ie_single_widget)
        sg_lay.setContentsMargins(0, 0, 0, 0)
        btn_open_single = QPushButton("📄 选择图像")
        btn_open_single.clicked.connect(self._ie_open_single)
        sg_lay.addWidget(btn_open_single)
        self.ie_single_label = QLabel("未选择文件")
        self.ie_single_label.setStyleSheet("color: #888; border: none;")
        sg_lay.addWidget(self.ie_single_label)
        sb_lay.addWidget(self.ie_single_widget)

        # 批量模式控件
        self.ie_batch_widget = QWidget()
        bg_lay = QHBoxLayout(self.ie_batch_widget)
        bg_lay.setContentsMargins(0, 0, 0, 0)
        btn_src_dir = QPushButton("输入目录")
        btn_src_dir.clicked.connect(self._ie_pick_src_dir)
        bg_lay.addWidget(btn_src_dir)
        self.ie_src_label = QLabel("未选择")
        self.ie_src_label.setStyleSheet("color: #888; border: none;")
        bg_lay.addWidget(self.ie_src_label)
        btn_dst_dir = QPushButton("输出目录")
        btn_dst_dir.clicked.connect(self._ie_pick_dst_dir)
        bg_lay.addWidget(btn_dst_dir)
        self.ie_dst_label = QLabel("未选择")
        self.ie_dst_label.setStyleSheet("color: #888; border: none;")
        bg_lay.addWidget(self.ie_dst_label)
        bg_lay.addWidget(QLabel("CPU workers"))
        self.ie_workers_spin = QSpinBox()
        self.ie_workers_spin.setRange(1, default_worker_count(os.cpu_count()))
        self.ie_workers_spin.setValue(default_worker_count())
        self.ie_workers_spin.setToolTip("Parallel image-processing workers; default leaves CPU cores free.")
        bg_lay.addWidget(self.ie_workers_spin)
        sb_lay.addWidget(self.ie_batch_widget)
        self.ie_batch_widget.hide()

        # 保存/批量按钮
        sb_lay.addStretch()
        self.ie_batch_run_btn = QPushButton("▶ 批量处理")
        self.ie_batch_run_btn.setStyleSheet(
            "QPushButton { background: #E64A19; color: white; border: none; "
            "  border-radius: 3px; padding: 4px 12px; font-weight: bold; }"
            "QPushButton:hover { background: #BF360C; }"
            "QPushButton:disabled { background: #555; color: #888; }"
        )
        self.ie_batch_run_btn.clicked.connect(self._ie_run_batch)
        sb_lay.addWidget(self.ie_batch_run_btn)

        btn_save_single = QPushButton("💾 保存")
        btn_save_single.setStyleSheet(
            "QPushButton { background: #388E3C; color: white; border: none; "
            "  border-radius: 3px; padding: 4px 12px; font-weight: bold; }"
            "QPushButton:hover { background: #2E7D32; }"
        )
        btn_save_single.clicked.connect(self._ie_save_single)
        sb_lay.addWidget(btn_save_single)

        # 文件名模式行
        fname_bar = QFrame()
        fname_bar.setStyleSheet(
            "QFrame { background: #2b2b2b; border-radius: 3px; border: none; }"
            "QLabel { color: #aaa; font-size: 11px; border: none; }"
            "QLineEdit { background: #3c3c3c; color: #eee; border: 1px solid #555; "
            "  border-radius: 3px; padding: 2px 6px; font-size: 11px; }"
            "QLineEdit:focus { border: 1px solid #2196F3; }"
        )
        fb_lay = QHBoxLayout(fname_bar)
        fb_lay.setContentsMargins(self._sp(6), self._sp(2), self._sp(6), self._sp(2))

        fb_lay.addWidget(QLabel("输出文件名:"))

        self.ie_filename_edit = QLineEdit()
        self.ie_filename_edit.setPlaceholderText("{original}_processed")
        self.ie_filename_edit.setToolTip(
            "支持占位符:\n"
            "{original} - 原文件名（不含扩展名）\n"
            "{index} - 帧序号（批量时）\n"
            "{step} - 处理步骤名\n"
            "{date} - 当前日期 YYYYMMDD"
        )
        self.ie_filename_edit.setStyleSheet(
            "QLineEdit { background: #3c3c3c; color: #eee; border: 1px solid #555; "
            "  border-radius: 3px; padding: 2px 6px; font-size: 11px; }"
            "QLineEdit:focus { border: 1px solid #2196F3; }"
        )
        fb_lay.addWidget(self.ie_filename_edit, stretch=1)

        v_lay.addWidget(source_bar)
        v_lay.addWidget(fname_bar)

        # 上下分割的图像查看器
        viewer_splitter = QSplitter(Qt.Vertical)
        viewer_splitter.setStyleSheet(
            "QSplitter::handle { background: #444; height: 2px; }"
        )

        # 源图区
        orig_frame = QFrame()
        orig_frame.setStyleSheet(
            "QFrame { background: #1a1a1a; border: 1px solid #333; border-radius: 4px; }"
        )
        of_lay = QVBoxLayout(orig_frame)
        of_lay.setContentsMargins(self._sp(2), self._sp(2), self._sp(2), self._sp(2))

        orig_header = QLabel("📷 源图像")
        orig_header.setStyleSheet(
            "color: #4CAF50; font-size: 12px; font-weight: bold; "
            "padding: 2px 6px; background: #2b2b2b; border-radius: 2px; border: none;"
        )
        of_lay.addWidget(orig_header)

        self.ie_orig_label = _IEImageViewer("（未加载）", self._sp)
        self.ie_orig_label.setMinimumSize(self._sp(300), self._sp(200))
        self.ie_orig_label.setStyleSheet(
            "border: 1px solid #333; background: #111; color: #666; font-size: 13px;"
        )
        of_lay.addWidget(self.ie_orig_label, stretch=1)

        # 坐标/像素值信息条
        self.ie_orig_info_bar = QLabel("")
        self.ie_orig_info_bar.setFixedHeight(self._sp(18))
        self.ie_orig_info_bar.setStyleSheet(
            "color: #aaa; font-size: 10px; background: #222; "
            "padding: 0 6px; border: none;"
        )
        # 鼠标移动时更新坐标信息
        self.ie_orig_label.pixel_hover.connect(
            lambda x, y, v: self.ie_orig_info_bar.setText(
                f"  坐标: ({x}, {y})    像素值: {v}" if x >= 0 else "")
        )
        of_lay.addWidget(self.ie_orig_info_bar)

        # 颜色条
        self.ie_orig_colorbar = _IEColorbarWidget(self._sp)
        self.ie_orig_colorbar.setFixedHeight(self._sp(20))
        of_lay.addWidget(self.ie_orig_colorbar)

        viewer_splitter.addWidget(orig_frame)

        # 结果图区
        result_frame = QFrame()
        result_frame.setStyleSheet(
            "QFrame { background: #1a1a1a; border: 1px solid #333; border-radius: 4px; }"
        )
        rf_lay = QVBoxLayout(result_frame)
        rf_lay.setContentsMargins(self._sp(2), self._sp(2), self._sp(2), self._sp(2))

        result_header = QLabel("🔬 处理结果")
        result_header.setStyleSheet(
            "color: #2196F3; font-size: 12px; font-weight: bold; "
            "padding: 2px 6px; background: #2b2b2b; border-radius: 2px; border: none;"
        )
        rf_lay.addWidget(result_header)

        self.ie_preview_label = _IEImageViewer("（未处理）", self._sp)
        self.ie_preview_label.setMinimumSize(self._sp(300), self._sp(200))
        self.ie_preview_label.setStyleSheet(
            "border: 1px solid #333; background: #111; color: #666; font-size: 13px;"
        )
        rf_lay.addWidget(self.ie_preview_label, stretch=1)

        # 坐标/像素值信息条
        self.ie_result_info_bar = QLabel("")
        self.ie_result_info_bar.setFixedHeight(self._sp(18))
        self.ie_result_info_bar.setStyleSheet(
            "color: #aaa; font-size: 10px; background: #222; "
            "padding: 0 6px; border: none;"
        )
        self.ie_preview_label.pixel_hover.connect(
            lambda x, y, v: self.ie_result_info_bar.setText(
                f"  坐标: ({x}, {y})    像素值: {v}" if x >= 0 else "")
        )
        rf_lay.addWidget(self.ie_result_info_bar)

        # 颜色条
        self.ie_result_colorbar = _IEColorbarWidget(self._sp)
        self.ie_result_colorbar.setFixedHeight(self._sp(20))
        rf_lay.addWidget(self.ie_result_colorbar)

        viewer_splitter.addWidget(result_frame)

        viewer_splitter.setSizes([self._sp(300), self._sp(300)])
        v_lay.addWidget(viewer_splitter, stretch=1)

        # 帧导航栏（批量模式浏览多帧）
        frame_nav = QFrame()
        frame_nav.setStyleSheet(
            "QFrame { background: #2b2b2b; border-radius: 3px; border: none; }"
            "QPushButton { background: #3c3c3c; color: #ddd; border: 1px solid #555; "
            "  border-radius: 3px; padding: 2px 8px; font-size: 11px; min-width: 24px; }"
            "QPushButton:hover { background: #505050; }"
            "QPushButton:disabled { background: #333; color: #555; }"
            "QLabel { color: #aaa; font-size: 11px; border: none; }"
            "QSlider::groove:horizontal { background: #555; height: 4px; }"
            "QSlider::handle:horizontal { background: #2196F3; width: 12px; "
            "  margin: -4px 0; border-radius: 6px; }"
        )
        fn_lay = QHBoxLayout(frame_nav)
        fn_lay.setContentsMargins(self._sp(6), self._sp(2), self._sp(6), self._sp(2))

        fn_lay.addWidget(QLabel("帧:"))
        self.ie_frame_first_btn = QPushButton("⏮")
        self.ie_frame_first_btn.clicked.connect(lambda: self._ie_navigate_frame('first'))
        fn_lay.addWidget(self.ie_frame_first_btn)

        self.ie_frame_prev_btn = QPushButton("◀")
        self.ie_frame_prev_btn.clicked.connect(lambda: self._ie_navigate_frame('prev'))
        fn_lay.addWidget(self.ie_frame_prev_btn)

        self.ie_frame_slider = QSlider(Qt.Horizontal)
        self.ie_frame_slider.setRange(0, 0)
        self.ie_frame_slider.setValue(0)
        self.ie_frame_slider.valueChanged.connect(self._ie_on_frame_slider)
        fn_lay.addWidget(self.ie_frame_slider, stretch=1)

        self.ie_frame_next_btn = QPushButton("▶")
        self.ie_frame_next_btn.clicked.connect(lambda: self._ie_navigate_frame('next'))
        fn_lay.addWidget(self.ie_frame_next_btn)

        self.ie_frame_last_btn = QPushButton("⏭")
        self.ie_frame_last_btn.clicked.connect(lambda: self._ie_navigate_frame('last'))
        fn_lay.addWidget(self.ie_frame_last_btn)

        self.ie_frame_label = QLabel("0 / 0")
        self.ie_frame_label.setFixedWidth(self._sp(70))
        self.ie_frame_label.setAlignment(Qt.AlignCenter)
        fn_lay.addWidget(self.ie_frame_label)

        # 播放按钮
        self.ie_frame_play_btn = QPushButton("▶ 播放")
        self.ie_frame_play_btn.setCheckable(True)
        self.ie_frame_play_btn.clicked.connect(self._ie_toggle_frame_play)
        fn_lay.addWidget(self.ie_frame_play_btn)

        # FPS
        self.ie_fps_spin = QSpinBox()
        self.ie_fps_spin.setRange(1, 60)
        self.ie_fps_spin.setValue(10)
        self.ie_fps_spin.setFixedWidth(self._sp(45))
        self.ie_fps_spin.setStyleSheet(
            "QSpinBox { background: #3c3c3c; color: #eee; border: 1px solid #555; "
            "  border-radius: 3px; padding: 1px 3px; }"
        )
        fn_lay.addWidget(QLabel("FPS:"))
        fn_lay.addWidget(self.ie_fps_spin)

        v_lay.addWidget(frame_nav)

        # 帧播放定时器
        self._ie_frame_timer = QTimer(self)
        self._ie_frame_timer.setInterval(100)
        self._ie_frame_timer.timeout.connect(lambda: self._ie_navigate_frame('next'))

        # 批量帧列表
        self._ie_batch_files = []

        # 进度条
        self.ie_progress_bar = QProgressBar()
        self.ie_progress_bar.setValue(0)
        self.ie_progress_bar.setVisible(False)
        self.ie_progress_bar.setStyleSheet(
            "QProgressBar { background: #2b2b2b; border: none; border-radius: 2px; "
            "  text-align: center; color: #ddd; height: 16px; }"
            "QProgressBar::chunk { background: #4CAF50; border-radius: 2px; }"
        )
        v_lay.addWidget(self.ie_progress_bar)

        # 日志区
        self.ie_log = QTextEdit()
        self.ie_log.setReadOnly(True)
        self.ie_log.setPlaceholderText("操作日志...")
        self.ie_log.setMaximumHeight(self._sp(100))
        self.ie_log.setStyleSheet(
            "QTextEdit { background: #1a1a1a; color: #aaa; border: 1px solid #333; "
            "  border-radius: 4px; font-size: 11px; padding: 4px; }"
        )
        v_lay.addWidget(self.ie_log)

        return viewer

    # ===== 工作流节点管理 =====

    def _ie_init_workflow_nodes(self, _IE):
        """初始化默认工作流：数据源节点。"""
        # 添加数据源节点
        self._ie_add_data_source_node()
        # 默认添加所有处理节点
        for step_key in _IE.ALL_STEPS:
            self._ie_add_node(step_key, request_preview=False)
        # 触发一次预览
        self._ie_request_preview()

    def _ie_add_data_source_node(self):
        """添加数据源节点（绿色标题栏）。"""
        node = QFrame()
        node.setStyleSheet(
            "QFrame { background: #2b2b2b; border: 1px solid #4CAF50; "
            "  border-radius: 6px; }"
        )
        n_lay = QVBoxLayout(node)
        n_lay.setContentsMargins(0, 0, 0, 0)
        n_lay.setSpacing(0)

        # 绿色标题栏
        title_bar = QFrame()
        title_bar.setStyleSheet(
            "QFrame { background: #4CAF50; border-radius: 5px 5px 0 0; }"
        )
        tb_lay = QHBoxLayout(title_bar)
        tb_lay.setContentsMargins(self._sp(8), self._sp(4), self._sp(8), self._sp(4))

        icon = QLabel("📷")
        icon.setStyleSheet("border: none; font-size: 14px;")
        tb_lay.addWidget(icon)

        title_lbl = QLabel("<b>数据源</b>")
        title_lbl.setStyleSheet("color: white; font-size: 12px; border: none;")
        tb_lay.addWidget(title_lbl)
        tb_lay.addStretch()
        n_lay.addWidget(title_bar)

        # 内容区
        body = QWidget()
        b_lay = QVBoxLayout(body)
        b_lay.setContentsMargins(self._sp(8), self._sp(6), self._sp(8), self._sp(6))
        b_lay.setSpacing(self._sp(2))

        self.ie_source_info = QLabel("未加载图像")
        self.ie_source_info.setStyleSheet("color: #999; font-size: 11px; border: none;")
        b_lay.addWidget(self.ie_source_info)

        self.ie_source_dim = QLabel("")
        self.ie_source_dim.setStyleSheet("color: #888; font-size: 10px; border: none;")
        b_lay.addWidget(self.ie_source_dim)

        n_lay.addWidget(body)

        # 插入到画布布局（stretch之前）
        insert_idx = self.ie_canvas_layout.count() - 1  # 最后一个是stretch
        self.ie_canvas_layout.insertWidget(insert_idx, node)
        self._ie_data_source_node = node

        # 添加箭头
        self._ie_add_arrow()

    def _ie_add_node(self, step_key, request_preview=True):
        """添加处理节点到工作流画布。"""
        if step_key in self._ie_workflow_nodes:
            return  # 已存在

        from utils.image_editor import ImageEditor as _IE
        label = _IE.STEP_LABELS.get(step_key, step_key)

        node = QFrame()
        node.setProperty("step_key", step_key)
        node.setStyleSheet(
            "QFrame { background: #2b2b2b; border: 1px solid #2196F3; "
            "  border-radius: 6px; }"
        )
        n_lay = QVBoxLayout(node)
        n_lay.setContentsMargins(0, 0, 0, 0)
        n_lay.setSpacing(0)

        # 蓝色标题栏
        title_bar = QFrame()
        title_bar.setStyleSheet(
            "QFrame { background: #2196F3; border-radius: 5px 5px 0 0; }"
        )
        tb_lay = QHBoxLayout(title_bar)
        tb_lay.setContentsMargins(self._sp(8), self._sp(4), self._sp(8), self._sp(4))

        # 启用/禁用 checkbox
        enable_check = QCheckBox()
        enable_check.setChecked(True)
        enable_check.setStyleSheet(
            "QCheckBox { border: none; spacing: 0; }"
            "QCheckBox::indicator { width: 14px; height: 14px; }"
            "QCheckBox::indicator:unchecked { background: #555; border: 1px solid #777; border-radius: 2px; }"
            "QCheckBox::indicator:checked { background: #4CAF50; border: 1px solid #4CAF50; border-radius: 2px; }"
        )
        enable_check.stateChanged.connect(lambda s, k=step_key: self._ie_toggle_node(k, s))
        tb_lay.addWidget(enable_check)

        title_lbl = QLabel(f"<b>{label}</b>")
        title_lbl.setStyleSheet("color: white; font-size: 12px; border: none;")
        tb_lay.addWidget(title_lbl)
        tb_lay.addStretch()

        # 删除按钮
        del_btn = QPushButton("✕")
        del_btn.setFixedSize(self._sp(20), self._sp(20))
        del_btn.setStyleSheet(
            "QPushButton { background: transparent; color: #ccc; border: none; "
            "  font-size: 14px; font-weight: bold; }"
            "QPushButton:hover { color: #ff5252; }"
        )
        del_btn.clicked.connect(lambda _, k=step_key: self._ie_remove_node(k))
        tb_lay.addWidget(del_btn)

        n_lay.addWidget(title_bar)

        # 内容区
        body = QWidget()
        b_lay = QVBoxLayout(body)
        b_lay.setContentsMargins(self._sp(8), self._sp(6), self._sp(8), self._sp(6))
        b_lay.setSpacing(self._sp(2))

        # 参数显示（简要）
        param_summary = self._ie_get_param_summary(step_key)
        param_lbl = QLabel(param_summary)
        param_lbl.setStyleSheet("color: #bbb; font-size: 11px; border: none;")
        param_lbl.setWordWrap(True)
        param_lbl.setProperty("param_summary", True)
        b_lay.addWidget(param_lbl)

        # 输出名
        output_row = QHBoxLayout()
        output_row.setSpacing(self._sp(4))
        output_lbl = QLabel("输出:")
        output_lbl.setStyleSheet("color: #888; font-size: 10px; border: none;")
        output_row.addWidget(output_lbl)
        output_name = QLabel(f"img_{step_key}")
        output_name.setStyleSheet(
            "color: #4CAF50; font-size: 10px; border: none; "
            "background: #1e1e1e; padding: 1px 4px; border-radius: 2px;"
        )
        output_row.addWidget(output_name)
        output_row.addStretch()
        b_lay.addLayout(output_row)

        # 进度条
        progress = QProgressBar()
        progress.setValue(0)
        progress.setFixedHeight(self._sp(4))
        progress.setStyleSheet(
            "QProgressBar { background: #1e1e1e; border: none; border-radius: 2px; }"
            "QProgressBar::chunk { background: #4CAF50; border-radius: 2px; }"
        )
        progress.setVisible(False)
        b_lay.addWidget(progress)

        # 编辑参数按钮
        edit_btn = QPushButton("编辑参数...")
        edit_btn.setStyleSheet(
            "QPushButton { background: #383838; color: #ddd; border: 1px solid #555; "
            "  border-radius: 3px; padding: 2px 8px; font-size: 11px; }"
            "QPushButton:hover { background: #505050; }"
        )
        edit_btn.clicked.connect(lambda _, k=step_key: self._ie_edit_node_params(k))
        b_lay.addWidget(edit_btn)

        n_lay.addWidget(body)

        # 插入到画布（stretch之前，箭头之后）
        insert_idx = self.ie_canvas_layout.count() - 1
        self.ie_canvas_layout.insertWidget(insert_idx, node)

        # 添加箭头
        self._ie_add_arrow()

        self._ie_workflow_nodes[step_key] = {
            "widget": node,
            "enable_check": enable_check,
            "progress": progress,
            "param_label": param_lbl,
        }

        if request_preview:
            self._ie_request_preview()

    def _ie_remove_node(self, step_key):
        """从工作流移除节点。"""
        if step_key not in self._ie_workflow_nodes:
            return

        info = self._ie_workflow_nodes.pop(step_key)
        widget = info["widget"]
        # 找到widget在layout中的index，同时移除其后的箭头
        idx = self.ie_canvas_layout.indexOf(widget)
        if idx >= 0:
            # 移除箭头（在节点之后，stretch之前）
            arrow_idx = idx + 1
            if arrow_idx < self.ie_canvas_layout.count():
                arrow_item = self.ie_canvas_layout.itemAt(arrow_idx)
                if arrow_item and arrow_item.widget():
                    arrow_item.widget().deleteLater()

        widget.deleteLater()
        self._ie_request_preview()

    def _ie_toggle_node(self, step_key, state):
        """启用/禁用节点。"""
        if step_key not in self._ie_workflow_nodes:
            return
        info = self._ie_workflow_nodes[step_key]
        enabled = (state == Qt.Checked)
        # 视觉反馈：改变边框颜色
        border_color = "#2196F3" if enabled else "#555"
        opacity = "1.0" if enabled else "0.5"
        info["widget"].setStyleSheet(
            f"QFrame {{ background: #2b2b2b; border: 1px solid {border_color}; "
            f"  border-radius: 6px; opacity: {opacity}; }}"
        )
        self._ie_request_preview()

    def _ie_add_arrow(self):
        """在工作流中添加向下箭头。"""
        arrow = QLabel("▼")
        arrow.setAlignment(Qt.AlignCenter)
        arrow.setFixedHeight(self._sp(18))
        arrow.setStyleSheet(
            "color: #666; font-size: 14px; background: transparent; border: none;"
        )
        insert_idx = self.ie_canvas_layout.count() - 1
        self.ie_canvas_layout.insertWidget(insert_idx, arrow)

    def _ie_get_param_summary(self, step_key):
        """获取步骤参数的简要描述。"""
        cfg = self.ie_config
        summaries = {
            "crop": f"ROI: ({cfg.crop.x}, {cfg.crop.y}) {cfg.crop.w}×{cfg.crop.h}",
            "gray": "彩色 → 8-bit 灰度",
            "mirror": f"模式: {['水平', '垂直', '水平+垂直'][['horizontal', 'vertical', 'both'].index(cfg.mirror.mode)]}",
            "rotate": f"方式: {cfg.rotate.mode}" + (f" {cfg.rotate.angle}°" if cfg.rotate.mode == 'custom' else ''),
            "bit_depth": f"源位深: {cfg.bit_depth.source_bits or '自动'}",
            "gray_math": f"运算: {cfg.gray_math.operation}" + (f" k={cfg.gray_math.kernel_size}" if cfg.gray_math.operation == 'average' else ''),
            "bc": f"α={cfg.bc.alpha:.1f}, β={cfg.bc.beta}",
            "arithmetic": f"{cfg.arithmetic.operation}" + (f" val={cfg.arithmetic.scalar_value}" if not cfg.arithmetic.operand_path else " 双图"),
            "threshold": f"模式: {cfg.threshold.mode} T={cfg.threshold.threshold_value}",
        }
        return summaries.get(step_key, "")

    def _ie_edit_node_params(self, step_key):
        """弹出参数编辑对话框。"""
        from PyQt5.QtWidgets import QDialog, QDialogButtonBox
        from utils.image_editor import ImageEditor as _IE

        dialog = QDialog(self)
        dialog.setWindowTitle(f"编辑参数 — {_IE.STEP_LABELS.get(step_key, step_key)}")
        dialog.setMinimumWidth(self._sp(360))
        dialog.setStyleSheet(
            "QDialog { background: #2b2b2b; }"
            "QLabel { color: #ccc; }"
            "QGroupBox { color: #aaa; border: 1px solid #444; "
            "  margin-top: 12px; padding-top: 16px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }"
            "QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox { "
            "  background: #3c3c3c; color: #eee; border: 1px solid #555; "
            "  border-radius: 3px; padding: 3px 6px; }"
            "QCheckBox { color: #ccc; }"
            "QSlider::groove:horizontal { background: #555; height: 4px; }"
            "QSlider::handle:horizontal { background: #2196F3; width: 12px; "
            "  margin: -4px 0; border-radius: 6px; }"
        )

        d_lay = QVBoxLayout(dialog)

        # 构建参数控件（复用现有逻辑）
        param_widget = QWidget()
        p_lay = QVBoxLayout(param_widget)

        if step_key == "crop":
            self._build_crop_param_widget(p_lay)
        elif step_key == "gray":
            p_lay.addWidget(QLabel("将图像转换为 8-bit 灰度图（无可调参数）"))
        elif step_key == "mirror":
            self._build_mirror_param_widget(p_lay)
        elif step_key == "rotate":
            self._build_rotate_param_widget(p_lay)
        elif step_key == "bit_depth":
            self._build_bit_depth_param_widget(p_lay)
        elif step_key == "gray_math":
            self._build_gray_math_param_widget(p_lay)
        elif step_key == "bc":
            self._build_bc_param_widget(p_lay)
        elif step_key == "arithmetic":
            self._build_arith_param_widget(p_lay)
        elif step_key == "threshold":
            self._build_threshold_param_widget(p_lay)

        p_lay.addStretch()
        d_lay.addWidget(param_widget)

        # 按钮
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.setStyleSheet(
            "QPushButton { background: #3c3c3c; color: #ddd; border: 1px solid #555; "
            "  border-radius: 3px; padding: 5px 16px; }"
            "QPushButton:hover { background: #505050; }"
        )
        btns.accepted.connect(dialog.accept)
        btns.rejected.connect(dialog.reject)
        d_lay.addWidget(btns)

        if dialog.exec_() == QDialog.Accepted:
            self._ie_sync_config_from_ui()
            # 更新节点参数摘要
            if step_key in self._ie_workflow_nodes:
                param_lbl = self._ie_workflow_nodes[step_key]["param_label"]
                param_lbl.setText(self._ie_get_param_summary(step_key))
            self._ie_request_preview()

    # ===== 参数构建器（对话框版本，与原Tab版共享控件） =====

    def _build_crop_param_widget(self, layout):
        labels = ["X (px):", "Y (px):", "宽 (px):", "高 (px):"]
        if not hasattr(self, 'ie_crop_spins'):
            self.ie_crop_spins = []
        for i, lbl in enumerate(labels):
            if i < len(self.ie_crop_spins):
                layout.addWidget(QLabel(lbl))
                layout.addWidget(self.ie_crop_spins[i])
            else:
                layout.addWidget(QLabel(lbl))
                sp = QSpinBox()
                sp.setRange(0, 99999)
                sp.valueChanged.connect(lambda _: self._ie_request_preview())
                layout.addWidget(sp)
                self.ie_crop_spins.append(sp)

    def _build_mirror_param_widget(self, layout):
        layout.addWidget(QLabel("镜像方向:"))
        if not hasattr(self, 'ie_mirror_combo'):
            self.ie_mirror_combo = QComboBox()
            self.ie_mirror_combo.addItems(["水平镜像", "垂直镜像", "水平+垂直"])
            self.ie_mirror_combo.currentIndexChanged.connect(lambda _: self._ie_request_preview())
        layout.addWidget(self.ie_mirror_combo)

    def _build_rotate_param_widget(self, layout):
        layout.addWidget(QLabel("旋转方式:"))
        if not hasattr(self, 'ie_rotate_mode_combo'):
            self.ie_rotate_mode_combo = QComboBox()
            self.ie_rotate_mode_combo.addItems(["顺时针 90°", "逆时针 90°", "180°", "自定义角度"])
            self.ie_rotate_mode_combo.currentIndexChanged.connect(self._ie_on_rotate_mode_changed)
        layout.addWidget(self.ie_rotate_mode_combo)
        layout.addWidget(QLabel("角度:"))
        if not hasattr(self, 'ie_rotate_angle_spin'):
            self.ie_rotate_angle_spin = QDoubleSpinBox()
            self.ie_rotate_angle_spin.setRange(-360.0, 360.0)
            self.ie_rotate_angle_spin.setSingleStep(1.0)
            self.ie_rotate_angle_spin.setEnabled(False)
            self.ie_rotate_angle_spin.valueChanged.connect(lambda _: self._ie_request_preview())
        layout.addWidget(self.ie_rotate_angle_spin)
        if not hasattr(self, 'ie_rotate_expand_check'):
            self.ie_rotate_expand_check = QCheckBox("自动扩展画布")
            self.ie_rotate_expand_check.setChecked(True)
            self.ie_rotate_expand_check.stateChanged.connect(lambda _: self._ie_request_preview())
        layout.addWidget(self.ie_rotate_expand_check)
        layout.addWidget(QLabel("边界灰度:"))
        if not hasattr(self, 'ie_rotate_border_spin'):
            self.ie_rotate_border_spin = QSpinBox()
            self.ie_rotate_border_spin.setRange(0, 255)
            self.ie_rotate_border_spin.valueChanged.connect(lambda _: self._ie_request_preview())
        layout.addWidget(self.ie_rotate_border_spin)

    def _build_bit_depth_param_widget(self, layout):
        layout.addWidget(QLabel("源位深:"))
        if not hasattr(self, 'ie_bit_depth_combo'):
            self.ie_bit_depth_combo = QComboBox()
            self.ie_bit_depth_combo.addItems(["自动识别", "24 位", "16 位", "12 位"])
            self.ie_bit_depth_combo.currentIndexChanged.connect(lambda _: self._ie_request_preview())
        layout.addWidget(self.ie_bit_depth_combo)

    def _build_gray_math_param_widget(self, layout):
        layout.addWidget(QLabel("计算方式:"))
        if not hasattr(self, 'ie_gray_math_combo'):
            self.ie_gray_math_combo = QComboBox()
            self.ie_gray_math_combo.addItems(["平均", "log", "exp", "sqrt", "sqr"])
            self.ie_gray_math_combo.currentIndexChanged.connect(self._ie_on_gray_math_changed)
        layout.addWidget(self.ie_gray_math_combo)
        layout.addWidget(QLabel("平均核大小:"))
        if not hasattr(self, 'ie_gray_math_kernel_spin'):
            self.ie_gray_math_kernel_spin = QSpinBox()
            self.ie_gray_math_kernel_spin.setRange(1, 99)
            self.ie_gray_math_kernel_spin.setSingleStep(2)
            self.ie_gray_math_kernel_spin.setValue(3)
            self.ie_gray_math_kernel_spin.valueChanged.connect(lambda _: self._ie_request_preview())
        layout.addWidget(self.ie_gray_math_kernel_spin)

    def _build_bc_param_widget(self, layout):
        layout.addWidget(QLabel("对比度 α:"))
        if not hasattr(self, 'ie_alpha_spin'):
            self.ie_alpha_spin = QDoubleSpinBox()
            self.ie_alpha_spin.setRange(0.1, 5.0)
            self.ie_alpha_spin.setSingleStep(0.1)
            self.ie_alpha_spin.setValue(1.0)
            self.ie_alpha_spin.valueChanged.connect(lambda _: self._ie_request_preview())
        layout.addWidget(self.ie_alpha_spin)
        layout.addWidget(QLabel("亮度 β:"))
        if not hasattr(self, 'ie_beta_spin'):
            self.ie_beta_spin = QSpinBox()
            self.ie_beta_spin.setRange(-255, 255)
            self.ie_beta_spin.setValue(0)
            self.ie_beta_spin.valueChanged.connect(lambda _: self._ie_request_preview())
        layout.addWidget(self.ie_beta_spin)
        layout.addWidget(QLabel("α 滑块:"))
        if not hasattr(self, 'ie_alpha_slider'):
            self.ie_alpha_slider = QSlider(Qt.Horizontal)
            self.ie_alpha_slider.setRange(1, 50)
            self.ie_alpha_slider.setValue(10)
            self.ie_alpha_slider.valueChanged.connect(self._ie_alpha_slider_moved)
        layout.addWidget(self.ie_alpha_slider)
        layout.addWidget(QLabel("β 滑块:"))
        if not hasattr(self, 'ie_beta_slider'):
            self.ie_beta_slider = QSlider(Qt.Horizontal)
            self.ie_beta_slider.setRange(-255, 255)
            self.ie_beta_slider.setValue(0)
            self.ie_beta_slider.valueChanged.connect(self._ie_beta_slider_moved)
        layout.addWidget(self.ie_beta_slider)

    def _build_arith_param_widget(self, layout):
        layout.addWidget(QLabel("操作:"))
        if not hasattr(self, 'ie_arith_combo'):
            self.ie_arith_combo = QComboBox()
            self.ie_arith_combo.addItems(["加法 (add)", "减法 (subtract)"])
            self.ie_arith_combo.currentIndexChanged.connect(lambda _: self._ie_request_preview())
        layout.addWidget(self.ie_arith_combo)
        layout.addWidget(QLabel("操作数来源:"))
        if not hasattr(self, 'ie_operand_src_combo'):
            self.ie_operand_src_combo = QComboBox()
            self.ie_operand_src_combo.addItems(["使用标量值", "使用第二张图像"])
            self.ie_operand_src_combo.currentIndexChanged.connect(self._ie_on_operand_src_changed)
        layout.addWidget(self.ie_operand_src_combo)
        layout.addWidget(QLabel("标量值:"))
        if not hasattr(self, 'ie_scalar_spin'):
            self.ie_scalar_spin = QSpinBox()
            self.ie_scalar_spin.setRange(0, 255)
            self.ie_scalar_spin.setValue(0)
            self.ie_scalar_spin.valueChanged.connect(lambda _: self._ie_request_preview())
        layout.addWidget(self.ie_scalar_spin)

    def _build_threshold_param_widget(self, layout):
        layout.addWidget(QLabel("模式:"))
        if not hasattr(self, 'ie_thr_mode_combo'):
            self.ie_thr_mode_combo = QComboBox()
            self.ie_thr_mode_combo.addItems([
                "全局阈值 (Global)", "大津法 (Otsu)",
                "自适应均值 (Adaptive Mean)", "自适应高斯 (Adaptive Gaussian)"
            ])
            self.ie_thr_mode_combo.currentIndexChanged.connect(self._ie_on_thr_mode_changed)
        layout.addWidget(self.ie_thr_mode_combo)
        layout.addWidget(QLabel("阈值:"))
        if not hasattr(self, 'ie_thr_val_spin'):
            self.ie_thr_val_spin = QSpinBox()
            self.ie_thr_val_spin.setRange(0, 255)
            self.ie_thr_val_spin.setValue(128)
            self.ie_thr_val_spin.valueChanged.connect(lambda _: self._ie_request_preview())
        layout.addWidget(self.ie_thr_val_spin)
        if not hasattr(self, 'ie_thr_val_slider'):
            self.ie_thr_val_slider = QSlider(Qt.Horizontal)
            self.ie_thr_val_slider.setRange(0, 255)
            self.ie_thr_val_slider.setValue(128)
            self.ie_thr_val_slider.valueChanged.connect(self._ie_thr_slider_moved)
        layout.addWidget(self.ie_thr_val_slider)
        layout.addWidget(QLabel("最大值:"))
        if not hasattr(self, 'ie_thr_max_spin'):
            self.ie_thr_max_spin = QSpinBox()
            self.ie_thr_max_spin.setRange(1, 255)
            self.ie_thr_max_spin.setValue(255)
            self.ie_thr_max_spin.valueChanged.connect(lambda _: self._ie_request_preview())
        layout.addWidget(self.ie_thr_max_spin)

    # ===== 工作流辅助方法 =====

    def _ie_filter_ops(self, text):
        """搜索过滤操作面板算法卡片。"""
        text = text.lower()
        for key, card in self._ie_algo_cards.items():
            from utils.image_editor import ImageEditor as _IE
            label = _IE.STEP_LABELS.get(key, key).lower()
            visible = text in label or text in key.lower()
            card.setVisible(visible)

    def _ie_reset_workflow(self):
        """重置工作流到默认状态。"""
        # 移除所有处理节点
        keys = list(self._ie_workflow_nodes.keys())
        for key in keys:
            self._ie_remove_node(key)

        # 移除数据源节点
        if hasattr(self, '_ie_data_source_node'):
            self._ie_data_source_node.deleteLater()

        # 清空画布
        while self.ie_canvas_layout.count() > 0:
            item = self.ie_canvas_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self.ie_canvas_layout.addStretch()
        self._ie_workflow_nodes.clear()

        # 重新初始化
        from utils.image_editor import ImageEditor as _IE
        self._ie_init_workflow_nodes(_IE)

    def _ie_zoom(self, direction):
        """缩放查看器图像。direction: 1=放大, -1=缩小, 0=适应窗口。"""
        if direction == 0:
            self._ie_zoom_level = 100
        else:
            self._ie_zoom_level = max(25, min(400, self._ie_zoom_level + direction * 25))
        self.ie_zoom_label.setText(f"{self._ie_zoom_level}%")
        # 重新刷新预览
        self._ie_request_preview()

    def _ie_get_step_order(self):
        """从工作流节点获取已启用的步骤 key 列表（保持画布顺序）。"""
        order = []
        for i in range(self.ie_canvas_layout.count()):
            item = self.ie_canvas_layout.itemAt(i)
            if item and item.widget():
                widget = item.widget()
                step_key = widget.property("step_key")
                if step_key and step_key in self._ie_workflow_nodes:
                    info = self._ie_workflow_nodes[step_key]
                    if info["enable_check"].isChecked():
                        order.append(step_key)
        return order

    def _ie_move_node(self, step_key, new_idx):
        """移动工作流节点到新位置。"""
        if step_key not in self._ie_workflow_nodes:
            return
        info = self._ie_workflow_nodes[step_key]
        widget = info["widget"]
        cur_idx = self.ie_canvas_layout.indexOf(widget)
        if cur_idx < 0 or cur_idx == new_idx:
            return

        # 找到对应的箭头（节点后面紧跟的箭头）
        arrow_idx = cur_idx + 1

        # 移除节点和箭头
        self.ie_canvas_layout.removeWidget(widget)
        if arrow_idx < self.ie_canvas_layout.count():
            arrow_item = self.ie_canvas_layout.itemAt(arrow_idx)
            if arrow_item and arrow_item.widget():
                arrow_widget = arrow_item.widget()
                self.ie_canvas_layout.removeWidget(arrow_widget)
            else:
                arrow_widget = None
        else:
            arrow_widget = None

        # 重新插入（new_idx 可能因为移除而偏移）
        target_idx = min(new_idx, self.ie_canvas_layout.count() - 1)  # -1 for stretch
        self.ie_canvas_layout.insertWidget(target_idx, widget)
        if arrow_widget:
            self.ie_canvas_layout.insertWidget(target_idx + 1, arrow_widget)

        self._ie_request_preview()

    # ===== 帧导航 =====

    def _ie_navigate_frame(self, direction):
        """帧导航：first/prev/next/last。"""
        n = len(self._ie_batch_files)
        if n == 0:
            return
        cur = self.ie_frame_slider.value()
        if direction == 'first':
            cur = 0
        elif direction == 'prev':
            cur = max(0, cur - 1)
        elif direction == 'next':
            cur = min(n - 1, cur + 1)
        elif direction == 'last':
            cur = n - 1
        self.ie_frame_slider.setValue(cur)

    def _ie_on_frame_slider(self, idx):
        """帧滑块变化时加载对应帧。"""
        n = len(self._ie_batch_files)
        if n == 0 or idx >= n:
            return
        path = self._ie_batch_files[idx]
        self.ie_single_path = path
        if hasattr(self, 'ie_single_label'):
            self.ie_single_label.setText(os.path.basename(path))
        self._ie_load_orig_preview(path)
        self._ie_request_preview()
        self.ie_frame_label.setText(f"{idx + 1} / {n}")

    def _ie_toggle_frame_play(self, checked):
        """播放/暂停帧动画。"""
        if checked:
            fps = self.ie_fps_spin.value()
            self._ie_frame_timer.setInterval(max(16, int(1000 / fps)))
            self._ie_frame_timer.start()
            self.ie_frame_play_btn.setText("⏸ 暂停")
        else:
            self._ie_frame_timer.stop()
            self.ie_frame_play_btn.setText("▶ 播放")

    def _ie_update_frame_nav(self):
        """当帧列表变化时更新帧导航控件。"""
        n = len(self._ie_batch_files)
        self.ie_frame_slider.setRange(0, max(0, n - 1))
        self.ie_frame_slider.setValue(0)
        self.ie_frame_label.setText(f"{'1' if n > 0 else '0'} / {n}")
        enabled = n > 0
        self.ie_frame_first_btn.setEnabled(enabled)
        self.ie_frame_prev_btn.setEnabled(enabled)
        self.ie_frame_next_btn.setEnabled(enabled)
        self.ie_frame_last_btn.setEnabled(enabled)
        self.ie_frame_slider.setEnabled(enabled)
        self.ie_frame_play_btn.setEnabled(enabled)

    # ------------------------------------------------------------
    # Image editor slots
    # ------------------------------------------------------------

    def _ie_on_mode_changed(self, idx):
        """切换单张/批量模式"""
        single = (idx == 0)
        self.ie_single_widget.setVisible(single)
        self.ie_batch_widget.setVisible(not single)
        # 更新工具栏运行按钮行为
        if single:
            self.ie_run_btn.setText("▶ 预览")
        else:
            self.ie_run_btn.setText("▶ 批量处理")

    # 旧的 step_list/param_tabs 相关 slot 已移除（DaVis工作流不需要）

    def _ie_on_rotate_mode_changed(self, idx):
        self.ie_rotate_angle_spin.setEnabled(idx == 3)
        self._ie_request_preview()

    def _ie_on_gray_math_changed(self, idx):
        self.ie_gray_math_kernel_spin.setEnabled(idx == 0)
        self._ie_request_preview()

    def _ie_on_operand_src_changed(self, idx):
        use_file = (idx == 1)
        self.ie_operand_widget.setVisible(use_file)
        self.ie_scalar_spin.setEnabled(not use_file)
        self._ie_request_preview()

    def _ie_on_thr_mode_changed(self, idx):
        is_adapt = (idx >= 2)
        self.ie_thr_val_spin.setEnabled(not is_adapt)
        self.ie_thr_val_slider.setEnabled(not is_adapt)
        self.ie_adapt_widget.setVisible(is_adapt)
        self._ie_request_preview()

    def _ie_alpha_slider_moved(self, v):
        """α 滑块联动 SpinBox"""
        self.ie_alpha_spin.blockSignals(True)
        self.ie_alpha_spin.setValue(v / 10.0)
        self.ie_alpha_spin.blockSignals(False)
        self._ie_request_preview()

    def _ie_beta_slider_moved(self, v):
        """β 滑块联动 SpinBox"""
        self.ie_beta_spin.blockSignals(True)
        self.ie_beta_spin.setValue(v)
        self.ie_beta_spin.blockSignals(False)
        self._ie_request_preview()

    def _ie_thr_slider_moved(self, v):
        """阈值滑块联动 SpinBox。"""
        self.ie_thr_val_spin.blockSignals(True)
        self.ie_thr_val_spin.setValue(v)
        self.ie_thr_val_spin.blockSignals(False)
        self._ie_request_preview()

    def _ie_request_preview(self):
        """防抖：重启 300ms 定时器，到期后执行预览。"""
        self._ie_preview_timer.start()

    def _ie_open_single(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择图像", "",
            "图像文件 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)")
        if path:
            self.ie_single_path = path
            self.ie_single_label.setText(os.path.basename(path))
            self.ie_single_label.setStyleSheet("")
            self._ie_load_orig_preview(path)
            self._ie_request_preview()

    def _ie_load_orig_preview(self, path):
        """在原图区域显示图像，同时更新工作流数据源节点信息。"""
        import cv2 as _cv2
        img = robust_imread(path, _cv2.IMREAD_UNCHANGED)
        if img is None:
            return
        h, w = img.shape[:2]
        self.ie_log.append(f"已加载 {os.path.basename(path)}  ({w}x{h})")
        # 更新数据源节点信息
        if hasattr(self, 'ie_source_info'):
            self.ie_source_info.setText(os.path.basename(path))
            self.ie_source_info.setStyleSheet("color: #4CAF50; font-size: 11px; border: none;")
        if hasattr(self, 'ie_source_dim'):
            self.ie_source_dim.setText(f"{w} × {h}  |  {img.dtype}  |  {img.ndim}D")
        pixmap = self._ie_ndarray_to_pixmap(img)
        if pixmap:
            self.ie_orig_label.setPixmap(
                pixmap.scaled(self.ie_orig_label.size(),
                              Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _ie_pick_src_dir(self):
        d = QFileDialog.getExistingDirectory(self, "选择输入目录")
        if d:
            self.ie_src_dir = d
            self.ie_src_label.setText(d)
            self.ie_src_label.setStyleSheet("color: #ddd;")
            from utils.image_editor import SUPPORTED_EXTS
            self._ie_batch_files = sorted(
                str(f) for f in Path(d).iterdir()
                if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS
            )
            cnt = len(self._ie_batch_files)
            self.ie_log.append(f"输入目录: {d}  (共 {cnt} 张图像)")
            self._ie_update_frame_nav()
            # 自动加载第一帧
            if cnt > 0:
                self.ie_single_path = self._ie_batch_files[0]
                self.ie_single_label.setText(os.path.basename(self.ie_single_path))
                self._ie_load_orig_preview(self.ie_single_path)

    def _ie_pick_dst_dir(self):
        d = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if not d:
            d = QFileDialog.getSaveFileName(self, "创建输出目录",
                                             "", "目录")[0]
        if d:
            self.ie_dst_dir = d
            self.ie_dst_label.setText(d)
            self.ie_dst_label.setStyleSheet("color: #ddd;")
            self.ie_log.append(f"输出目录: {d}")

    def _ie_pick_operand(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择第二张图", "",
            "图像文件 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)")
        if path:
            self.ie_operand_path = path
            self.ie_operand_label.setText(os.path.basename(path))
            self._ie_request_preview()

    # 旧 _ie_get_step_order（基于 ie_step_list）已移除，新版在工作流画布上方定义


    # _ie_sync_config_from_ui 已废弃——新工作流通过对话框直接更新 ie_config，
    # 不再需要从 UI 控件同步。保留空方法以避免其他地方调用时出错。
    def _ie_sync_config_from_ui(self):
        """已废弃：新工作流通过对话框直接更新 ie_config，无需从 UI 同步。"""
        pass

    # 旧的 _ie_sync_config_from_ui 方法已删除（引用了已不存在的旧版 UI 控件）
    # 如果将来需要，应从 self.ie_config 属性中读取参数。

    def _ie_do_preview(self):
        """处理（由"处理"按钮触发，不再自动预览）。"""
        import cv2 as _cv2
        if not self.ie_single_path or not os.path.isfile(self.ie_single_path):
            return

        step_order = self._ie_get_step_order()
        if not step_order:
            self._ie_current_preview = None
            return

        # 不再调用 _ie_sync_config_from_ui()，
        # 因为新工作流通过对话框直接更新 self.ie_config
        try:
            img = robust_imread(self.ie_single_path, _cv2.IMREAD_UNCHANGED)
            op_img = None
            if hasattr(self, 'ie_config') and self.ie_config.arithmetic.operand_path:
                op_img = robust_imread(self.ie_config.arithmetic.operand_path,
                                     _cv2.IMREAD_UNCHANGED)
            editor = ImageEditor(self.ie_config)
            result = editor.process(img, op_img, step_order=step_order)
            self._ie_current_preview = result
            pixmap = self._ie_ndarray_to_pixmap(result)
            if pixmap:
                self.ie_preview_label.setPixmap(
                    pixmap.scaled(self.ie_preview_label.size(),
                                  Qt.KeepAspectRatio, Qt.SmoothTransformation))
            h, w = result.shape[:2]

            # 更新颜色条
            if hasattr(self, 'ie_result_colorbar'):
                vmin, vmax = float(result.min()), float(result.max())
                self.ie_result_colorbar.set_range(vmin, vmax)
            if hasattr(self, 'ie_orig_colorbar'):
                vmin, vmax = float(img.min()), float(img.max())
                self.ie_orig_colorbar.set_range(vmin, vmax)

            # 更新 image viewer 的原始图像数据（用于鼠标悬停像素值读取）
            self.ie_preview_label.set_image_data(result)
            self.ie_orig_label.set_image_data(img)

            steps_str = " -> ".join(
                ImageEditor.STEP_LABELS.get(s, s) for s in step_order)
            self.ie_log.append(f"处理完成: {steps_str}  输出: {w}x{h}")
        except Exception as e:
            self.ie_log.append(f"处理失败: {e}")

    def _ie_resolve_filename(self, original_path, index=None, step_order=None):
        """根据输出文件名模式生成实际文件名。"""
        import datetime
        original = Path(original_path)
        name = original.stem
        ext = original.suffix

        pattern = self.ie_filename_edit.text().strip() or "{original}_processed"

        date_str = datetime.datetime.now().strftime("%Y%m%d")
        step_str = ""
        if step_order:
            step_str = "_".join(
                ImageEditor.STEP_LABELS.get(s, s) for s in step_order
            )[:20]  # 防止文件名过长

        result = pattern
        result = result.replace("{original}", name)
        result = result.replace("{index}", f"{index:04d}" if index is not None else "0000")
        result = result.replace("{step}", step_str)
        result = result.replace("{date}", date_str)

        # 确保有扩展名
        if not Path(result).suffix:
            result += ext
        return result

    def _ie_save_single(self):
        """保存单张处理结果（使用文件名模式）。"""
        import cv2 as _cv2
        if self._ie_current_preview is None:
            QMessageBox.warning(self, "提示", "请先点击\"处理\"生成结果")
            return
        # 使用文件名模式自动生成路径
        if self.ie_single_path:
            out_name = self._ie_resolve_filename(self.ie_single_path)
            out_dir = self.ie_dst_dir or os.path.dirname(self.ie_single_path)
            os.makedirs(out_dir, exist_ok=True)
            path = os.path.join(out_dir, out_name)
        else:
            path, _ = QFileDialog.getSaveFileName(
                self, "保存处理结果", "",
                "PNG (*.png);;JPEG (*.jpg);;BMP (*.bmp);;TIFF (*.tif)")
            if not path:
                return

        try:
            _cv2.imwrite(path, self._ie_current_preview)
            self.ie_log.append(f"已保存: {os.path.basename(path)}")
            # 保存后刷新文件树
            self._file_tree_panel.watch_extra_dir(os.path.dirname(path))
            self._file_tree_panel.refresh()
            QMessageBox.information(self, "完成", f"已保存至:\n{path}")
        except Exception as e:
            QMessageBox.warning(self, "保存失败", str(e))

    def _ie_run_batch(self):
        """启动批量处理（自动创建输出文件夹）。"""
        if not self.ie_src_dir or not os.path.isdir(self.ie_src_dir):
            QMessageBox.warning(self, "提示", "请先选择输入目录")
            return

        # 自动创建输出文件夹：若未设置则使用 {输入目录}/processed
        if not self.ie_dst_dir:
            self.ie_dst_dir = os.path.join(self.ie_src_dir, "processed")
            self.ie_dst_label.setText(self.ie_dst_dir)
            self.ie_log.append(f"自动设置输出目录: {self.ie_dst_dir}")

        os.makedirs(self.ie_dst_dir, exist_ok=True)

        # 让文件树监视输出目录（若它在工作目录下，会被自动发现；若在外部则追加监视）
        self._file_tree_panel.watch_extra_dir(self.ie_dst_dir)

        step_order = self._ie_get_step_order()
        if not step_order:
            QMessageBox.warning(self, "提示", "请至少勾选一个处理步骤")
            return

        # 不再调用 _ie_sync_config_from_ui()，配置已通过对话框同步到 ie_config
        filename_pattern = self.ie_filename_edit.text().strip() or "{original}_processed"

        self.ie_batch_run_btn.setEnabled(False)
        self.ie_stop_btn.setEnabled(True)
        self.ie_progress_bar.setValue(0)
        self.ie_progress_bar.setVisible(True)
        self.ie_log.append(f"开始批量处理: {self.ie_src_dir}")
        self.ie_log.append(f"   步骤: {' -> '.join(ImageEditor.STEP_LABELS.get(s, s) for s in step_order)}")

        self.ie_worker_thread = _IEBatchWorker(
            self.ie_src_dir, self.ie_dst_dir,
            self.ie_config, step_order, filename_pattern,
            max_workers=self.ie_workers_spin.value())
        self.ie_worker_thread.progress.connect(self._ie_on_batch_progress)
        self.ie_worker_thread.finished.connect(self._ie_on_batch_finished)
        self.ie_worker_thread.error.connect(self._ie_on_batch_error)
        self.ie_worker_thread.start()

    def _ie_stop_batch(self):
        if self.ie_worker_thread and self.ie_worker_thread.isRunning():
            self.ie_worker_thread.stop()
            self.ie_log.append("已发送停止信号")

    def _ie_on_batch_progress(self, done, total, fname):
        pct = int(done / total * 100) if total > 0 else 0
        self.ie_progress_bar.setValue(pct)
        self.ie_log.append(f"  [{done}/{total}] {fname}")

    def _ie_on_batch_finished(self, success, total):
        self.ie_batch_run_btn.setEnabled(True)
        self.ie_stop_btn.setEnabled(False)
        self.ie_progress_bar.setVisible(False)
        self.ie_log.append(
            f"批量处理完成: {success}/{total} 成功  输出: {self.ie_dst_dir}")
        # 批处理完成后主动刷新文件树（确保立即显示新文件）
        self._file_tree_panel.refresh()
        QMessageBox.information(self, "完成",
                                f"批量处理完成!\n成功: {success}/{total}\n输出目录: {self.ie_dst_dir}")

    def _ie_on_batch_error(self, msg):
        self.ie_batch_run_btn.setEnabled(True)
        self.ie_stop_btn.setEnabled(False)
        self.ie_progress_bar.setVisible(False)
        self.ie_log.append(f"批量处理出错: {msg}")
        QMessageBox.critical(self, "错误", msg)

    @staticmethod
    def _ie_ndarray_to_pixmap(img: np.ndarray) -> "Optional[QPixmap]":
        """numpy 数组转 QPixmap（自动归一化非8位图像到0-255显示）。"""
        import cv2 as _cv2
        if img is None:
            return None
        if img.dtype != np.uint8:
            vmin, vmax = float(img.min()), float(img.max())
            if vmax <= vmin:
                img = np.zeros(img.shape[:2], dtype=np.uint8)
            else:
                img = _cv2.normalize(img, None, 0, 255, _cv2.NORM_MINMAX, dtype=_cv2.CV_8U)
        if img.ndim == 2:
            img_rgb = _cv2.cvtColor(img, _cv2.COLOR_GRAY2RGB)
        elif img.ndim == 3 and img.shape[2] == 4:
            img_rgb = _cv2.cvtColor(img, _cv2.COLOR_BGRA2RGB)
        elif img.ndim == 3 and img.shape[2] == 1:
            img_rgb = _cv2.cvtColor(img, _cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = _cv2.cvtColor(img, _cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        qimg = QImage(img_rgb.data.tobytes(), w, h, w * ch,
                      QImage.Format_RGB888)
        return QPixmap.fromImage(qimg)

    # Image editor batch worker
    def _on_nav_changed(self, btn):
        page_map = {
            "image_editor": 0,
            "calibration": 1,
            "reconstruction": 2,
            "raytrace": 3,
            "particle": 4,
            "piv2d": 5,
        }
        for page_id, nav_btn in self._nav_buttons.items():
            if btn is nav_btn:
                self.content_stack.setCurrentIndex(page_map.get(page_id, 0))
                return
        self.content_stack.setCurrentIndex(0)

    def _navigate_to(self, page_idx):
        legacy_to_current = {
            0: 1,
            1: 2,
            2: 3,
            3: 4,
            4: 5,
            5: 0,
        }
        current_idx = legacy_to_current.get(page_idx, page_idx)
        self.content_stack.setCurrentIndex(current_idx)
        nav_keys = [
            "image_editor",
            "calibration",
            "reconstruction",
            "raytrace",
            "particle",
            "piv2d",
        ]
        if current_idx < len(nav_keys):
            self._nav_buttons[nav_keys[current_idx]].setChecked(True)

    def _clear_current_log(self):
        page_idx = self.content_stack.currentIndex()
        if page_idx == 0:
            self.ie_log.clear()
        elif page_idx == 1:
            self.calib_result_text.clear()
        elif page_idx == 2:
            self.recon_log.clear()
        elif page_idx == 3:
            self.rt_log.clear()
        elif page_idx == 4:
            self.piv_log.clear()
        elif page_idx == 5:
            self.piv2d_log.clear()

    def _show_image(self, path):
        pixmap = QPixmap(path)
        page_idx = self.content_stack.currentIndex()
        if page_idx == 0:
            self.ie_preview_label.setPixmap(pixmap.scaled(
                self.ie_preview_label.size(), Qt.KeepAspectRatio,
                Qt.SmoothTransformation))
        elif page_idx == 1:
            self.calib_preview_label.show_static_image(path, title=os.path.basename(path))
        elif page_idx == 2:
            self.proj_preview_label.setPixmap(pixmap.scaled(
                self.proj_preview_label.size(), Qt.KeepAspectRatio,
                Qt.SmoothTransformation))
        elif page_idx == 3:
            self.rt_preview_label.setPixmap(pixmap.scaled(
                self.rt_preview_label.size(), Qt.KeepAspectRatio,
                Qt.SmoothTransformation))
        elif page_idx == 4:
            self.piv_preview.setPixmap(pixmap.scaled(
                self.piv_preview.size(), Qt.KeepAspectRatio,
                Qt.SmoothTransformation))
        elif page_idx == 5:
            self._piv2d_set_preview(path, self.piv2d_result_preview)


def _ie_resolve_batch_filename(original_name, index, step_order, pattern):
    """为批量处理解析文件名模式。"""
    import datetime
    from pathlib import Path
    p = Path(original_name)
    name = p.stem
    ext = p.suffix
    if not pattern:
        pattern = "{original}_processed"
    result = pattern
    result = result.replace("{original}", name)
    result = result.replace("{index}", f"{index:04d}")
    if "{step}" in result and step_order:
        step_str = "_".join(step_order)[:20]
        result = result.replace("{step}", step_str)
    if "{date}" in result:
        date_str = datetime.datetime.now().strftime("%Y%m%d")
        result = result.replace("{date}", date_str)
    if not Path(result).suffix:
        result += ext
    return result


class _IEBatchWorker(QThread):
    progress = pyqtSignal(int, int, str)   # done, total, filename
    finished = pyqtSignal(int, int)         # success, total
    error    = pyqtSignal(str)

    def __init__(self, src_dir, dst_dir, config: "ImageEditConfig",
                 step_order=None, filename_pattern="{original}_processed",
                 max_workers=None):
        super().__init__()
        self.src_dir = src_dir
        self.dst_dir = dst_dir
        self.config  = config
        self.step_order = step_order or []
        self.filename_pattern = filename_pattern
        self.max_workers = max_workers
        self._stop   = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            from utils.image_editor import ImageEditor, SUPPORTED_EXTS, robust_imread
            import cv2 as _cv2

            files = [
                f for f in Path(self.src_dir).iterdir()
                if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS
            ]
            total   = len(files)
            success = 0

            # 预加载操作数图像
            op_img = None
            if (self.config.arithmetic.operand_path
                    and os.path.isfile(self.config.arithmetic.operand_path)):
                op_img = robust_imread(self.config.arithmetic.operand_path,
                                     _cv2.IMREAD_UNCHANGED)

            worker_count = default_worker_count(self.max_workers)
            if total <= 1:
                worker_count = 1

            def process_one(index, file_path):
                editor = ImageEditor(self.config)
                dst_name = _ie_resolve_batch_filename(
                    file_path.name, index, self.step_order, self.filename_pattern)
                dst = str(Path(self.dst_dir) / dst_name)
                img = robust_imread(str(file_path), _cv2.IMREAD_UNCHANGED)
                if img is None:
                    return index, file_path.name, False
                result = editor.process(img, op_img, step_order=self.step_order)
                ok = _cv2.imwrite(dst, result)
                return index, file_path.name, bool(ok)

            if worker_count <= 1:
                for i, f in enumerate(files):
                    if self._stop:
                        break
                    _, fname, ok = process_one(i, f)
                    if ok:
                        success += 1
                    self.progress.emit(i + 1, total, fname)
                self.finished.emit(success, total)
                return

            completed = 0
            next_index = 0
            in_flight = set()
            with limited_opencv_threads(), ThreadPoolExecutor(max_workers=worker_count) as executor:
                while (next_index < total or in_flight) and not self._stop:
                    while next_index < total and len(in_flight) < worker_count and not self._stop:
                        in_flight.add(executor.submit(process_one, next_index, files[next_index]))
                        next_index += 1

                    if not in_flight:
                        break

                    done, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
                    for future in done:
                        _, fname, ok = future.result()
                        completed += 1
                        if ok:
                            success += 1
                        self.progress.emit(completed, total, fname)

            self.finished.emit(success, total)
        except Exception as e:
            self.error.emit(str(e))


def run_gui():
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = BubbleTomographyGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    run_gui()
