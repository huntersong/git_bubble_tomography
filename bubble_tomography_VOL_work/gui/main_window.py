"""
气泡三维层析重建系统 - PyQt5 GUI主窗口
"""

import sys
import os
import json
import re
import cv2
import numpy as np
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
    QMenu
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QModelIndex
from PyQt5.QtGui import QImage, QPixmap, QIcon, QIntValidator, QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from calibration.camera_calibrator import MultiCameraCalibrator, CameraParams
from mart.mart_reconstructor import MARTReconstructor, MARTConfig
from utils.image_processor import BubbleImageProcessor
from visualization.visualizer import ResultVisualizer
from particles.particle_reconstructor import (
    Particle3DReconstructor, TriangulationConfig
)
from particles.velocity_field import (
    VelocityFieldCalculator, CorrelationConfig
)
from particles.piv2d import PIV2DCalculator, PIV2DConfig, SUPPORTED_EXTS as PIV2D_EXTS
from ptv.tracker import PTVTracker, PTVConfig, ForwardBackwardTracker, NearestNeighborTracker, RelaxationTracker, ShakeTheBoxTracker
from ptv.velocity import PTVVelocityCalculator, VelocityProfile
from raytrace.raytrace_reconstructor import RaytraceProcessor
from bubble_segment.pre_processing import pre_processing
from bubble_segment.sobel_filter import sobel_defocus_remove
from bubble_segment.bubble_processor import bubble_processing
from bubble_segment.overlap_handler import overlap_bubbles
from bubble_segment.bubble_filter import bubble_deleting
from bubble_segment.postprocessor import postprocessing
from bubble_segment.statistics import bubble_statistics
from bubble_segment.visualizer import draw_bubbles
from utils.image_editor import (
    ImageEditor, ImageEditConfig,
    CropParams, GrayParams, BrightnessContrastParams,
    MirrorParams, RotateParams, BitDepthParams, GrayMathParams,
    ArithmeticParams, ThresholdParams
)


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
        image = cv2.imread(self.image_path, cv2.IMREAD_UNCHANGED)
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


class _BubbleSegmentWorker(QThread):
    """气泡分割与统计后台处理线程"""
    progress = pyqtSignal(int, int, str)     # current, total, message
    result_ready = pyqtSignal(dict)          # 处理结果
    error = pyqtSignal(str)

    def __init__(self, image_paths, background_path, params):
        super().__init__()
        self.image_paths = image_paths
        self.background_path = background_path
        self.params = params  # dict: imagesize, small_bub_size, maxsize_th, etc.

    def run(self):
        try:
            import cv2
            total = len(self.image_paths)
            all_results = []

            # 读取背景图
            image_back = None
            if self.background_path:
                image_back = cv2.imread(self.background_path, cv2.IMREAD_GRAYSCALE)
                if image_back is None:
                    image_back = None

            tracking_data = None
            imagesize = self.params.get('imagesize', None)

            for i, path in enumerate(self.image_paths):
                self.progress.emit(i + 1, total, f"处理: {os.path.basename(path)}")

                # 读取图像
                image_bub = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if image_bub is None:
                    continue

                # 预处理
                out_image, boundaries, gray_image, lab_image = pre_processing(
                    image_bub, image_back,
                    imagesize=imagesize,
                    small_bub_size=self.params.get('small_bub_size', 20),
                )

                # 失焦去除
                defocus = sobel_defocus_remove(
                    boundaries, gray_image, lab_image,
                    grad_th=self.params.get('grad_th', 0.25),
                    bub_grad_size_th=self.params.get('bub_grad_size_th', 50),
                )
                # 移除失焦气泡
                if defocus:
                    boundaries = [b for idx, b in enumerate(boundaries) if idx not in defocus]

                # 气泡处理（凹点识别/分类/拟合）
                bubble, bub_overlap, ao_data = bubble_processing(
                    boundaries, lab_image,
                    roundness_th=self.params.get('roundness_th', 0.94),
                    bub_size_th=self.params.get('bub_size_th', 30),
                    method_flag=self.params.get('method_flag', 1),
                    single_ao_length=self.params.get('single_ao_length', 10),
                )

                # 重叠气泡处理
                if bub_overlap:
                    bubble_overlap = overlap_bubbles(bub_overlap, ao_data, out_image)
                    # 合并重叠结果回 bubble
                    for entry in bubble_overlap:
                        orig_idx = entry.get('original_index', -1)
                        if 0 <= orig_idx < len(bubble):
                            fits = entry.get('fit', [])
                            fit_bnds = entry.get('fit_boundary', [])
                            bubble[orig_idx]['fit'] = fits
                            bubble[orig_idx]['fit_boundary'] = fit_bnds

                # 异常删除
                bubble = bubble_deleting(
                    bubble,
                    maxsize_th=self.params.get('maxsize_th', 1000),
                    minsize_th=self.params.get('minsize_th', 5),
                    fitting_th=self.params.get('fitting_th', 0.65),
                )

                # 追踪
                from bubble_segment.tracker import bubble_tracking
                if tracking_data is None:
                    tracking_data = bubble_tracking(bubble, mode='first',
                                                   maxlength_th=self.params.get('maxlength_th', 20))
                else:
                    tracking_data = bubble_tracking(bubble, mode='others',
                                                   tracking_data=tracking_data,
                                                   maxlength_th=self.params.get('maxlength_th', 20))

                # 后处理
                bubble_data = postprocessing(bubble)

                # 统计
                sizes = bubble_data[:, 1]
                sizes = sizes[sizes > 0]
                stats = bubble_statistics(sizes) if len(sizes) > 0 else {}

                # 可视化
                result_image = draw_bubbles(out_image, bubble)

                all_results.append({
                    'path': path,
                    'bubble': bubble,
                    'bubble_data': bubble_data,
                    'stats': stats,
                    'result_image': result_image,
                    'num_bubbles': len([b for b in bubble if b.get('fit', [])]),
                })

            self.result_ready.emit({
                'results': all_results,
                'tracking_data': tracking_data,
            })
        except Exception as e:
            import traceback
            self.error.emit(f"{str(e)}\n{traceback.format_exc()}")


class ReconstructionWorker(QThread):
    """重建工作线程"""
    progress = pyqtSignal(str)
    iteration_done = pyqtSignal(int, float)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, reconstructor: MARTReconstructor,
                 projections: Dict[str, np.ndarray],
                 camera_params: Dict[str, dict]):
        super().__init__()
        self.reconstructor = reconstructor
        self.projections = projections
        self.camera_params = camera_params

    def run(self):
        try:
            self.progress.emit("开始 MART 重建...")
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

    def __init__(self, reconstructor: MARTReconstructor,
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
                    f"[{tp_idx+1}/{total}] 处理时间点 t{tp_idx} ..."
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

    def __init__(self, src_dir: str, dst_dir: str, config: PIV2DConfig):
        super().__init__()
        self.src_dir = src_dir
        self.dst_dir = dst_dir
        self.config = config
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            calculator = PIV2DCalculator(self.config)
            success, total, outputs = calculator.process_batch_directory(
                self.src_dir,
                self.dst_dir,
                progress_callback=lambda done, total, name: self.progress.emit(done, total, name),
                stop_checker=lambda: self._stop,
            )
            self.finished.emit(success, total, outputs)
        except Exception as e:
            self.error.emit(str(e))


class PTVBatchWorker(QThread):
    """PTV批量跟踪工作线程。"""
    progress = pyqtSignal(str)
    finished = pyqtSignal(object)  # TrackingResult
    error = pyqtSignal(str)

    def __init__(self, frames_particles: dict, frame_indices: list,
                 ptv_config: PTVConfig,
                 calibrator=None, images: dict = None):
        super().__init__()
        self.frames_particles = frames_particles
        self.frame_indices = frame_indices
        self.ptv_config = ptv_config
        self.calibrator = calibrator
        self.images = images

    def run(self):
        try:
            tracker = PTVTracker(self.ptv_config)
            result = tracker.track(
                self.frames_particles,
                self.frame_indices,
                calibrator=self.calibrator,
                images=self.images
            )
            self.progress.emit(f"跟踪完成: {result.n_tracks} 条轨迹")
            self.finished.emit(result)
        except Exception as e:
            import traceback
            self.error.emit(f"{str(e)}\n{traceback.format_exc()}")


class PIV2DPreviewWidget(QWidget):
    """Grayscale image preview with colorbar, pixel ruler, physical scale bar, and optional vectors."""

    def __init__(self, placeholder: str = "预览区域", parent=None):
        super().__init__(parent)
        self._image = None
        self._result = None
        self._title = placeholder
        self._vector_scale = 0.15
        self._vector_color_mode = "speed"
        self._vector_width = 0.003
        self._vector_headwidth = 4.0
        self._vector_headlength = 5.0
        self._colorbar = None

        # 标尺功能
        self._ruler_enabled = False
        self._ruler_points = []
        self._ruler_dragging = False
        self._ruler_distance_px = 0.0

        # 物理尺度参数 (pixel_scale: mm/px)
        self._pixel_scale = None  # None 表示无物理尺度
        self._scale_bar_visible = True

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.figure = Figure(figsize=(4, 3), tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.axes = self.figure.add_subplot(111)
        self.canvas.setMinimumSize(260, 200)
        self.canvas.mpl_connect("motion_notify_event", self._on_mouse_move)
        self.canvas.mpl_connect("button_press_event", self._on_mouse_press)
        self.canvas.mpl_connect("button_release_event", self._on_mouse_release)
        self.canvas.setContextMenuPolicy(Qt.CustomContextMenu)
        self.canvas.customContextMenuRequested.connect(self._show_context_menu)
        layout.addWidget(self.canvas, stretch=1)

        self.info_label = QLabel("灰度值: --")
        self.info_label.setStyleSheet("color: #444; font-family: Consolas;")
        layout.addWidget(self.info_label)
        self._render()

    # ---- 公共接口 ----

    def set_image_path(self, path: str):
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if image is None:
            return
        self.set_image_array(image, os.path.basename(path))

    def set_image_array(self, image: np.ndarray, title: str = ""):
        self._image = self._to_gray(image)
        self._result = None
        self._title = title or self._title
        self._render()

    def set_vector_result(self, image: np.ndarray, result: dict, title: str = ""):
        self._image = self._to_gray(image)
        self._result = result
        self._title = title or "速度矢量"
        self._render()

    def set_vector_style(self, scale: float, color_mode: str, *,
                         width: float = None, headwidth: float = None,
                         headlength: float = None):
        self._vector_scale = float(scale)
        self._vector_color_mode = color_mode
        if width is not None:
            self._vector_width = width
        if headwidth is not None:
            self._vector_headwidth = headwidth
        if headlength is not None:
            self._vector_headlength = headlength
        self._render()

    def set_pixel_scale(self, mm_per_px: Optional[float]):
        """设置物理尺度 (mm/px)，设为 None 隐藏物理标尺条。"""
        self._pixel_scale = mm_per_px
        self._render()

    def set_scale_bar_visible(self, visible: bool):
        self._scale_bar_visible = visible
        self._render()

    def clear(self, text: Optional[str] = None):
        self._image = None
        self._result = None
        if text:
            self._title = text
        self.info_label.setText("灰度值: --")
        self._render()

    # ---- 标尺 ----

    def enable_ruler(self, enabled: bool = True):
        self._ruler_enabled = enabled
        self._ruler_dragging = False
        if enabled:
            self.info_label.setText("标尺模式: 拖动鼠标测量像素距离")
        self._render()

    def clear_ruler(self):
        self._ruler_points = []
        self._ruler_dragging = False
        self._ruler_distance_px = 0.0
        self._render()

    def _show_context_menu(self, pos):
        menu = QMenu(self)
        toggle_ruler = menu.addAction("打开标尺" if not self._ruler_enabled else "关闭标尺")
        clear_ruler = menu.addAction("清除标尺")
        clear_ruler.setEnabled(bool(self._ruler_points))
        menu.addSeparator()
        toggle_scalebar = menu.addAction("隐藏尺度条" if self._scale_bar_visible else "显示尺度条")
        action = menu.exec_(self.canvas.mapToGlobal(pos))
        if action == toggle_ruler:
            self.enable_ruler(not self._ruler_enabled)
        elif action == clear_ruler:
            self.clear_ruler()
        elif action == toggle_scalebar:
            self.set_scale_bar_visible(not self._scale_bar_visible)

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
        self._render()

    def _on_mouse_release(self, event):
        if not self._ruler_enabled or not self._ruler_dragging:
            return
        self._ruler_dragging = False
        if event.inaxes == self.axes and event.xdata is not None and event.ydata is not None:
            self._ruler_points[-1] = (float(event.xdata), float(event.ydata))
        self._update_ruler_distance()
        self._render()

    def _update_ruler_distance(self):
        if len(self._ruler_points) != 2:
            self._ruler_distance_px = 0.0
        else:
            (x1, y1), (x2, y2) = self._ruler_points
            self._ruler_distance_px = float(np.hypot(x2 - x1, y2 - y1))

    # ---- 渲染 ----

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

        if self._result is not None:
            self._draw_vectors()

        # 物理尺度标尺条
        if self._scale_bar_visible and self._pixel_scale is not None and self._pixel_scale > 0:
            self._draw_scale_bar()

        # 像素标尺叠加
        self._draw_ruler_overlay()

        self.canvas.draw_idle()

    def _draw_vectors(self):
        result = self._result
        valid = result.get("valid")
        if valid is None or not np.any(valid):
            return

        x = result["x"][valid]
        y = result["y"][valid]
        u = result["u"][valid] * self._vector_scale
        v = result["v"][valid] * self._vector_scale

        quiver_kwargs = dict(
            angles="xy", scale_units="xy", scale=1,
            width=self._vector_width,
            headwidth=self._vector_headwidth,
            headlength=self._vector_headlength,
        )

        if self._vector_color_mode == "speed":
            colors = result["speed"][valid]
            quiver = self.axes.quiver(x, y, u, v, colors, cmap="jet", **quiver_kwargs)
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
                **quiver_kwargs,
            )

    def _draw_scale_bar(self):
        """在图像右下角绘制物理尺度标尺条（比例尺）。"""
        img_h, img_w = self._image.shape[:2]

        # 选择一个合适的整数长度 (mm)：目标约占图像宽度的 15%
        target_px = img_w * 0.15
        target_mm = target_px * self._pixel_scale

        # 找到最近的"整数"物理长度
        candidates = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500]
        best = min(candidates, key=lambda c: abs(c - target_mm))
        bar_px = best / self._pixel_scale

        # 标尺条位置：右下角偏移
        margin_x = img_w * 0.05
        margin_y = img_h * 0.08
        x0 = img_w - bar_px - margin_x
        y0 = img_h - margin_y

        # 画标尺条（粗线 + 端点短竖线）
        lw = max(1.5, img_w / 600)
        self.axes.plot([x0, x0 + bar_px], [y0, y0], color="white", linewidth=lw, solid_capstyle="butt")
        tick_h = img_h * 0.015
        self.axes.plot([x0, x0], [y0 - tick_h, y0 + tick_h], color="white", linewidth=lw)
        self.axes.plot([x0 + bar_px, x0 + bar_px], [y0 - tick_h, y0 + tick_h], color="white", linewidth=lw)

        # 标注文字
        if best >= 1:
            label = f"{best:.0f} mm"
        else:
            label = f"{best:.1f} mm"
        fs = max(8, int(img_w / 100))
        self.axes.text(
            x0 + bar_px / 2, y0 + tick_h * 2.5, label,
            color="white", ha="center", va="bottom", fontsize=fs, fontweight="bold",
            bbox=dict(facecolor="black", alpha=0.5, pad=1.5, boxstyle="round,pad=0.2"),
        )

    def _draw_ruler_overlay(self):
        """绘制像素标尺测量线和标注。"""
        if len(self._ruler_points) != 2:
            return
        (x1, y1), (x2, y2) = self._ruler_points
        self.axes.plot([x1, x2], [y1, y2], color="#ff3b30", linewidth=2)
        self.axes.scatter([x1, x2], [y1, y2], color="#ff3b30", s=28, zorder=5)

        if self._ruler_distance_px > 0:
            text = f"{self._ruler_distance_px:.1f} px"
            if self._pixel_scale is not None and self._pixel_scale > 0:
                text += f" ({self._ruler_distance_px * self._pixel_scale:.2f} mm)"
            self.axes.text(
                (x1 + x2) / 2, (y1 + y2) / 2, text,
                color="white", ha="center", va="center", fontsize=9,
                bbox=dict(facecolor="#222", alpha=0.8, pad=3),
            )

    def _on_mouse_move(self, event):
        if self._image is None or event.inaxes is not self.axes:
            self.info_label.setText("灰度值: --")
            return
        if event.xdata is None or event.ydata is None:
            self.info_label.setText("灰度值: --")
            return

        # 标尺拖拽实时更新
        if self._ruler_enabled and self._ruler_dragging and len(self._ruler_points) == 2:
            self._ruler_points[-1] = (float(event.xdata), float(event.ydata))
            self._update_ruler_distance()
            self._render()

        x = int(round(event.xdata))
        y = int(round(event.ydata))
        h, w = self._image.shape[:2]
        if 0 <= x < w and 0 <= y < h:
            value = float(self._image[y, x])
            text = f"x={x}  y={y}  灰度值={value:.1f}"
            if self._ruler_distance_px > 0:
                text += f"  |  标尺: {self._ruler_distance_px:.1f} px"
                if self._pixel_scale is not None and self._pixel_scale > 0:
                    text += f" ({self._ruler_distance_px * self._pixel_scale:.2f} mm)"
            self.info_label.setText(text)
        else:
            self.info_label.setText("灰度值: --")

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
        self.reconstructor: Optional[MARTReconstructor] = None
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

        # 撤销栈：每项为 (描述, 回调函数)
        self._undo_stack: list = []
        self._undo_max_depth: int = 30

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
            ("bubble_segment", "气泡分割与统计"),
            ("particle",       "Particle / PIV"),
            ("ptv",            "PTV粒子跟踪"),
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

        # === 右侧内容区 (QStackedWidget) ===
        self.content_stack = QStackedWidget()
        self.content_stack.setStyleSheet("QStackedWidget { border: none; }")

        # Page 0: 相机标定
        self._create_calibration_page()

        # Page 1: 气泡重建
        self._create_reconstruction_page()

        # Page 2: 单相机3D重建（射线追踪）
        self._create_raytrace_page()

        # Page 3: 气泡分割与统计
        self._create_bubble_segment_page()

        # Page 4: 粒子追踪与PIV
        self._create_particle_page()

        # Page 5: PTV粒子跟踪
        self._create_ptv_page()

        # Page 6: 二维PIV
        self._create_piv2d_page()

        # Page 7: 通用图像处理
        self._create_image_editor_page()
        image_editor_page = self.content_stack.widget(7)
        if image_editor_page is not None:
            self.content_stack.removeWidget(image_editor_page)
            self.content_stack.insertWidget(0, image_editor_page)
            self.content_stack.setCurrentIndex(0)

        # 默认选中图像处理
        self._nav_buttons["image_editor"].setChecked(True)

        main_layout.addWidget(self._nav_panel)
        main_layout.addWidget(self.content_stack, stretch=1)

        # 连接导航切换
        self._nav_btn_group.buttonClicked.connect(self._on_nav_changed)

        # 状态栏
        self.statusBar().showMessage("就绪")

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.statusBar().addPermanentWidget(self.progress_bar)

    def _make_separator(self):
        """创建分隔线。"""
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("color: #4a6785;")
        return line

    def _on_nav_changed(self, btn):
        """左侧导航切换"""
        page_map = {
            "相机标定": 1,
            "气泡重建": 2,
            "单相机3D重建": 3,
            "气泡分割与统计": 4,
            "Particle / PIV": 5,
            "PTV粒子跟踪": 6,
            "二维PIV": 7,
            "图像处理": 0,
        }
        idx = page_map.get(btn.text(), 0)
        self.content_stack.setCurrentIndex(idx)

    def _create_menubar(self):
        """创建菜单栏。"""
        menubar = self.menuBar()

        # ===== 文件菜单 =====
        file_menu = menubar.addMenu("文件(&F)")

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

        undo_action = QAction("撤销(&U)", self)
        undo_action.setShortcut("Ctrl+Z")
        undo_action.setStatusTip("撤销上一步操作")
        undo_action.triggered.connect(self._undo_last_action)
        edit_menu.addAction(undo_action)
        self._undo_action_ref = undo_action

        edit_menu.addSeparator()

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

        nav_to_bseg = QAction("气泡分割与统计", self)
        nav_to_bseg.triggered.connect(lambda: self._navigate_to(3))
        window_menu.addAction(nav_to_bseg)

        nav_to_particle = QAction("Particle / PIV", self)
        nav_to_particle.triggered.connect(lambda: self._navigate_to(4))
        window_menu.addAction(nav_to_particle)

        nav_to_piv2d = QAction("二维PIV", self)
        nav_to_piv2d.triggered.connect(lambda: self._navigate_to(5))
        window_menu.addAction(nav_to_piv2d)

        nav_to_ie = QAction("图像处理", self)
        nav_to_ie.triggered.connect(lambda: self._navigate_to(6))
        window_menu.addAction(nav_to_ie)

        # ===== 帮助菜单 =====
        help_menu = menubar.addMenu("帮助(&H)")

        open_manual_action = QAction("打开PDF说明...", self)
        open_manual_action.triggered.connect(self._open_pdf_manual)
        help_menu.addAction(open_manual_action)

        help_menu.addSeparator()

        about_action = QAction("关于", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _apply_global_style(self):
        """应用全局样式。"""
        self.setStyleSheet(self._scale_stylesheet("""
            /* === 全局字体 === */
            * {
                font-family: "Microsoft YaHei", "SimHei", "Segoe UI", sans-serif;
                font-size: 13px;
            }
            QMainWindow {
                font-size: 13px;
            }
            QMenuBar {
                font-size: 13px;
                padding: 2px;
            }
            QMenuBar::item {
                padding: 4px 12px;
            }
            QMenu {
                font-size: 13px;
            }
            QMenu::item {
                padding: 5px 30px 5px 20px;
            }
            QStatusBar {
                font-size: 12px;
            }
            QGroupBox {
                font-size: 13px;
                font-weight: bold;
                border: 1px solid #ccc;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 15px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QLabel {
                font-size: 13px;
            }
            QPushButton {
                font-size: 13px;
                padding: 5px 14px;
            }
            QLineEdit {
                font-size: 13px;
                padding: 4px 8px;
            }
            QComboBox {
                font-size: 13px;
                padding: 4px 8px;
            }
            QSpinBox, QDoubleSpinBox {
                font-size: 13px;
                padding: 3px 6px;
            }
            QTextEdit {
                font-size: 12px;
            }
            QListWidget {
                font-size: 13px;
            }
            QSlider::groove:horizontal {
                height: 6px;
            }
            QProgressBar {
                font-size: 12px;
            }
            QTabWidget::tab {
                font-size: 13px;
                padding: 6px 16px;
            }
            QToolBar {
                font-size: 13px;
            }
            QScrollArea {
                font-size: 13px;
            }
        """))

    def _navigate_to(self, page_idx):
        """窗口菜单导航辅助"""
        self.content_stack.setCurrentIndex(page_idx)
        nav_keys = ["image_editor", "calibration", "reconstruction", "raytrace", "bubble_segment", "particle", "ptv", "piv2d"]
        if page_idx < len(nav_keys):
            self._nav_buttons[nav_keys[page_idx]].setChecked(True)

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
                f"已打开说明文件: {os.path.basename(pdf_path)}"
            )
        except Exception as e:
            QMessageBox.critical(self, "打开失败", f"无法打开PDF说明文件\n{e}")

    # ── 撤销框架 ─────────────────────────────────────────────────────────

    def _push_undo(self, description: str, callback):
        """压入一条撤销记录。callback() 被调用时执行撤销动作。"""
        self._undo_stack.append((description, callback))
        if len(self._undo_stack) > self._undo_max_depth:
            self._undo_stack.pop(0)
        self.statusBar().showMessage(f"可撤销: {description}", 3000)

    def _undo_last_action(self):
        """执行上一步撤销。"""
        if not self._undo_stack:
            self.statusBar().showMessage("没有可撤销的操作", 2000)
            return
        description, callback = self._undo_stack.pop()
        try:
            callback()
            self.statusBar().showMessage(f"已撤销: {description}", 3000)
        except Exception as exc:
            self.statusBar().showMessage(f"撤销失败: {exc}", 4000)

    # ── 日志操作 ─────────────────────────────────────────────────────────

    def _clear_current_log(self):
        """清空当前页面的日志区（支持撤销）。"""
        page_idx = self.content_stack.currentIndex()
        log_widget_map = {
            0: getattr(self, "ie_log", None),
            1: getattr(self, "calib_result_text", None),
            2: getattr(self, "recon_log", None),
            3: getattr(self, "rt_log", None),
            4: getattr(self, "bseg_log", None),
            5: getattr(self, "piv_log", None),
            6: getattr(self, "ptv_log", None),
            7: getattr(self, "piv2d_log", None),
        }
        widget = log_widget_map.get(page_idx)
        if widget is None:
            return
        # 保存旧内容用于撤销
        old_text = widget.toPlainText()
        page_names = {0: "图像处理", 1: "相机标定", 2: "气泡重建",
                      3: "单相机3D重建", 4: "气泡分割", 5: "Particle/PIV",
                      6: "PTV粒子跟踪", 7: "二维PIV"}
        name = page_names.get(page_idx, "当前页面")

        def _restore(w=widget, text=old_text):
            w.setPlainText(text)

        self._push_undo(f"清空 {name} 日志", _restore)
        widget.clear()

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

        # === 多相机标定区域（默认）===
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
        recon_group = QGroupBox("MART重建参数")
        grid = QGridLayout()

        grid.addWidget(QLabel("网格X:"), 0, 0)
        self.grid_x = QSpinBox()
        self.grid_x.setRange(16, 256)
        self.grid_x.setValue(64)
        grid.addWidget(self.grid_x, 0, 1)

        grid.addWidget(QLabel("网格Y:"), 1, 0)
        self.grid_y = QSpinBox()
        self.grid_y.setRange(16, 256)
        self.grid_y.setValue(64)
        grid.addWidget(self.grid_y, 1, 1)

        grid.addWidget(QLabel("网格Z:"), 2, 0)
        self.grid_z = QSpinBox()
        self.grid_z.setRange(16, 256)
        self.grid_z.setValue(64)
        grid.addWidget(self.grid_z, 2, 1)

        grid.addWidget(QLabel("域尺寸X (mm):"), 3, 0)
        self.domain_x = QDoubleSpinBox()
        self.domain_x.setRange(1, 200)
        self.domain_x.setValue(20.0)
        grid.addWidget(self.domain_x, 3, 1)

        grid.addWidget(QLabel("域尺寸Y (mm):"), 4, 0)
        self.domain_y = QDoubleSpinBox()
        self.domain_y.setRange(1, 200)
        self.domain_y.setValue(20.0)
        grid.addWidget(self.domain_y, 4, 1)

        grid.addWidget(QLabel("域尺寸Z (mm):"), 5, 0)
        self.domain_z = QDoubleSpinBox()
        self.domain_z.setRange(1, 200)
        self.domain_z.setValue(20.0)
        grid.addWidget(self.domain_z, 5, 1)

        grid.addWidget(QLabel("松弛因子:"), 6, 0)
        self.relax_spin = QDoubleSpinBox()
        self.relax_spin.setRange(0.01, 1.0)
        self.relax_spin.setValue(0.5)
        self.relax_spin.setSingleStep(0.05)
        grid.addWidget(self.relax_spin, 6, 1)

        grid.addWidget(QLabel("最大迭代次数:"), 7, 0)
        self.iter_spin = QSpinBox()
        self.iter_spin.setRange(1, 200)
        self.iter_spin.setValue(50)
        grid.addWidget(self.iter_spin, 7, 1)

        grid.addWidget(QLabel("体素阈值:"), 8, 0)
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.01, 0.5)
        self.threshold_spin.setValue(0.1)
        self.threshold_spin.setSingleStep(0.01)
        grid.addWidget(self.threshold_spin, 8, 1)

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
            "请先设置相机数目，然后点击上方按钮逐个加载各相机的粒子图像序列。\n"
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
        self.piv_frame_select_info = QLabel("加载序列后可选帧")
        self.piv_frame_select_info.setStyleSheet("color: #666;")
        self.piv_frame_select_info.setWordWrap(True)
        frame_select_layout.addWidget(self.piv_frame_select_info, 2, 0, 1, 2)
        img_layout.addWidget(frame_select_group)

        # 单帧手动加载
        single_frame_layout = QHBoxLayout()
        btn_load_frame1 = QPushButton("单帧: 加载第1帧...")
        btn_load_frame1.clicked.connect(self._load_particle_frame1)
        single_frame_layout.addWidget(btn_load_frame1)

        btn_load_frame2 = QPushButton("单帧: 加载第2帧...")
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

        det_grid.addWidget(QLabel("最小圆度:"), 2, 0)
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

        # PIV参数
        vel_group = QGroupBox("PIV参数")
        vel_grid = QGridLayout()
        vel_grid.addWidget(QLabel("dt (s):"), 0, 0)
        self.piv_dt = QDoubleSpinBox()
        self.piv_dt.setRange(0.0001, 10)
        self.piv_dt.setValue(0.001)
        self.piv_dt.setDecimals(4)
        self.piv_dt.setSingleStep(0.0001)
        vel_grid.addWidget(self.piv_dt, 0, 1)

        vel_grid.addWidget(QLabel("查询窗口大小 (mm):"), 1, 0)
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

        vel_grid.addWidget(QLabel("SNR阈值:"), 3, 0)
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

        btn_piv_all = QPushButton("批量PIV分析")
        btn_piv_all.setStyleSheet(
            "QPushButton { background-color: #E65100; color: white; "
            "font-size: 13px; padding: 8px; border-radius: 5px; }"
            "QPushButton:hover { background-color: #BF360C; }"
        )
        btn_piv_all.clicked.connect(self._run_piv_batch)
        piv_btn_layout.addWidget(btn_piv_all)

        left_layout.addLayout(piv_btn_layout)

        btn_velocity = QPushButton("计算三维速度场")
        btn_velocity.setStyleSheet(
            "QPushButton { background-color: #9C27B0; color: white; "
            "font-size: 13px; padding: 8px; border-radius: 5px; }"
            "QPushButton:hover { background-color: #7B1FA2; }"
        )
        btn_velocity.clicked.connect(self._run_velocity_computation)
        left_layout.addWidget(btn_velocity)

        left_layout.addStretch()

        # 右侧: 日志 + 可视化 + 时间点选择
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
                f"已加载 {len(self.particle_timepoint_images)} 个时间点"
                f"{len(active)} ???"
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
            f"当前双帧: ??= {_name(self.piv_frame1_combo)}?"
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

    def _run_single_calibration_legacy(self):
        """执行单相机标定。"""
        if not hasattr(self, '_single_calib_files') or not self._single_calib_files:
            QMessageBox.warning(self, "警告", "请先加载标定图像")
            return
        if len(self._single_calib_files) < 3:
            QMessageBox.warning(self, "警告", "至少需要 3 张标定图像")
            return

        import cv2

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

        # 生成标定板物理坐标
        objp = np.zeros(
            (pattern_size[0] * pattern_size[1], 3), np.float32
        )
        objp[:, :2] = np.mgrid[
            0:pattern_size[0], 0:pattern_size[1]
        ].T.reshape(-1, 2) * square_size

        obj_points = []
        img_points = []

        for fpath in self._single_calib_files:
            img = cv2.imread(fpath)
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            if pattern_type == "checkerboard":
                ret, corners = cv2.findChessboardCorners(
                    gray, pattern_size, None
                )
                if ret:
                    corners = cv2.cornerSubPix(
                        gray, corners, (11, 11), (-1, -1),
                        criteria=(cv2.TERM_CRITERIA_EPS +
                                  cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                    )
                    obj_points.append(objp)
                    img_points.append(corners)
            else:
                flags = (
                    cv2.CALIB_CB_SYMMETRIC_GRID if pattern_type == "circles"
                    else cv2.CALIB_CB_ASYMMETRIC_GRID
                )
                ret, centers = cv2.findCirclesGrid(
                    gray, pattern_size, flags=flags
                )
                if ret:
                    obj_points.append(objp)
                    img_points.append(centers)

        if len(obj_points) < 3:
            QMessageBox.warning(self, "警告",
                "能成功检测点的图像不足 3 张，请检查图像质量")
            return

        h, w = cv2.imread(self._single_calib_files[0]).shape[:2]
        rms, camera_matrix, dist_coeffs, rvecs, tvecs = \
            cv2.calibrateCamera(
                obj_points, img_points, (w, h), None, None
            )

        self.single_camera_params = {
            'camera_matrix': camera_matrix.tolist(),
            'dist_coeffs': dist_coeffs.flatten().tolist(),
            'rms': float(rms),
            'image_size': [w, h],
            'rvecs': [r.flatten().tolist() for r in rvecs],
            'tvecs': [t.flatten().tolist() for t in tvecs],
            'n_images': len(obj_points)
        }

        report = (
            f"=== 单相机标定完成 ===\n\n"
            f"图像数量: {len(obj_points)}\n"
            f"图像尺寸: {w} x {h}\n"
            f"重投影误差(RMS): {rms:.4f} px\n\n"
            f"--- 内参矩阵 ---\n"
            f"fx={camera_matrix[0,0]:.2f}  fy={camera_matrix[1,1]:.2f}\n"
            f"cx={camera_matrix[0,2]:.2f}  cy={camera_matrix[1,2]:.2f}\n\n"
            f"--- 畸变系数 ---\n"
            f"k1={dist_coeffs[0,0]:.6f}\n"
            f"k2={dist_coeffs[0,1]:.6f}\n"
            f"p1={dist_coeffs[0,2]:.6f}\n"
            f"p2={dist_coeffs[0,3]:.6f}\n"
            f"k3={dist_coeffs[0,4]:.6f}\n\n"
            f"标定结果已保存，可用于单相机射线追踪三维重建"
        )
        self.calib_result_text.setPlainText(report)
        self.statusBar().showMessage(
            f"单相机标定完成 RMS={rms:.4f}px"
        )
        QMessageBox.information(self, "完成",
            f"单相机标定完成\nRMS误差: {rms:.4f} px\n"
            "可用于单相机 3D 重建模块。")

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
            f"=== 单相机标定完成 ===\n\n"
            f"标定板类型: {pattern_type}\n"
            f"标定板尺寸: {pattern_size[0]} x {pattern_size[1]}\n"
            f"有效图像数量: {n_images}\n"
            f"图像尺寸: {image_w} x {image_h}\n"
            f"重投影误差(RMS): {params.rms_error:.4f} px\n\n"
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
            f"单相机标定完成 RMS={params.rms_error:.4f}px"
        )
        QMessageBox.information(
            self,
            "完成",
            f"单相机标定完成\nRMS误差: {params.rms_error:.4f} px\n"
            "可用于单相机 3D 重建模块。"
        )

    def _run_single_calibration(self):
        """Finish single-camera scale calibration from one image and a ruler distance."""
        if not hasattr(self, '_single_calib_files') or not self._single_calib_files:
            QMessageBox.warning(self, "警告", "请先加载 1 张单相机标定图像")
            return

        image_path = self._single_calib_files[0]
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
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
            f"单相机标定完成: {mm_per_px:.6f} mm/px"
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
                f"左相机: 已加载 {len(files)} 张图像"
            )
            self.stereo_left_label.setStyleSheet("color: green;")
            self.statusBar().showMessage(
                f"左相机: 已加载 {len(files)} 张标定图像"
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
                f"右相机: 已加载 {len(files)} 张图像"
            )
            self.stereo_right_label.setStyleSheet("color: green;")
            self.statusBar().showMessage(
                f"右相机: 已加载 {len(files)} 张标定图像"
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

        # 先分别确定左右相机
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
            f"--- 左相机 ---\n"
            f"RMS: {rms_l:.4f} px\n"
            f"fx={K_l[0,0]:.2f}  fy={K_l[1,1]:.2f}\n\n"
            f"--- 右相机 ---\n"
            f"RMS: {rms_r:.4f} px\n"
            f"fx={K_r[0,0]:.2f}  fy={K_r[1,1]:.2f}\n\n"
            f"--- 立体参数 ---\n"
            f"立体RMS: {rms_stereo:.4f} px\n"
            f"平移 T: [{T[0]:.2f}, {T[1]:.2f}, {T[2]:.2f}] mm\n"
            f"旋转角: {np.degrees(np.arccos(np.clip((np.trace(R)-1)/2, -1, 1))):.2f} deg"
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
            f"标定板类型: {pattern_type}\n"
            f"标定板尺寸: {pattern_size[0]} x {pattern_size[1]}\n"
            f"有效图像数: {len(objpoints)}\n"
            f"图像尺寸: {image_size[0]} x {image_size[1]}\n\n"
            f"--- 左相机 ---\n"
            f"RMS: {rms_l:.4f} px\n"
            f"fx={K_l[0,0]:.2f}  fy={K_l[1,1]:.2f}\n\n"
            f"--- 右相机 ---\n"
            f"RMS: {rms_r:.4f} px\n"
            f"fx={K_r[0,0]:.2f}  fy={K_r[1,1]:.2f}\n\n"
            f"--- 立体参数 ---\n"
            f"立体RMS: {rms_stereo:.4f} px\n"
            f"平移 T: [{T[0,0]:.2f}, {T[1,0]:.2f}, {T[2,0]:.2f}] mm\n"
            f"旋转角: {np.degrees(np.arccos(np.clip((np.trace(R)-1)/2, -1, 1))):.2f} deg"
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
            QMessageBox.warning(self, "警告", f"相机 {cam_id} 已存在")
            return
        self.camera_calib_images[cam_id] = []
        self.camera_list.addItem(f"{cam_id} (0 张标定图)")
        self.cam_id_input.clear()
        self.statusBar().showMessage(f"已添加相机 {cam_id}")

    def _remove_camera(self):
        current = self.camera_list.currentRow()
        if current < 0:
            return
        cam_ids = list(self.camera_calib_images.keys())
        cam_id = cam_ids[current]
        self.camera_calib_images.pop(cam_id, None)
        self.camera_list.takeItem(current)
        self.statusBar().showMessage(f"已移除相机 {cam_id}")

    def _load_calibration_images(self):
        current = self.camera_list.currentRow()
        if current < 0:
            QMessageBox.warning(self, "警告", "请先选择相机")
            return

        cam_ids = list(self.camera_calib_images.keys())
        cam_id = cam_ids[current]

        files, _ = QFileDialog.getOpenFileNames(
            self, f"选择相机 {cam_id} 的标定图像", "",
            "图像文件 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;所有文件 (*)"
        )

        if files:
            self.camera_calib_images[cam_id] = files
            self.camera_list.item(current).setText(
                f"{cam_id} ({len(files)} 张标定图像)"
            )
            self.calib_image_label.setText(
                f"相机 {cam_id}: 已加载 {len(files)} 张标定图像"
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

            # 缩放图像到 cell 区域
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

        # 保存标定结果预览
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

        # 将子文件夹作为时间点
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
                # 在子文件夹中查找包含相机ID的图像文件
                # 策略1: 文件名包含cam_id
                # Match image files by camera ID
                files_in_dir = [
                    f for f in os.listdir(subdir_path)
                    if os.path.splitext(f)[1].lower() in img_exts
                ]
                files_in_dir.sort()

                # 查找匹配当前相机的图像
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
            # 设置时间点选择器
            self._setup_bubble_timepoint_selector()
            # 同步设置当前时间点的单帧数据（兼容旧方法）
            self._set_current_bubble_timepoint(0)
            self.bubble_status_label.setText(
                f"已加载 {loaded_count} 个时间点，每点 {len(cam_ids)} 相机"
            )
            self.bubble_status_label.setStyleSheet("color: green; font-weight: bold;")
            self.recon_log.append(
                f"=== 批量加载完成 ===\n"
                f"根目录: {root_dir}\n"
                f"时间点数: {loaded_count}\n"
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

        # 更新预览
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

        # 拼接各相机图像为行
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
            self.recon_log.append(f"已切换到时间点 {name} 的重建结果")
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
                self, f"选择相机 {cam_id} 的气泡图像", "",
                "图像文件 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
            )
            if file_path:
                img = cv2.imread(file_path, cv2.IMREAD_COLOR)
                if img is not None:
                    self.camera_bubble_images[cam_id] = img

        if self.camera_bubble_images:
            self.bubble_status_label.setText(
                f"已加载 {len(self.camera_bubble_images)} 个相机的气泡图像（单帧）"
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
                self, f"选择相机 {cam_id} 的背景参考图（可跳过）", "",
                "图像文件 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
            )
            if file_path:
                import cv2
                img = cv2.imread(file_path, cv2.IMREAD_COLOR)
                if img is not None:
                    self.camera_reference_images[cam_id] = img

        if self.camera_reference_images:
            self.statusBar().showMessage(
                f"已加载 {len(self.camera_reference_images)} 张背景参考图"
            )

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

        # 图像预览
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

        # 配置MART重建器
        config = MARTConfig(
            grid_size=(self.grid_x.value(), self.grid_y.value(),
                       self.grid_z.value()),
            domain_size=(self.domain_x.value(), self.domain_y.value(),
                         self.domain_z.value()),
            relaxation_factor=self.relax_spin.value(),
            max_iterations=self.iter_spin.value(),
            voxel_threshold=self.threshold_spin.value()
        )

        self.reconstructor = MARTReconstructor(config)

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

        config = MARTConfig(
            grid_size=(self.grid_x.value(), self.grid_y.value(),
                       self.grid_z.value()),
            domain_size=(self.domain_x.value(), self.domain_y.value(),
                         self.domain_z.value()),
            relaxation_factor=self.relax_spin.value(),
            max_iterations=self.iter_spin.value(),
            voxel_threshold=self.threshold_spin.value()
        )

        self.reconstructor = MARTReconstructor(config)

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

    def _show_image(self, path):
        """在当前活动页面的预览区显示图像。"""
        pixmap = QPixmap(path)
        page_idx = self.content_stack.currentIndex()
        if page_idx == 0:
            self.calib_preview_label.show_static_image(path, title=os.path.basename(path))
        elif page_idx == 1:
            self.proj_preview_label.setPixmap(pixmap.scaled(
                self.proj_preview_label.size(), Qt.KeepAspectRatio,
                Qt.SmoothTransformation))
        elif page_idx == 2:
            self.rt_preview_label.setPixmap(pixmap.scaled(
                self.rt_preview_label.size(), Qt.KeepAspectRatio,
                Qt.SmoothTransformation))
        elif page_idx == 3:
            self.piv_preview.setPixmap(pixmap.scaled(
                self.piv_preview.size(), Qt.KeepAspectRatio,
                Qt.SmoothTransformation))
        elif page_idx == 4:
            self.piv2d_result_preview.setPixmap(pixmap.scaled(
                self.piv2d_result_preview.size(), Qt.KeepAspectRatio,
                Qt.SmoothTransformation))
        elif page_idx == 5:
            self.ie_preview_label.setPixmap(pixmap.scaled(
                self.ie_preview_label.size(), Qt.KeepAspectRatio,
                Qt.SmoothTransformation))

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

    # ---- 气泡分割与统计 (Bubble Image Segmentation & Statistics) ----

    def _create_bubble_segment_page(self):
        """创建气泡图像分割与统计页面：MATLAB VOL 1.0 算法集成。"""
        page = QWidget()
        layout = QHBoxLayout(page)

        # ---- 左侧控制面板 ----
        left_panel = QWidget()
        left_panel.setMaximumWidth(self._sp(340))
        left_panel.setMinimumWidth(self._sp(300))
        left_layout = QVBoxLayout(left_panel)

        # 图像加载
        img_group = QGroupBox("📷 图像加载")
        img_layout = QVBoxLayout()

        btn_load_bub = QPushButton("选择气泡图像（可多选）")
        btn_load_bub.setStyleSheet("QPushButton { font-weight: bold; padding: 6px; }")
        btn_load_bub.clicked.connect(self._load_bub_segment_images)
        img_layout.addWidget(btn_load_bub)

        self.bseg_img_label = QLabel("未加载图像")
        self.bseg_img_label.setWordWrap(True)
        self.bseg_img_label.setStyleSheet("color: gray;")
        img_layout.addWidget(self.bseg_img_label)

        btn_load_back = QPushButton("选择背景图像（可选）")
        btn_load_back.clicked.connect(self._load_bub_background)
        img_layout.addWidget(btn_load_back)

        self.bseg_back_label = QLabel("未加载背景")
        self.bseg_back_label.setStyleSheet("color: gray; font-size: 11px;")
        img_layout.addWidget(self.bseg_back_label)

        img_group.setLayout(img_layout)
        left_layout.addWidget(img_group)

        # 参数设置
        param_group = QGroupBox("⚙️ 参数设置")
        param_form = QGridLayout(param_group)

        param_form.addWidget(QLabel("图像裁剪:"), 0, 0)
        self.bseg_imagesize_check = QCheckBox("启用")
        self.bseg_imagesize_check.stateChanged.connect(self._on_bseg_imagesize_toggle)
        param_form.addWidget(self.bseg_imagesize_check, 0, 1)
        self.bseg_row_start = QSpinBox()
        self.bseg_row_start.setRange(0, 9999)
        self.bseg_row_start.setValue(1)
        self.bseg_row_start.setEnabled(False)
        self.bseg_row_end = QSpinBox()
        self.bseg_row_end.setRange(0, 9999)
        self.bseg_row_end.setValue(1024)
        self.bseg_row_end.setEnabled(False)
        self.bseg_col_start = QSpinBox()
        self.bseg_col_start.setRange(0, 9999)
        self.bseg_col_start.setValue(1)
        self.bseg_col_start.setEnabled(False)
        self.bseg_col_end = QSpinBox()
        self.bseg_col_end.setRange(0, 9999)
        self.bseg_col_end.setValue(1280)
        self.bseg_col_end.setEnabled(False)
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("R:"))
        size_layout.addWidget(self.bseg_row_start)
        size_layout.addWidget(QLabel("~"))
        size_layout.addWidget(self.bseg_row_end)
        size_layout.addWidget(QLabel("C:"))
        size_layout.addWidget(self.bseg_col_start)
        size_layout.addWidget(QLabel("~"))
        size_layout.addWidget(self.bseg_col_end)
        param_form.addLayout(size_layout, 1, 0, 1, 2)

        param_form.addWidget(QLabel("最小气泡尺寸:"), 2, 0)
        self.bseg_small_size = QSpinBox()
        self.bseg_small_size.setRange(1, 200)
        self.bseg_small_size.setValue(20)
        self.bseg_small_size.setToolTip("预处理中去小泡的最小面积")
        param_form.addWidget(self.bseg_small_size, 2, 1)

        param_form.addWidget(QLabel("最大尺寸阈值:"), 3, 0)
        self.bseg_maxsize = QSpinBox()
        self.bseg_maxsize.setRange(10, 5000)
        self.bseg_maxsize.setValue(1000)
        self.bseg_maxsize.setToolTip("异常删除中的最大椭圆半轴")
        param_form.addWidget(self.bseg_maxsize, 3, 1)

        param_form.addWidget(QLabel("拟合度阈值:"), 4, 0)
        self.bseg_fit_th = QDoubleSpinBox()
        self.bseg_fit_th.setRange(0.0, 1.0)
        self.bseg_fit_th.setValue(0.65)
        self.bseg_fit_th.setSingleStep(0.05)
        param_form.addWidget(self.bseg_fit_th, 4, 1)

        param_form.addWidget(QLabel("判定方法:"), 5, 0)
        self.bseg_method_combo = QComboBox()
        self.bseg_method_combo.addItems(["长宽比法", "圆度法", "综合法"])
        self.bseg_method_combo.setCurrentIndex(0)
        param_form.addWidget(self.bseg_method_combo, 5, 1)

        param_group.setLayout(param_form)
        left_layout.addWidget(param_group)

        # 处理按钮
        btn_run = QPushButton("▶ 开始处理")
        btn_run.setStyleSheet("""
            QPushButton {
                background-color: #27ae60; color: white; font-size: 14px;
                font-weight: bold; padding: 10px; border-radius: 6px;
            }
            QPushButton:hover { background-color: #2ecc71; }
        """)
        btn_run.clicked.connect(self._run_bub_segment)
        left_layout.addWidget(btn_run)

        left_layout.addStretch()
        layout.addWidget(left_panel)

        # ---- 右侧内容区 ----
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # 预览区
        self.bseg_preview = QLabel()
        self.bseg_preview.setAlignment(Qt.AlignCenter)
        self.bseg_preview.setStyleSheet("border: 1px dashed #ccc; background: #1a1a2e;")
        self.bseg_preview.setMinimumHeight(self._sp(350))
        right_layout.addWidget(self.bseg_preview, stretch=2)

        # 日志区
        log_group = QGroupBox("📋 处理日志")
        log_layout = QVBoxLayout()
        self.bseg_log = QTextEdit()
        self.bseg_log.setReadOnly(True)
        self.bseg_log.setStyleSheet("font-family: Consolas; font-size: 12px; background: #1a1a2e; color: #ecf0f1;")
        log_layout.addWidget(self.bseg_log)
        log_group.setLayout(log_layout)
        right_layout.addWidget(log_group, stretch=1)

        # 统计结果
        stat_group = QGroupBox("📊 统计结果")
        stat_layout = QVBoxLayout()
        self.bseg_stat_label = QLabel("等待处理...")
        self.bseg_stat_label.setStyleSheet("font-size: 13px; padding: 8px;")
        self.bseg_stat_label.setWordWrap(True)
        stat_layout.addWidget(self.bseg_stat_label)
        stat_group.setLayout(stat_layout)
        right_layout.addWidget(stat_group)

        layout.addWidget(right_panel, stretch=1)

        self.content_stack.addWidget(page)

        # 初始化数据
        self._bseg_image_paths = []
        self._bseg_background_path = None
        self._bseg_worker = None
        self._bseg_results = []

    # ---- 气泡分割 UI slots ----

    def _on_bseg_imagesize_toggle(self, state):
        enabled = (state == Qt.Checked)
        self.bseg_row_start.setEnabled(enabled)
        self.bseg_row_end.setEnabled(enabled)
        self.bseg_col_start.setEnabled(enabled)
        self.bseg_col_end.setEnabled(enabled)

    def _load_bub_segment_images(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "选择气泡图像", "",
            "图像文件 (*.bmp *.png *.jpg *.tif *.tiff);;所有文件 (*)"
        )
        if not paths:
            return
        self._bseg_image_paths = sorted(paths)
        self.bseg_img_label.setText(f"已加载 {len(paths)} 张图像")
        self.bseg_img_label.setStyleSheet("color: #00897B; font-weight: bold;")
        self.bseg_log.append(f"[信息] 已加载 {len(paths)} 张气泡图像")

        # 显示第一张预览
        import cv2
        img = cv2.imread(paths[0], cv2.IMREAD_GRAYSCALE)
        if img is not None:
            self._show_bub_preview(img)

    def _load_bub_background(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择背景图像", "",
            "图像文件 (*.bmp *.png *.jpg *.tif *.tiff);;所有文件 (*)"
        )
        if not path:
            return
        self._bseg_background_path = path
        self.bseg_back_label.setText(os.path.basename(path))
        self.bseg_back_label.setStyleSheet("color: #00897B;")
        self.bseg_log.append(f"[信息] 已加载背景图: {os.path.basename(path)}")

    def _show_bub_preview(self, img):
        """在预览区显示灰度图像。"""
        if img is None:
            return
        import cv2
        if len(img.shape) == 2:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            img_bgr = img
        h, w = img_bgr.shape[:2]
        scale = min(self.bseg_preview.width() / w, self.bseg_preview.height() / h, 1.0)
        new_w, new_h = int(w * scale), int(h * scale)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        qimg = QImage(img_rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg).scaled(new_w, new_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.bseg_preview.setPixmap(pixmap)

    def _run_bub_segment(self):
        """启动气泡分割处理。"""
        if not self._bseg_image_paths:
            QMessageBox.warning(self, "警告", "请先加载气泡图像！")
            return

        # 收集参数
        imagesize = None
        if self.bseg_imagesize_check.isChecked():
            imagesize = (
                self.bseg_row_start.value(), self.bseg_row_end.value(),
                self.bseg_col_start.value(), self.bseg_col_end.value(),
            )

        params = {
            'imagesize': imagesize,
            'small_bub_size': self.bseg_small_size.value(),
            'maxsize_th': self.bseg_maxsize.value(),
            'fitting_th': self.bseg_fit_th.value(),
            'method_flag': self.bseg_method_combo.currentIndex(),
            'maxlength_th': 20,
        }

        self.bseg_log.clear()
        self.bseg_log.append(f"[信息] 开始处理 {len(self._bseg_image_paths)} 张图像...")
        self.bseg_stat_label.setText("处理中...")

        self._bseg_worker = _BubbleSegmentWorker(
            self._bseg_image_paths, self._bseg_background_path, params
        )
        self._bseg_worker.progress.connect(self._on_bseg_progress)
        self._bseg_worker.result_ready.connect(self._on_bseg_result)
        self._bseg_worker.error.connect(self._on_bseg_error)
        self._bseg_worker.start()

        self.progress_bar.setVisible(True)

    def _on_bseg_progress(self, current, total, message):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        self.statusBar().showMessage(message)

    def _on_bseg_result(self, data):
        self.progress_bar.setVisible(False)
        self._bseg_results = data['results']
        self.bseg_log.append(f"[完成] 共处理 {len(self._bseg_results)} 张图像")

        # 显示最后一张结果
        if self._bseg_results:
            last = self._bseg_results[-1]
            if last['result_image'] is not None:
                self._show_bub_preview(last['result_image'])

            # 统计汇总
            total_bubbles = sum(r['num_bubbles'] for r in self._bseg_results)
            avg_bubbles = total_bubbles / len(self._bseg_results) if self._bseg_results else 0

            stat_text = f"处理图像数: {len(self._bseg_results)}\n"
            stat_text += f"总识别气泡数: {total_bubbles}\n"
            stat_text += f"平均每张: {avg_bubbles:.1f}\n"
            stat_text += "─" * 30 + "\n"

            if self._bseg_results:
                last_stats = self._bseg_results[-1]['stats']
                if last_stats:
                    stat_text += f"[最后一张图像统计]\n"
                    stat_text += f"  过滤前均值: {last_stats['mean_original']:.2f} px\n"
                    stat_text += f"  过滤后均值: {last_stats['mean_filtered']:.2f} px\n"
                    stat_text += f"  过滤前数量: {last_stats['count_original']}\n"
                    stat_text += f"  过滤后数量: {last_stats['count_filtered']}\n"
                    stat_text += f"  过滤前最大: {last_stats['max_original']:.2f} px\n"
                    stat_text += f"  过滤后最大: {last_stats['max_filtered']:.2f} px\n"

            self.bseg_stat_label.setText(stat_text)
            for r in self._bseg_results:
                self.bseg_log.append(
                    f"  {os.path.basename(r['path'])}: "
                    f"{r['num_bubbles']} 个气泡"
                )

        self.statusBar().showMessage("气泡分割处理完成")

    def _on_bseg_error(self, msg):
        self.progress_bar.setVisible(False)
        self.bseg_log.append(f"[错误] {msg}")
        QMessageBox.critical(self, "处理错误", msg)
        self.statusBar().showMessage("处理出错")

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

            # 缩略图区
            h, w = img.shape[:2]
            scale = min(280 / w, 140 / h)
            new_w, new_h = int(w * scale), int(h * scale)
            if len(img.shape) == 2:
                arr = (img * 255).astype(np.uint8)
                qimg = QImage(arr.data, w, h, w, QImage.Format_Grayscale8)
            else:
                arr = (img * 255).astype(np.uint8)
                qimg = QImage(arr.data, w, h, 3 * w, QImage.Format_RGB888)
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

            # ???
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

            # ???
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
                f"???: {len(self.particles_3d_frame1)} ??\n"
                f"???: {len(self.particles_3d_frame2)} ??")
        except Exception as e:
            QMessageBox.critical(self, "错误", str(e))
            self.piv_log.append(f"??: {e}")
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
                    f"  ?? {name} -> {n2} | "
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
        self.statusBar().showMessage(f"??PIV??: {total} ?")
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

            # ???
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
            self.piv_log.append(f"??: {e}")
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
    #  PTV粒子跟踪页面（Page 6）
    # ------------------------------------------------------------

    def _create_ptv_page(self):
        """创建PTV粒子跟踪页面: 基于PTV_report.docx的4种跟踪算法。"""
        page = QWidget()
        layout = QHBoxLayout(page)

        # ===== 左侧参数面板 =====
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # --- 数据源 ---
        src_group = QGroupBox("数据源")
        src_layout = QVBoxLayout()

        self.ptv_data_source_combo = QComboBox()
        self.ptv_data_source_combo.addItems(["从Particle/PIV模块导入", "手动加载3D粒子数据"])
        src_layout.addWidget(QLabel("数据来源:"))
        src_layout.addWidget(self.ptv_data_source_combo)

        self.ptv_import_btn = QPushButton("从Particle/PIV导入已重建粒子...")
        self.ptv_import_btn.setStyleSheet("QPushButton { font-weight: bold; padding: 6px; }")
        self.ptv_import_btn.clicked.connect(self._ptv_import_from_particle)
        src_layout.addWidget(self.ptv_import_btn)

        self.ptv_load_btn = QPushButton("加载3D粒子数据 (NPZ/CSV)...")
        self.ptv_load_btn.clicked.connect(self._ptv_load_particle_data)
        src_layout.addWidget(self.ptv_load_btn)

        self.ptv_data_info = QLabel("尚未加载粒子数据\n\n支持格式:\n- NPZ: 包含多帧粒子位置的压缩文件\n- CSV: 每帧一个文件，格式: x, y, z")
        self.ptv_data_info.setStyleSheet("color: gray; font-size: 11px;")
        self.ptv_data_info.setWordWrap(True)
        src_layout.addWidget(self.ptv_data_info)

        src_group.setLayout(src_layout)
        left_layout.addWidget(src_group)

        # --- 跟踪算法选择 ---
        algo_group = QGroupBox("跟踪算法")
        algo_layout = QVBoxLayout()

        self.ptv_algo_combo = QComboBox()
        self.ptv_algo_combo.addItems([
            "四帧前后跟踪 (Forward-Backward)",
            "最近邻跟踪 (Nearest Neighbor)",
            "松弛法跟踪 (Relaxation)",
            "Shake-The-Box (STB)",
        ])
        self.ptv_algo_combo.currentIndexChanged.connect(self._on_ptv_algo_changed)
        algo_layout.addWidget(QLabel("算法:"))
        algo_layout.addWidget(self.ptv_algo_combo)

        self.ptv_algo_desc = QLabel(
            "四帧前后跟踪法:\n"
            "使用连续4帧 (i-1, i, i+1, i+2)，通过前向和后向\n"
            "跟踪的一致性验证剔除错误匹配。\n"
            "适用于中等粒子密度的场景。"
        )
        self.ptv_algo_desc.setStyleSheet("color: #555; font-size: 11px; padding: 4px;")
        self.ptv_algo_desc.setWordWrap(True)
        algo_layout.addWidget(self.ptv_algo_desc)

        algo_group.setLayout(algo_layout)
        left_layout.addWidget(algo_group)

        # --- 通用跟踪参数 ---
        param_group = QGroupBox("跟踪参数")
        param_grid = QGridLayout()

        param_grid.addWidget(QLabel("最大位移 (mm/frame):"), 0, 0)
        self.ptv_max_disp = QDoubleSpinBox()
        self.ptv_max_disp.setRange(0.1, 50.0)
        self.ptv_max_disp.setValue(5.0)
        self.ptv_max_disp.setSingleStep(0.5)
        param_grid.addWidget(self.ptv_max_disp, 0, 1)

        param_grid.addWidget(QLabel("最短轨迹长度 (帧):"), 1, 0)
        self.ptv_min_track_len = QSpinBox()
        self.ptv_min_track_len.setRange(2, 100)
        self.ptv_min_track_len.setValue(4)
        param_grid.addWidget(self.ptv_min_track_len, 1, 1)

        param_grid.addWidget(QLabel("dt (s):"), 2, 0)
        self.ptv_dt = QDoubleSpinBox()
        self.ptv_dt.setRange(0.00001, 10.0)
        self.ptv_dt.setValue(0.001)
        self.ptv_dt.setDecimals(5)
        self.ptv_dt.setSingleStep(0.0001)
        param_grid.addWidget(self.ptv_dt, 2, 1)

        param_group.setLayout(param_grid)
        left_layout.addWidget(param_group)

        # --- 高级参数（按算法显示不同参数）---
        self.ptv_advanced_group = QGroupBox("高级参数")
        self.ptv_advanced_layout = QGridLayout()

        # 前后跟踪参数
        self.ptv_advanced_layout.addWidget(QLabel("速度比阈值:"), 0, 0)
        self.ptv_speed_ratio = QDoubleSpinBox()
        self.ptv_speed_ratio.setRange(1.1, 10.0)
        self.ptv_speed_ratio.setValue(2.0)
        self.ptv_speed_ratio.setSingleStep(0.1)
        self.ptv_advanced_layout.addWidget(self.ptv_speed_ratio, 0, 1)

        self.ptv_advanced_layout.addWidget(QLabel("加速度限制 (mm/f^2):"), 1, 0)
        self.ptv_accel_limit = QDoubleSpinBox()
        self.ptv_accel_limit.setRange(0.1, 50.0)
        self.ptv_accel_limit.setValue(3.0)
        self.ptv_accel_limit.setSingleStep(0.5)
        self.ptv_advanced_layout.addWidget(self.ptv_accel_limit, 1, 1)

        # 松弛法参数 (初始隐藏)
        self.ptv_adv_iter_label = QLabel("松弛迭代次数:")
        self.ptv_adv_iter_spin = QSpinBox()
        self.ptv_adv_iter_spin.setRange(1, 100)
        self.ptv_adv_iter_spin.setValue(10)
        self.ptv_adv_iter_label.hide()
        self.ptv_adv_iter_spin.hide()
        self.ptv_advanced_layout.addWidget(self.ptv_adv_iter_label, 2, 0)
        self.ptv_advanced_layout.addWidget(self.ptv_adv_iter_spin, 2, 1)

        self.ptv_adv_neighbor_label = QLabel("邻域粒子数:")
        self.ptv_adv_neighbor_spin = QSpinBox()
        self.ptv_adv_neighbor_spin.setRange(3, 30)
        self.ptv_adv_neighbor_spin.setValue(6)
        self.ptv_adv_neighbor_label.hide()
        self.ptv_adv_neighbor_spin.hide()
        self.ptv_advanced_layout.addWidget(self.ptv_adv_neighbor_label, 3, 0)
        self.ptv_advanced_layout.addWidget(self.ptv_adv_neighbor_spin, 3, 1)

        # STB参数 (初始隐藏)
        self.ptv_adv_stb_iter_label = QLabel("STB迭代次数:")
        self.ptv_adv_stb_iter_spin = QSpinBox()
        self.ptv_adv_stb_iter_spin.setRange(10, 500)
        self.ptv_adv_stb_iter_spin.setValue(50)
        self.ptv_adv_stb_iter_label.hide()
        self.ptv_adv_stb_iter_spin.hide()
        self.ptv_advanced_layout.addWidget(self.ptv_adv_stb_iter_label, 4, 0)
        self.ptv_advanced_layout.addWidget(self.ptv_adv_stb_iter_spin, 4, 1)

        self.ptv_adv_shake_label = QLabel("初始扰动幅度 (mm):")
        self.ptv_adv_shake_spin = QDoubleSpinBox()
        self.ptv_adv_shake_spin.setRange(0.01, 10.0)
        self.ptv_adv_shake_spin.setValue(1.0)
        self.ptv_adv_shake_spin.setSingleStep(0.1)
        self.ptv_adv_shake_label.hide()
        self.ptv_adv_shake_spin.hide()
        self.ptv_advanced_layout.addWidget(self.ptv_adv_shake_label, 5, 0)
        self.ptv_advanced_layout.addWidget(self.ptv_adv_shake_spin, 5, 1)

        self.ptv_advanced_group.setLayout(self.ptv_advanced_layout)
        left_layout.addWidget(self.ptv_advanced_group)

        # --- 操作按钮 ---
        btn_layout = QVBoxLayout()

        self.ptv_run_btn = QPushButton("执行PTV跟踪")
        self.ptv_run_btn.setStyleSheet(
            "QPushButton { background-color: #2196F3; color: white; "
            "font-size: 14px; padding: 10px; border-radius: 5px; }"
            "QPushButton:hover { background-color: #1976D2; }"
            "QPushButton:disabled { background-color: #BDBDBD; }"
        )
        self.ptv_run_btn.clicked.connect(self._run_ptv_tracking)
        btn_layout.addWidget(self.ptv_run_btn)

        self.ptv_velocity_btn = QPushButton("计算速度场")
        self.ptv_velocity_btn.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; "
            "font-size: 13px; padding: 8px; border-radius: 5px; }"
            "QPushButton:hover { background-color: #388E3C; }"
            "QPushButton:disabled { background-color: #BDBDBD; }"
        )
        self.ptv_velocity_btn.clicked.connect(self._run_ptv_velocity)
        self.ptv_velocity_btn.setEnabled(False)
        btn_layout.addWidget(self.ptv_velocity_btn)

        self.ptv_export_btn = QPushButton("导出轨迹 (CSV)...")
        self.ptv_export_btn.setEnabled(False)
        self.ptv_export_btn.clicked.connect(self._ptv_export_tracks)
        btn_layout.addWidget(self.ptv_export_btn)

        left_layout.addLayout(btn_layout)
        left_layout.addStretch()

        # ===== 右侧结果面板 =====
        right_container = QWidget()
        right_container_layout = QVBoxLayout(right_container)
        right_container_layout.setContentsMargins(self._sp(6), self._sp(6), self._sp(6), self._sp(6))

        # 日志区
        self.ptv_log = QTextEdit()
        self.ptv_log.setReadOnly(True)
        self.ptv_log.setFontFamily("Consolas")
        self.ptv_log.setPlaceholderText("PTV跟踪日志...")
        right_container_layout.addWidget(self.ptv_log, stretch=1)

        # 预览区
        self.ptv_preview = QLabel()
        self.ptv_preview.setAlignment(Qt.AlignCenter)
        self.ptv_preview.setMinimumSize(self._sp(400), self._sp(300))
        self.ptv_preview.setStyleSheet(
            "border: 1px solid #ccc; background-color: #f8f8f8; padding: 10px;"
        )
        self.ptv_preview.setText("PTV跟踪与速度场结果预览")
        right_container_layout.addWidget(self.ptv_preview)

        # 可视化工具栏
        viz_toolbar = self._make_viz_toolbar("ptv")
        right_container_layout.addWidget(viz_toolbar)

        # 进度条
        self.ptv_progress = QProgressBar()
        self.ptv_progress.setVisible(False)
        right_container_layout.addWidget(self.ptv_progress)

        # 统计信息
        self.ptv_stats_label = QLabel("等待执行PTV跟踪...")
        self.ptv_stats_label.setStyleSheet("color: #666; font-size: 11px; padding: 4px;")
        self.ptv_stats_label.setWordWrap(True)
        right_container_layout.addWidget(self.ptv_stats_label)

        layout.addWidget(left_panel, stretch=1)
        layout.addWidget(right_container, stretch=2)

        self.content_stack.addWidget(page)

        # PTV数据缓存
        self.ptv_frames_particles: Dict[int, np.ndarray] = {}
        self.ptv_frame_indices: List[int] = []
        self.ptv_tracking_result: Optional[object] = None
        self.ptv_velocity_result: Optional[VelocityProfile] = None

    def _on_ptv_algo_changed(self, index: int):
        """切换跟踪算法时更新高级参数面板"""
        # 默认隐藏所有高级参数
        self.ptv_speed_ratio.parent().show()
        self.ptv_accel_limit.parent().show()

        # 松弛法参数
        is_relaxation = (index == 2)
        self.ptv_adv_iter_label.setVisible(is_relaxation)
        self.ptv_adv_iter_spin.setVisible(is_relaxation)
        self.ptv_adv_neighbor_label.setVisible(is_relaxation)
        self.ptv_adv_neighbor_spin.setVisible(is_relaxation)

        # STB参数
        is_stb = (index == 3)
        self.ptv_adv_stb_iter_label.setVisible(is_stb)
        self.ptv_adv_stb_iter_spin.setVisible(is_stb)
        self.ptv_adv_shake_label.setVisible(is_stb)
        self.ptv_adv_shake_spin.setVisible(is_stb)

        # 前后跟踪参数只在前后跟踪模式下显示
        is_fb = (index == 0)
        self.ptv_speed_ratio.setVisible(is_fb)
        self.ptv_accel_limit.setVisible(is_fb)

        algo_names = ["forward_backward", "nearest_neighbor", "relaxation", "stb"]
        descriptions = [
            "四帧前后跟踪法:\n"
            "使用连续4帧 (i-1, i, i+1, i+2)，通过前向和后向\n"
            "跟踪的一致性验证剔除错误匹配。\n"
            "适用于中等粒子密度的场景。",

            "最近邻跟踪法:\n"
            "假设粒子在短时间内位移有限，取空间距离最近\n"
            "的粒子作为匹配目标。使用KDTree加速搜索。\n"
            "适用于低粒子密度、小位移场景。",

            "松弛法跟踪:\n"
            "迭代优化方法，利用位移场的空间平滑性约束。\n"
            "每次迭代中，根据邻域粒子的加权平均位移更新\n"
            "匹配结果，逐步收敛到最优匹配。\n"
            "适用于中高密度粒子场景。",

            "Shake-The-Box (STB):\n"
            "高密度3D-PTV优化方法 (Schanz et al., 2016)。\n"
            "通过迭代扰动粒子位置并投影验证，精化粒子\n"
            "三维坐标。适用于高浓度粒子场景 (source density > 0.05 ppp)。\n"
            "需要标定参数和原始图像。"
        ]

        self.ptv_algo_desc.setText(descriptions[index])

    def _build_ptv_config(self) -> PTVConfig:
        """从GUI参数构建PTVConfig"""
        algo_map = ["forward_backward", "nearest_neighbor", "relaxation", "stb"]
        idx = self.ptv_algo_combo.currentIndex()

        config = PTVConfig(
            tracking_method=algo_map[idx],
            max_displacement=self.ptv_max_disp.value(),
            min_track_length=self.ptv_min_track_len.value(),
            dt=self.ptv_dt.value(),
            fb_max_speed_ratio=self.ptv_speed_ratio.value(),
            fb_acceleration_limit=self.ptv_accel_limit.value(),
            relaxation_iterations=self.ptv_adv_iter_spin.value(),
            relaxation_neighbors=self.ptv_adv_neighbor_spin.value(),
            stb_iterations=self.ptv_adv_stb_iter_spin.value(),
            stb_shake_amplitude=self.ptv_adv_shake_spin.value(),
        )
        return config

    def _ptv_import_from_particle(self):
        """从Particle/PIV模块导入已重建的3D粒子数据"""
        if not self.particles_3d_frame1 and not self.particles_3d_frame2:
            QMessageBox.warning(
                self, "无数据",
                "请先在Particle/PIV页面执行粒子3D重建。\n"
                "需要至少加载两帧并完成重建。"
            )
            return

        self.ptv_log.clear()
        self.ptv_log.append("=== 从Particle/PIV模块导入粒子数据 ===\n")

        # 收集所有已重建的粒子
        # 使用批量PIV结果如果有的话，否则使用单帧结果
        if hasattr(self, '_batch_piv_results') and self._batch_piv_results:
            self.ptv_log.append("检测到批量PIV结果，正在提取...")

            for tp_idx in sorted(self._batch_piv_results.keys()):
                result = self._batch_piv_results[tp_idx]
                p3d = result.get('particles_3d_frame1', [])
                if p3d:
                    positions = np.array([p.position for p in p3d])
                    self.ptv_frames_particles[tp_idx] = positions
                    self.ptv_log.append(f"  时刻 t{tp_idx}: {len(p3d)} 个粒子")

                p3d_2 = result.get('particles_3d_frame2', [])
                if p3d_2:
                    # 最后一对的后帧作为额外时间点
                    next_tp = tp_idx + 1
                    if next_tp not in self.ptv_frames_particles:
                        positions = np.array([p.position for p in p3d_2])
                        self.ptv_frames_particles[next_tp] = positions
                        self.ptv_log.append(f"  时刻 t{next_tp}: {len(p3d_2)} 个粒子")

        elif self.particles_3d_frame1:
            p3d_1 = self.particles_3d_frame1
            p3d_2 = self.particles_3d_frame2

            pos1 = np.array([p.position for p in p3d_1])
            self.ptv_frames_particles[0] = pos1
            self.ptv_log.append(f"第1帧: {len(p3d_1)} 个粒子")

            if p3d_2:
                pos2 = np.array([p.position for p in p3d_2])
                self.ptv_frames_particles[1] = pos2
                self.ptv_log.append(f"第2帧: {len(p3d_2)} 个粒子")

        self.ptv_frame_indices = sorted(self.ptv_frames_particles.keys())

        total = sum(len(p) for p in self.ptv_frames_particles.values())
        self.ptv_data_info.setText(
            f"已导入 {len(self.ptv_frame_indices)} 个时刻\n"
            f"总计 {total} 个粒子\n"
            f"时刻: {self.ptv_frame_indices}"
        )
        self.ptv_data_info.setStyleSheet("color: #2196F3; font-size: 11px;")
        self.ptv_run_btn.setEnabled(True)
        self.ptv_log.append(f"\n数据导入完成。共 {len(self.ptv_frame_indices)} 个时刻。")

    def _ptv_load_particle_data(self):
        """加载3D粒子数据文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "加载粒子数据", "",
            "粒子数据 (*.npz *.csv);;所有文件 (*)"
        )
        if not file_path:
            return

        self.ptv_log.clear()
        self.ptv_log.append(f"=== 加载粒子数据: {os.path.basename(file_path)} ===\n")

        try:
            if file_path.endswith('.npz'):
                data = np.load(file_path, allow_pickle=True)
                # 支持两种格式:
                # 1) 'particles_0', 'particles_1', ... 每帧一组
                # 2) 'all_particles' (N, M, 3) 所有帧
                self.ptv_frames_particles = {}

                for key in data.files:
                    if key.startswith('particles_') or key.startswith('frame_'):
                        idx = int(''.join(filter(str.isdigit, key)))
                        particles = data[key]
                        if particles.ndim == 2 and particles.shape[1] == 3:
                            self.ptv_frames_particles[idx] = particles
                            self.ptv_log.append(f"  时刻 {idx}: {len(particles)} 个粒子")

                if not self.ptv_frames_particles:
                    # 尝试 all_particles 格式
                    if 'all_particles' in data:
                        all_p = data['all_particles']
                        if all_p.ndim == 3:
                            for i in range(all_p.shape[0]):
                                self.ptv_frames_particles[i] = all_p[i]
                                self.ptv_log.append(f"  时刻 {i}: {len(all_p[i])} 个粒子")

                data.close()

            elif file_path.endswith('.csv'):
                # CSV格式: 每行 x, y, z[, frame_id]
                raw = np.genfromtxt(file_path, delimiter=',')
                if raw.ndim != 2:
                    raise ValueError("CSV文件格式错误")

                if raw.shape[1] >= 4:
                    # 有帧ID列
                    frame_ids = raw[:, 3].astype(int)
                    unique_frames = sorted(set(frame_ids))
                    for fi in unique_frames:
                        mask = frame_ids == fi
                        self.ptv_frames_particles[fi] = raw[mask, :3]
                        self.ptv_log.append(f"  时刻 {fi}: {mask.sum()} 个粒子")
                else:
                    # 单帧
                    self.ptv_frames_particles[0] = raw[:, :3]
                    self.ptv_log.append(f"  加载 {len(raw)} 个粒子 (单帧)")
            else:
                raise ValueError("不支持的文件格式")

            self.ptv_frame_indices = sorted(self.ptv_frames_particles.keys())
            total = sum(len(p) for p in self.ptv_frames_particles.values())

            self.ptv_data_info.setText(
                f"已加载: {os.path.basename(file_path)}\n"
                f"{len(self.ptv_frame_indices)} 个时刻, {total} 个粒子\n"
                f"时刻: {self.ptv_frame_indices}"
            )
            self.ptv_data_info.setStyleSheet("color: #2196F3; font-size: 11px;")
            self.ptv_run_btn.setEnabled(True)
            self.ptv_log.append(f"\n加载完成: {total} 个粒子, {len(self.ptv_frame_indices)} 个时刻")

        except Exception as e:
            import traceback
            self.ptv_log.append(f"加载失败: {e}\n{traceback.format_exc()}")
            QMessageBox.critical(self, "加载失败", str(e))

    def _run_ptv_tracking(self):
        """执行PTV跟踪"""
        if not self.ptv_frames_particles:
            QMessageBox.warning(self, "无数据", "请先加载粒子数据")
            return

        config = self._build_ptv_config()
        self.ptv_log.append(f"\n{'='*50}")
        self.ptv_log.append(f"开始PTV跟踪...")
        self.ptv_log.append(f"  算法: {self.ptv_algo_combo.currentText()}")
        self.ptv_log.append(f"  帧数: {len(self.ptv_frame_indices)}")
        self.ptv_log.append(f"  最大位移: {config.max_displacement} mm")
        self.ptv_log.append(f"  最短轨迹: {config.min_track_length} 帧")
        self.ptv_log.append(f"  dt: {config.dt * 1000:.2f} ms")
        self.ptv_log.append(f"{'='*50}\n")

        self.ptv_run_btn.setEnabled(False)
        self.ptv_progress.setVisible(True)
        self.ptv_progress.setValue(0)

        # 获取标定参数（STB需要）
        calibrator = self.calibration_results_calibrator if hasattr(self, 'calibration_results_calibrator') else None

        worker = PTVBatchWorker(
            frames_particles=self.ptv_frames_particles,
            frame_indices=self.ptv_frame_indices,
            ptv_config=config,
            calibrator=calibrator,
        )
        worker.progress.connect(lambda msg: self.ptv_log.append(msg))
        worker.progress.connect(lambda msg: self.statusBar().showMessage(msg))
        worker.finished.connect(self._on_ptv_tracking_done)
        worker.error.connect(self._on_ptv_tracking_error)
        worker.finished.connect(lambda: self.ptv_progress.setVisible(False))
        worker.error.connect(lambda: self.ptv_progress.setVisible(False))
        self._ptv_worker = worker
        worker.start()

    def _on_ptv_tracking_done(self, result):
        """PTV跟踪完成"""
        self.ptv_tracking_result = result
        self.ptv_run_btn.setEnabled(True)
        self.ptv_velocity_btn.setEnabled(True)
        self.ptv_export_btn.setEnabled(True)

        # 显示统计
        self.ptv_log.append(f"\n{result.summary()}")
        self.ptv_stats_label.setText(
            f"轨迹数: {result.n_tracks} | "
            f"平均长度: {result.avg_track_length:.1f} 帧 | "
            f"跟踪效率: {result.tracking_efficiency:.1%}"
        )
        self.statusBar().showMessage(f"PTV跟踪完成: {result.n_tracks} 条轨迹")

        # 可视化
        self._visualize_ptv_tracks(result)

    def _on_ptv_tracking_error(self, msg):
        """PTV跟踪错误"""
        self.ptv_run_btn.setEnabled(True)
        self.ptv_log.append(f"错误: {msg}")
        QMessageBox.critical(self, "PTV跟踪错误", msg)

    def _visualize_ptv_tracks(self, result):
        """可视化PTV轨迹"""
        if not result.tracks:
            return

        fig = Figure(figsize=(8, 6), tight_layout=True)
        ax = fig.add_subplot(111, projection='3d')

        # 绘制轨迹
        colors = np.random.rand(len(result.tracks), 3)
        for i, track in enumerate(result.tracks):
            positions = np.array(track.positions)
            ax.plot(
                positions[:, 0], positions[:, 1], positions[:, 2],
                color=colors[i], alpha=0.6, linewidth=0.8
            )
            # 起点标记
            ax.scatter(*positions[0], color=colors[i], s=10, marker='o')

        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.set_title(f'PTV轨迹 ({result.n_tracks} 条)')

        # 保存到预览
        canvas = FigureCanvas(fig)
        canvas.draw()

        # 转为QPixmap
        buf = canvas.buffer()
        w, h = canvas.get_width_height()
        qimg = QImage(buf, w, h, w * 4, QImage.Format_RGBA8888)
        pixmap = QPixmap.fromImage(qimg)
        self.ptv_preview.setPixmap(
            pixmap.scaled(self.ptv_preview.size(),
                         Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    def _run_ptv_velocity(self):
        """从PTV轨迹计算速度场"""
        if self.ptv_tracking_result is None:
            QMessageBox.warning(self, "无跟踪结果", "请先执行PTV跟踪")
            return

        dt = self.ptv_dt.value()
        domain = (20.0, 20.0, 20.0)  # 使用默认域大小

        self.ptv_log.append(f"\n{'='*50}")
        self.ptv_log.append("计算PTV速度场...")
        self.ptv_log.append(f"  dt = {dt * 1000:.2f} ms")
        self.ptv_log.append(f"{'='*50}\n")

        try:
            calculator = PTVVelocityCalculator(dt=dt, domain_size=domain)
            profile = calculator.compute_from_tracks(self.ptv_tracking_result.tracks)

            self.ptv_velocity_result = profile

            # 插值到网格
            if len(profile.positions) >= 10:
                grid_pos, grid_vel, grid_shape = calculator.interpolate_to_grid(
                    profile.positions, profile.velocities,
                    grid_resolution=(16, 16, 16)
                )
                profile.grid_positions = grid_pos
                profile.grid_velocities = grid_vel
                profile.grid_shape = grid_shape

            self.ptv_log.append(profile.summary())
            self.ptv_stats_label.setText(
                f"数据点: {len(profile.positions)} | "
                f"平均速度: {profile.mean_speed:.2f} mm/s | "
                f"最大速度: {profile.max_speed:.2f} mm/s"
            )

            # 可视化速度场
            self._visualize_ptv_velocity(profile)

            self.statusBar().showMessage(f"PTV速度场计算完成: 平均 {profile.mean_speed:.2f} mm/s")

        except Exception as e:
            import traceback
            self.ptv_log.append(f"速度场计算错误: {e}\n{traceback.format_exc()}")
            QMessageBox.critical(self, "计算错误", str(e))

    def _visualize_ptv_velocity(self, profile: VelocityProfile):
        """可视化PTV速度场"""
        fig = Figure(figsize=(10, 4), tight_layout=True)

        # 子图1: 速度大小分布
        ax1 = fig.add_subplot(121)
        speeds = np.linalg.norm(profile.velocities, axis=1)
        ax1.hist(speeds, bins=30, color='#2196F3', alpha=0.7, edgecolor='white')
        ax1.set_xlabel('速度大小 (mm/s)')
        ax1.set_ylabel('粒子数')
        ax1.set_title('速度大小分布')
        ax1.axvline(profile.mean_speed, color='red', linestyle='--',
                    label=f'平均: {profile.mean_speed:.1f} mm/s')
        ax1.legend()

        # 子图2: 3D轨迹+速度箭头
        ax2 = fig.add_subplot(122, projection='3d')
        if len(profile.positions) > 0 and len(profile.velocities) > 0:
            sc = ax2.scatter(
                profile.positions[:, 0],
                profile.positions[:, 1],
                profile.positions[:, 2],
                c=speeds, cmap='jet', s=5, alpha=0.5
            )

            # 速度箭头（每隔一些采样）
            step = max(1, len(profile.positions) // 50)
            for i in range(0, len(profile.positions), step):
                pos = profile.positions[i]
                vel = profile.velocities[i]
                speed = np.linalg.norm(vel)
                if speed > 0.01:
                    scale = 2.0 / (profile.max_speed + 1e-10)
                    ax2.quiver(
                        pos[0], pos[1], pos[2],
                        vel[0] * scale, vel[1] * scale, vel[2] * scale,
                        color='red', alpha=0.4, linewidth=0.5
                    )

            fig.colorbar(sc, ax=ax2, label='速度 (mm/s)', shrink=0.6)

        ax2.set_xlabel('X (mm)')
        ax2.set_ylabel('Y (mm)')
        ax2.set_zlabel('Z (mm)')
        ax2.set_title('PTV速度场')

        canvas = FigureCanvas(fig)
        canvas.draw()
        buf = canvas.buffer()
        w, h = canvas.get_width_height()
        qimg = QImage(buf, w, h, w * 4, QImage.Format_RGBA8888)
        pixmap = QPixmap.fromImage(qimg)
        self.ptv_preview.setPixmap(
            pixmap.scaled(self.ptv_preview.size(),
                         Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    def _ptv_export_tracks(self):
        """导出PTV轨迹到CSV"""
        if self.ptv_tracking_result is None:
            QMessageBox.warning(self, "无数据", "请先执行PTV跟踪")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "导出PTV轨迹", "ptv_tracks.csv",
            "CSV文件 (*.csv);;所有文件 (*)"
        )
        if not file_path:
            return

        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write("track_id,frame_idx,x,y,z,vx,vy,vz,speed,quality\n")
                for track in self.ptv_tracking_result.tracks:
                    positions, velocities, frames = track.to_arrays()
                    for j in range(len(positions)):
                        f.write(f"{track.particle_id},{frames[j]},")
                        f.write(f"{positions[j, 0]:.6f},{positions[j, 1]:.6f},{positions[j, 2]:.6f},")
                        if j < len(velocities):
                            speed = np.linalg.norm(velocities[j])
                            f.write(f"{velocities[j, 0]:.6f},{velocities[j, 1]:.6f},{velocities[j, 2]:.6f},")
                            f.write(f"{speed:.6f},")
                        else:
                            f.write("0,0,0,0,")
                        f.write(f"{track.quality:.4f}\n")

            n_points = sum(t.length for t in self.ptv_tracking_result.tracks)
            self.ptv_log.append(f"\n轨迹已导出: {file_path}")
            self.ptv_log.append(f"  {self.ptv_tracking_result.n_tracks} 条轨迹, {n_points} 个数据点")
            self.statusBar().showMessage(f"已导出: {os.path.basename(file_path)}")
            QMessageBox.information(self, "导出成功", f"轨迹已保存到:\n{file_path}")

        except Exception as e:
            QMessageBox.critical(self, "导出失败", str(e))

    # ------------------------------------------------------------
    #  二维PIV页面（Page 7）
    # ------------------------------------------------------------

    def _create_piv2d_page(self):
        page = QWidget()
        layout = QHBoxLayout(page)

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFixedWidth(self._sp(380))

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

        cfg_group = QGroupBox("互相关参数")
        cfg_layout = QGridLayout(cfg_group)
        cfg_layout.addWidget(QLabel("窗口尺寸:"), 0, 0)
        self.piv2d_win_spin = QSpinBox()
        self.piv2d_win_spin.setRange(8, 256)
        self.piv2d_win_spin.setSingleStep(4)
        self.piv2d_win_spin.setValue(32)
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
        cfg_layout.addWidget(QLabel("像素尺度(mm/px):"), 4, 0)
        self.piv2d_scale_spin = QDoubleSpinBox()
        self.piv2d_scale_spin.setRange(1e-6, 1e6)
        self.piv2d_scale_spin.setDecimals(6)
        self.piv2d_scale_spin.setValue(1.0)
        self.piv2d_scale_spin.setToolTip("每像素对应的物理尺寸(mm)，用于速度换算和标尺条显示")
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
        self.piv2d_vector_color_combo.addItems(["按速度着色", "红色", "绿色", "蓝色", "黄色", "白色"])
        self.piv2d_vector_color_combo.currentIndexChanged.connect(self._piv2d_refresh_vector_preview)
        viz_layout.addWidget(self.piv2d_vector_color_combo, 1, 1)
        viz_layout.addWidget(QLabel("线宽:"), 2, 0)
        self.piv2d_vector_width_spin = QDoubleSpinBox()
        self.piv2d_vector_width_spin.setRange(0.0005, 0.05)
        self.piv2d_vector_width_spin.setDecimals(4)
        self.piv2d_vector_width_spin.setSingleStep(0.0005)
        self.piv2d_vector_width_spin.setValue(0.003)
        self.piv2d_vector_width_spin.setToolTip("箭杆宽度（归一化值，越小越细）")
        self.piv2d_vector_width_spin.valueChanged.connect(self._piv2d_refresh_vector_preview)
        viz_layout.addWidget(self.piv2d_vector_width_spin, 2, 1)
        viz_layout.addWidget(QLabel("箭头宽度:"), 3, 0)
        self.piv2d_vector_headwidth_spin = QDoubleSpinBox()
        self.piv2d_vector_headwidth_spin.setRange(0.5, 15.0)
        self.piv2d_vector_headwidth_spin.setSingleStep(0.5)
        self.piv2d_vector_headwidth_spin.setValue(4.0)
        self.piv2d_vector_headwidth_spin.setToolTip("箭头三角宽度（倍数）")
        self.piv2d_vector_headwidth_spin.valueChanged.connect(self._piv2d_refresh_vector_preview)
        viz_layout.addWidget(self.piv2d_vector_headwidth_spin, 3, 1)
        viz_layout.addWidget(QLabel("箭头长度:"), 4, 0)
        self.piv2d_vector_headlength_spin = QDoubleSpinBox()
        self.piv2d_vector_headlength_spin.setRange(1.0, 20.0)
        self.piv2d_vector_headlength_spin.setSingleStep(0.5)
        self.piv2d_vector_headlength_spin.setValue(5.0)
        self.piv2d_vector_headlength_spin.setToolTip("箭头三角长度（倍数）")
        self.piv2d_vector_headlength_spin.valueChanged.connect(self._piv2d_refresh_vector_preview)
        viz_layout.addWidget(self.piv2d_vector_headlength_spin, 4, 1)
        left_layout.addWidget(viz_group)

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
        top_layout.addWidget(self.piv2d_frame1_preview)
        self.piv2d_frame2_preview = PIV2DPreviewWidget("第2帧预览")
        self.piv2d_frame2_preview.setMinimumSize(self._sp(260), self._sp(220))
        top_layout.addWidget(self.piv2d_frame2_preview)
        preview_splitter.addWidget(top_widget)

        self.piv2d_result_preview = PIV2DPreviewWidget("速度场预览")
        self.piv2d_result_preview.setMinimumSize(self._sp(500), self._sp(280))
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
        )

    def _piv2d_run_single(self):
        import cv2 as _cv2
        if not self.piv2d_frame1_path or not self.piv2d_frame2_path:
            QMessageBox.warning(self, "提示", "请先选择两帧图像")
            return

        img1 = _cv2.imread(self.piv2d_frame1_path, _cv2.IMREAD_UNCHANGED)
        img2 = _cv2.imread(self.piv2d_frame2_path, _cv2.IMREAD_UNCHANGED)
        if img1 is None or img2 is None:
            QMessageBox.warning(self, "提示", "图像读取失败")
            return

        try:
            calculator = PIV2DCalculator(self._piv2d_get_config())
            result = calculator.compute_velocity_field(img1, img2)
            summary = calculator.summarize_result(result)
            self._piv2d_last_result = result
            self._piv2d_last_image = img1
            self._piv2d_show_vector_result(img1, result, "单组速度矢量")

            self.piv2d_log.append("=== 二维PIV单组计算完成 ===")
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

        self.piv2d_progress.setValue(0)
        self.piv2d_progress.setVisible(True)
        self.piv2d_batch_run_btn.setEnabled(False)
        self.piv2d_stop_btn.setEnabled(True)
        self.piv2d_log.append(f"=== 开始批量二维PIV ===\n输入: {self.piv2d_src_dir}\n输出: {self.piv2d_dst_dir}")

        self.piv2d_worker_thread = BatchPIV2DWorker(
            self.piv2d_src_dir,
            self.piv2d_dst_dir,
            self._piv2d_get_config(),
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
        self.piv2d_log.append(f"[{done}/{total}] {name}")

    def _piv2d_on_batch_finished(self, success, total, outputs):
        self.piv2d_progress.setVisible(False)
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
        self.piv2d_batch_run_btn.setEnabled(True)
        self.piv2d_stop_btn.setEnabled(False)
        self.piv2d_log.append(f"批量二维PIV出错: {msg}")
        QMessageBox.critical(self, "二维PIV错误", msg)

    def _piv2d_set_preview(self, path: str, label: QLabel):
        import cv2 as _cv2
        img = _cv2.imread(path, _cv2.IMREAD_UNCHANGED)
        if img is None:
            return
        if hasattr(label, "set_image_array"):
            label.set_image_array(img, os.path.basename(path))
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
            width=self.piv2d_vector_width_spin.value(),
            headwidth=self.piv2d_vector_headwidth_spin.value(),
            headlength=self.piv2d_vector_headlength_spin.value(),
        )
        # 设置物理尺度标尺条（由像素尺度参数控制，mm/px）
        pixel_scale_mm = self.piv2d_scale_spin.value() if hasattr(self, "piv2d_scale_spin") else None
        self.piv2d_result_preview.set_pixel_scale(pixel_scale_mm)
        self.piv2d_result_preview.set_vector_result(image, result, title)

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

        image = cv2.imread(str(files[0]), cv2.IMREAD_UNCHANGED)
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
    #  通用图像处理页面（Page 5）
    # ------------------------------------------------------------

    def _create_image_editor_page(self):
        """创建图像处理页面：单张预览 + 批量处理。"""
        from utils.image_editor import ImageEditor as _IE

        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        body_splitter = QSplitter(Qt.Horizontal)

        # File and directory selectors
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFixedWidth(self._sp(380))

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(self._sp(6), self._sp(6), self._sp(6), self._sp(6))
        left_layout.setSpacing(self._sp(4))

        # 文件 / 目录选择
        src_group = QGroupBox("📂 输入")
        src_layout = QGridLayout(src_group)

        src_layout.addWidget(QLabel("处理模式:"), 0, 0)
        self.ie_mode_combo = QComboBox()
        self.ie_mode_combo.addItems(["单张图像", "批量目录"])
        self.ie_mode_combo.currentIndexChanged.connect(self._ie_on_mode_changed)
        src_layout.addWidget(self.ie_mode_combo, 0, 1)

        # 单张模式
        self.ie_single_widget = QWidget()
        sg_lay = QVBoxLayout(self.ie_single_widget)
        sg_lay.setContentsMargins(0, 0, 0, 0)
        btn_open_single = QPushButton("📄 选择图像文件...")
        btn_open_single.clicked.connect(self._ie_open_single)
        sg_lay.addWidget(btn_open_single)
        self.ie_single_label = QLabel("未选择文件")
        self.ie_single_label.setWordWrap(True)
        self.ie_single_label.setStyleSheet("color: gray;")
        sg_lay.addWidget(self.ie_single_label)
        src_layout.addWidget(self.ie_single_widget, 1, 0, 1, 2)

        # 批量模式
        self.ie_batch_widget = QWidget()
        bg_lay = QGridLayout(self.ie_batch_widget)
        bg_lay.setContentsMargins(0, 0, 0, 0)
        btn_src_dir = QPushButton("输入目录...")
        btn_src_dir.clicked.connect(self._ie_pick_src_dir)
        bg_lay.addWidget(btn_src_dir, 0, 0)
        self.ie_src_label = QLabel("未选择")
        self.ie_src_label.setStyleSheet("color: gray;")
        self.ie_src_label.setWordWrap(True)
        bg_lay.addWidget(self.ie_src_label, 1, 0)
        btn_dst_dir = QPushButton("输出目录...")
        btn_dst_dir.clicked.connect(self._ie_pick_dst_dir)
        bg_lay.addWidget(btn_dst_dir, 2, 0)
        self.ie_dst_label = QLabel("未选择")
        self.ie_dst_label.setStyleSheet("color: gray;")
        self.ie_dst_label.setWordWrap(True)
        bg_lay.addWidget(self.ie_dst_label, 3, 0)
        src_layout.addWidget(self.ie_batch_widget, 1, 0, 1, 2)
        self.ie_batch_widget.hide()

        left_layout.addWidget(src_group)

        # 处理步骤列表（拖拽排序 + checkbox）
        pipeline_group = QGroupBox("处理步骤（拖拽排序，勾选启用）")
        pg_lay = QVBoxLayout(pipeline_group)

        self.ie_step_list = QListWidget()
        self.ie_step_list.setDragDropMode(QAbstractItemView.InternalMove)
        self.ie_step_list.setDefaultDropAction(Qt.MoveAction)
        self.ie_step_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.ie_step_list.setAlternatingRowColors(True)
        self.ie_step_list.setMaximumHeight(self._sp(180))
        self.ie_step_list.currentRowChanged.connect(self._ie_on_step_selected)
        self.ie_step_list.model().rowsMoved.connect(self._ie_on_step_reordered)
        self.ie_step_list.itemChanged.connect(lambda _: self._ie_request_preview())

        # 初始化5个步骤（全部勾选）
        for step_key in _IE.ALL_STEPS:
            item = QListWidgetItem(f"  {_IE.STEP_LABELS[step_key]}")
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsDragEnabled)
            item.setCheckState(Qt.Checked)
            item.setData(Qt.UserRole, step_key)   # Store step key
            self.ie_step_list.addItem(item)

        pg_lay.addWidget(self.ie_step_list)
        left_layout.addWidget(pipeline_group)

        # Parameter panel
        module_group = QGroupBox("🧩 模块参数")
        module_layout = QVBoxLayout(module_group)
        module_hint = QLabel("上方流程列表用于勾选启用和拖拽排序；下方页签用于调整各模块参数。")
        module_hint.setWordWrap(True)
        module_hint.setStyleSheet("color: #666;")
        module_layout.addWidget(module_hint)

        self.ie_param_tabs = QTabWidget()
        self.ie_param_tabs.currentChanged.connect(self._ie_on_param_tab_changed)
        self._build_crop_param_page()       # index 0
        self._build_gray_param_page()       # index 1
        self._build_mirror_param_page()     # index 2
        self._build_rotate_param_page()     # index 3
        self._build_bit_depth_param_page()  # index 4
        self._build_gray_math_param_page()  # index 5
        self._build_bc_param_page()         # index 6
        self._build_arith_param_page()      # index 7
        self._build_threshold_param_page()  # index 8
        module_layout.addWidget(self.ie_param_tabs)
        left_layout.addWidget(module_group)
        self.ie_step_list.setCurrentRow(0)

        # Single-image preview
        action_group = QGroupBox("执行")
        ag2_lay = QVBoxLayout(action_group)

        btn_save_single = QPushButton("💾 保存处理结果（单张）")
        btn_save_single.setStyleSheet(
            f"QPushButton {{ background-color: #388E3C; color: white; "
            f"padding: {self._sp(6)}px; border-radius: 5px; font-size: {self._sp(13)}px; }}"
            f"QPushButton:hover {{ background-color: #2E7D32; }}"
        )
        btn_save_single.clicked.connect(self._ie_save_single)
        ag2_lay.addWidget(btn_save_single)

        self.ie_batch_run_btn = QPushButton("批量处理所有图像")
        self.ie_batch_run_btn.setStyleSheet(
            f"QPushButton {{ background-color: #E64A19; color: white; "
            f"padding: {self._sp(6)}px; border-radius: 5px; font-size: {self._sp(13)}px; }}"
            f"QPushButton:hover {{ background-color: #BF360C; }}"
            f"QPushButton:disabled {{ background-color: #888; color: #ccc; }}"
        )
        self.ie_batch_run_btn.clicked.connect(self._ie_run_batch)
        ag2_lay.addWidget(self.ie_batch_run_btn)

        self.ie_stop_btn = QPushButton("停止批量")
        self.ie_stop_btn.setEnabled(False)
        self.ie_stop_btn.clicked.connect(self._ie_stop_batch)
        ag2_lay.addWidget(self.ie_stop_btn)

        self.ie_progress_bar = QProgressBar()
        self.ie_progress_bar.setValue(0)
        self.ie_progress_bar.setVisible(False)
        ag2_lay.addWidget(self.ie_progress_bar)

        left_layout.addWidget(action_group)
        left_layout.addStretch()

        left_scroll.setWidget(left_panel)

        # Original and processed preview
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(self._sp(4), self._sp(4), self._sp(4), self._sp(4))

        preview_splitter = QSplitter(Qt.Vertical)

        # 原图 / 处理后对比
        img_compare = QWidget()
        compare_layout = QHBoxLayout(img_compare)
        compare_layout.setContentsMargins(0, 0, 0, 0)

        orig_wrap = QGroupBox("原图")
        orig_lay = QVBoxLayout(orig_wrap)
        self.ie_orig_label = QLabel("（未加载）")
        self.ie_orig_label.setAlignment(Qt.AlignCenter)
        self.ie_orig_label.setMinimumSize(self._sp(300), self._sp(240))
        self.ie_orig_label.setStyleSheet("border: 1px solid #ccc; background: #f8f8f8;")
        orig_lay.addWidget(self.ie_orig_label)
        compare_layout.addWidget(orig_wrap)

        result_wrap = QGroupBox("处理结果（实时预览）")
        result_lay = QVBoxLayout(result_wrap)
        self.ie_preview_label = QLabel("（未处理）")
        self.ie_preview_label.setAlignment(Qt.AlignCenter)
        self.ie_preview_label.setMinimumSize(self._sp(300), self._sp(240))
        self.ie_preview_label.setStyleSheet("border: 1px solid #ccc; background: #f8f8f8;")
        result_lay.addWidget(self.ie_preview_label)
        compare_layout.addWidget(result_wrap)

        preview_splitter.addWidget(img_compare)

        # 日志
        self.ie_log = QTextEdit()
        self.ie_log.setReadOnly(True)
        self.ie_log.setPlaceholderText("操作日志将在此显示...")
        self.ie_log.setMaximumHeight(self._sp(150))
        preview_splitter.addWidget(self.ie_log)
        preview_splitter.setSizes([400, 150])

        right_layout.addWidget(preview_splitter)

        body_splitter.addWidget(left_scroll)
        body_splitter.addWidget(right_panel)
        body_splitter.setSizes([self._sp(380), self._sp(700)])

        outer.addWidget(body_splitter, stretch=1)
        self.content_stack.addWidget(page)

        # 实时预览定时器（防抖）
        self._ie_preview_timer = QTimer(self)
        self._ie_preview_timer.setSingleShot(True)
        self._ie_preview_timer.setInterval(300)  # 300ms 防抖
        self._ie_preview_timer.timeout.connect(self._ie_do_preview)

    # Parameter page builders

    def _build_crop_param_page(self):
        """裁剪参数。"""
        widget = QWidget()
        lay = QGridLayout(widget)
        lay.setContentsMargins(self._sp(4), self._sp(4), self._sp(4), self._sp(4))

        labels = ["X (px):", "Y (px):", "宽 (px):", "高 (px):"]
        self.ie_crop_spins = []
        for i, lbl in enumerate(labels):
            lay.addWidget(QLabel(lbl), i, 0)
            sp = QSpinBox()
            sp.setRange(0, 99999)
            sp.valueChanged.connect(lambda _: self._ie_request_preview())
            lay.addWidget(sp, i, 1)
            self.ie_crop_spins.append(sp)

        self.ie_param_tabs.addTab(widget, "裁剪 (ROI)")

    def _build_gray_param_page(self):
        """灰度参数。"""
        widget = QWidget()
        lay = QVBoxLayout(widget)
        lay.setContentsMargins(self._sp(4), self._sp(4), self._sp(4), self._sp(4))
        lay.addWidget(QLabel("将图像转换为 8-bit 灰度图（无可调参数）"))
        lay.addStretch()
        self.ie_param_tabs.addTab(widget, "灰度转换")

    def _build_mirror_param_page(self):
        widget = QWidget()
        lay = QGridLayout(widget)
        lay.setContentsMargins(self._sp(4), self._sp(4), self._sp(4), self._sp(4))
        lay.addWidget(QLabel("镜像方向:"), 0, 0)
        self.ie_mirror_combo = QComboBox()
        self.ie_mirror_combo.addItems(["水平镜像", "垂直镜像", "水平+垂直"])
        self.ie_mirror_combo.currentIndexChanged.connect(
            lambda _: self._ie_request_preview())
        lay.addWidget(self.ie_mirror_combo, 0, 1)
        self.ie_param_tabs.addTab(widget, "图像镜像")

    def _build_rotate_param_page(self):
        widget = QWidget()
        lay = QGridLayout(widget)
        lay.setContentsMargins(self._sp(4), self._sp(4), self._sp(4), self._sp(4))
        lay.addWidget(QLabel("旋转方式:"), 0, 0)
        self.ie_rotate_mode_combo = QComboBox()
        self.ie_rotate_mode_combo.addItems(["顺时针 90°", "逆时针 90°", "180°", "自定义角度"])
        self.ie_rotate_mode_combo.currentIndexChanged.connect(self._ie_on_rotate_mode_changed)
        lay.addWidget(self.ie_rotate_mode_combo, 0, 1)
        lay.addWidget(QLabel("角度:"), 1, 0)
        self.ie_rotate_angle_spin = QDoubleSpinBox()
        self.ie_rotate_angle_spin.setRange(-360.0, 360.0)
        self.ie_rotate_angle_spin.setSingleStep(1.0)
        self.ie_rotate_angle_spin.setEnabled(False)
        self.ie_rotate_angle_spin.valueChanged.connect(
            lambda _: self._ie_request_preview())
        lay.addWidget(self.ie_rotate_angle_spin, 1, 1)
        self.ie_rotate_expand_check = QCheckBox("自动扩展画布")
        self.ie_rotate_expand_check.setChecked(True)
        self.ie_rotate_expand_check.stateChanged.connect(
            lambda _: self._ie_request_preview())
        lay.addWidget(self.ie_rotate_expand_check, 2, 0, 1, 2)
        lay.addWidget(QLabel("边界灰度:"), 3, 0)
        self.ie_rotate_border_spin = QSpinBox()
        self.ie_rotate_border_spin.setRange(0, 255)
        self.ie_rotate_border_spin.valueChanged.connect(
            lambda _: self._ie_request_preview())
        lay.addWidget(self.ie_rotate_border_spin, 3, 1)
        self.ie_param_tabs.addTab(widget, "图像旋转")

    def _build_bit_depth_param_page(self):
        widget = QWidget()
        lay = QGridLayout(widget)
        lay.setContentsMargins(self._sp(4), self._sp(4), self._sp(4), self._sp(4))
        lay.addWidget(QLabel("源位深:"), 0, 0)
        self.ie_bit_depth_combo = QComboBox()
        self.ie_bit_depth_combo.addItems(["自动识别", "24 位", "16 位", "12 位"])
        self.ie_bit_depth_combo.currentIndexChanged.connect(
            lambda _: self._ie_request_preview())
        lay.addWidget(self.ie_bit_depth_combo, 0, 1)
        self.ie_param_tabs.addTab(widget, "转 8 位图")

    def _build_gray_math_param_page(self):
        widget = QWidget()
        lay = QGridLayout(widget)
        lay.setContentsMargins(self._sp(4), self._sp(4), self._sp(4), self._sp(4))
        lay.addWidget(QLabel("计算方式:"), 0, 0)
        self.ie_gray_math_combo = QComboBox()
        self.ie_gray_math_combo.addItems(["平均", "log", "exp", "sqrt", "sqr"])
        self.ie_gray_math_combo.currentIndexChanged.connect(self._ie_on_gray_math_changed)
        lay.addWidget(self.ie_gray_math_combo, 0, 1)
        lay.addWidget(QLabel("平均核大小:"), 1, 0)
        self.ie_gray_math_kernel_spin = QSpinBox()
        self.ie_gray_math_kernel_spin.setRange(1, 99)
        self.ie_gray_math_kernel_spin.setSingleStep(2)
        self.ie_gray_math_kernel_spin.setValue(3)
        self.ie_gray_math_kernel_spin.valueChanged.connect(
            lambda _: self._ie_request_preview())
        lay.addWidget(self.ie_gray_math_kernel_spin, 1, 1)
        self.ie_param_tabs.addTab(widget, "灰度值计算")

    def _build_bc_param_page(self):
        """亮度/对比度参数页。"""
        widget = QWidget()
        lay = QGridLayout(widget)
        lay.setContentsMargins(self._sp(4), self._sp(4), self._sp(4), self._sp(4))

        lay.addWidget(QLabel("对比度 α:"), 0, 0)
        self.ie_alpha_spin = QDoubleSpinBox()
        self.ie_alpha_spin.setRange(0.1, 5.0)
        self.ie_alpha_spin.setSingleStep(0.1)
        self.ie_alpha_spin.setValue(1.0)
        self.ie_alpha_spin.valueChanged.connect(
            lambda _: self._ie_request_preview())
        lay.addWidget(self.ie_alpha_spin, 0, 1)

        lay.addWidget(QLabel("亮度 β:"), 1, 0)
        self.ie_beta_spin = QSpinBox()
        self.ie_beta_spin.setRange(-255, 255)
        self.ie_beta_spin.setValue(0)
        self.ie_beta_spin.valueChanged.connect(
            lambda _: self._ie_request_preview())
        lay.addWidget(self.ie_beta_spin, 1, 1)

        lay.addWidget(QLabel("α 滑块:"), 2, 0)
        self.ie_alpha_slider = QSlider(Qt.Horizontal)
        self.ie_alpha_slider.setRange(1, 50)
        self.ie_alpha_slider.setValue(10)
        self.ie_alpha_slider.valueChanged.connect(self._ie_alpha_slider_moved)
        lay.addWidget(self.ie_alpha_slider, 2, 1)

        lay.addWidget(QLabel("β 滑块:"), 3, 0)
        self.ie_beta_slider = QSlider(Qt.Horizontal)
        self.ie_beta_slider.setRange(-255, 255)
        self.ie_beta_slider.setValue(0)
        self.ie_beta_slider.valueChanged.connect(self._ie_beta_slider_moved)
        lay.addWidget(self.ie_beta_slider, 3, 1)

        self.ie_param_tabs.addTab(widget, "亮度/对比度")

    def _build_arith_param_page(self):
        """加减法参数页"""
        widget = QWidget()
        lay = QGridLayout(widget)
        lay.setContentsMargins(self._sp(4), self._sp(4), self._sp(4), self._sp(4))

        lay.addWidget(QLabel("操作:"), 0, 0)
        self.ie_arith_combo = QComboBox()
        self.ie_arith_combo.addItems(["加法 (add)", "减法 (subtract)"])
        self.ie_arith_combo.currentIndexChanged.connect(
            lambda _: self._ie_request_preview())
        lay.addWidget(self.ie_arith_combo, 0, 1)

        lay.addWidget(QLabel("操作数来源:"), 1, 0)
        self.ie_operand_src_combo = QComboBox()
        self.ie_operand_src_combo.addItems(["使用标量值", "使用第二张图像"])
        self.ie_operand_src_combo.currentIndexChanged.connect(self._ie_on_operand_src_changed)
        lay.addWidget(self.ie_operand_src_combo, 1, 1)

        lay.addWidget(QLabel("标量值:"), 2, 0)
        self.ie_scalar_spin = QSpinBox()
        self.ie_scalar_spin.setRange(0, 255)
        self.ie_scalar_spin.setValue(0)
        self.ie_scalar_spin.valueChanged.connect(
            lambda _: self._ie_request_preview())
        lay.addWidget(self.ie_scalar_spin, 2, 1)

        # 第二张图选择
        self.ie_operand_widget = QWidget()
        ow_lay = QVBoxLayout(self.ie_operand_widget)
        ow_lay.setContentsMargins(0, 0, 0, 0)
        btn_pick_operand = QPushButton("选择第二张图...")
        btn_pick_operand.clicked.connect(self._ie_pick_operand)
        ow_lay.addWidget(btn_pick_operand)
        self.ie_operand_label = QLabel("未选择")
        self.ie_operand_label.setStyleSheet("color: gray;")
        self.ie_operand_label.setWordWrap(True)
        ow_lay.addWidget(self.ie_operand_label)
        lay.addWidget(self.ie_operand_widget, 3, 0, 1, 2)
        self.ie_operand_widget.hide()

        self.ie_param_tabs.addTab(widget, "图像加/减法")

    def _build_threshold_param_page(self):
        """阈值化参数。"""
        widget = QWidget()
        lay = QGridLayout(widget)
        lay.setContentsMargins(self._sp(4), self._sp(4), self._sp(4), self._sp(4))

        lay.addWidget(QLabel("模式:"), 0, 0)
        self.ie_thr_mode_combo = QComboBox()
        self.ie_thr_mode_combo.addItems([
            "全局阈值 (Global)",
            "大津法 (Otsu)",
            "自适应均值 (Adaptive Mean)",
            "自适应高斯 (Adaptive Gaussian)"
        ])
        self.ie_thr_mode_combo.currentIndexChanged.connect(self._ie_on_thr_mode_changed)
        lay.addWidget(self.ie_thr_mode_combo, 0, 1)

        lay.addWidget(QLabel("阈值:"), 1, 0)
        self.ie_thr_val_spin = QSpinBox()
        self.ie_thr_val_spin.setRange(0, 255)
        self.ie_thr_val_spin.setValue(128)
        self.ie_thr_val_spin.valueChanged.connect(
            lambda _: self._ie_request_preview())
        lay.addWidget(self.ie_thr_val_spin, 1, 1)

        self.ie_thr_val_slider = QSlider(Qt.Horizontal)
        self.ie_thr_val_slider.setRange(0, 255)
        self.ie_thr_val_slider.setValue(128)
        self.ie_thr_val_slider.valueChanged.connect(self._ie_thr_slider_moved)
        lay.addWidget(self.ie_thr_val_slider, 2, 0, 1, 2)

        lay.addWidget(QLabel("最大值:"), 3, 0)
        self.ie_thr_max_spin = QSpinBox()
        self.ie_thr_max_spin.setRange(1, 255)
        self.ie_thr_max_spin.setValue(255)
        self.ie_thr_max_spin.valueChanged.connect(
            lambda _: self._ie_request_preview())
        lay.addWidget(self.ie_thr_max_spin, 3, 1)

        # Adaptive threshold parameters
        self.ie_adapt_widget = QWidget()
        aw_lay = QGridLayout(self.ie_adapt_widget)
        aw_lay.setContentsMargins(0, 0, 0, 0)
        aw_lay.addWidget(QLabel("块大小(奇数):"), 0, 0)
        self.ie_thr_block_spin = QSpinBox()
        self.ie_thr_block_spin.setRange(3, 199)
        self.ie_thr_block_spin.setSingleStep(2)
        self.ie_thr_block_spin.setValue(11)
        self.ie_thr_block_spin.valueChanged.connect(
            lambda _: self._ie_request_preview())
        aw_lay.addWidget(self.ie_thr_block_spin, 0, 1)
        aw_lay.addWidget(QLabel("偏移 C:"), 1, 0)
        self.ie_thr_c_spin = QSpinBox()
        self.ie_thr_c_spin.setRange(-50, 50)
        self.ie_thr_c_spin.setValue(2)
        self.ie_thr_c_spin.valueChanged.connect(
            lambda _: self._ie_request_preview())
        aw_lay.addWidget(self.ie_thr_c_spin, 1, 1)
        lay.addWidget(self.ie_adapt_widget, 4, 0, 1, 2)
        self.ie_adapt_widget.hide()

        self.ie_param_tabs.addTab(widget, "闃堝€煎寲")

    # ------------------------------------------------------------
    # Image editor slots
    # ------------------------------------------------------------

    def _ie_on_mode_changed(self, idx):
        """切换单张/批量模式"""
        single = (idx == 0)
        self.ie_single_widget.setVisible(single)
        self.ie_batch_widget.setVisible(not single)

    def _ie_on_step_selected(self, row):
        """列表选中项变化：切换参数页。"""
        if row < 0:
            return
        step_key = self.ie_step_list.item(row).data(Qt.UserRole)
        idx_map = {
            "crop": 0, "gray": 1, "mirror": 2, "rotate": 3,
            "bit_depth": 4, "gray_math": 5, "bc": 6,
            "arithmetic": 7, "threshold": 8
        }
        tab_idx = idx_map.get(step_key, 0)
        if self.ie_param_tabs.currentIndex() != tab_idx:
            self.ie_param_tabs.setCurrentIndex(tab_idx)

    def _ie_on_param_tab_changed(self, idx):
        """参数页切换时，同步高亮对应的步骤。"""
        if idx < 0:
            return
        step_keys = [
            "crop", "gray", "mirror", "rotate", "bit_depth", "gray_math",
            "bc", "arithmetic", "threshold"
        ]
        target_key = step_keys[idx]
        for row in range(self.ie_step_list.count()):
            item = self.ie_step_list.item(row)
            if item.data(Qt.UserRole) == target_key:
                if self.ie_step_list.currentRow() != row:
                    self.ie_step_list.blockSignals(True)
                    self.ie_step_list.setCurrentRow(row)
                    self.ie_step_list.blockSignals(False)
                break

    def _ie_on_step_reordered(self):
        """拖拽排序后触发预览。"""
        self._ie_request_preview()

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
        """在原图区域显示图像。"""
        import cv2 as _cv2
        img = _cv2.imread(path, _cv2.IMREAD_UNCHANGED)
        if img is None:
            return
        h, w = img.shape[:2]
        self.ie_log.append(f"已加载 {os.path.basename(path)}  ({w}x{h})")
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
            self.ie_src_label.setStyleSheet("")
            from utils.image_editor import SUPPORTED_EXTS
            cnt = sum(1 for f in Path(d).iterdir()
                      if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS)
            self.ie_log.append(f"输入目录: {d}  (共 {cnt} 张图像)")

    def _ie_pick_dst_dir(self):
        d = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if not d:
            d = QFileDialog.getSaveFileName(self, "创建输出目录",
                                             "", "目录")[0]
        if d:
            self.ie_dst_dir = d
            self.ie_dst_label.setText(d)
            self.ie_dst_label.setStyleSheet("")
            self.ie_log.append(f"输出目录: {d}")

    def _ie_pick_operand(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择第二张图", "",
            "图像文件 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)")
        if path:
            self.ie_operand_path = path
            self.ie_operand_label.setText(os.path.basename(path))
            self._ie_request_preview()

    def _ie_get_step_order(self):
        """从 QListWidget 读取已勾选的步骤 key 列表（保持列表顺序）。"""
        order = []
        for i in range(self.ie_step_list.count()):
            item = self.ie_step_list.item(i)
            if item.checkState() == Qt.Checked:
                order.append(item.data(Qt.UserRole))
        return order


    def _ie_sync_config_from_ui(self):
        """从参数面板控件直接同步到 self.ie_config。"""
        cfg = self.ie_config

        cfg.crop.enabled = True   # 由 step_order 控制是否执行
        spins = self.ie_crop_spins
        cfg.crop.x, cfg.crop.y = spins[0].value(), spins[1].value()
        cfg.crop.w, cfg.crop.h = spins[2].value(), spins[3].value()

        cfg.gray.enabled = True
        # 灰度无可调参数

        cfg.mirror.enabled = True
        cfg.mirror.mode = ["horizontal", "vertical", "both"][self.ie_mirror_combo.currentIndex()]

        cfg.rotate.enabled = True
        rotate_modes = ["cw90", "ccw90", "180", "custom"]
        cfg.rotate.mode = rotate_modes[self.ie_rotate_mode_combo.currentIndex()]
        cfg.rotate.angle = self.ie_rotate_angle_spin.value()
        cfg.rotate.expand = self.ie_rotate_expand_check.isChecked()
        cfg.rotate.border_value = self.ie_rotate_border_spin.value()

        cfg.bit_depth.enabled = True
        bit_depth_values = [0, 24, 16, 12]
        cfg.bit_depth.source_bits = bit_depth_values[self.ie_bit_depth_combo.currentIndex()]

        cfg.gray_math.enabled = True
        gray_math_ops = ["average", "log", "exp", "sqrt", "sqr"]
        cfg.gray_math.operation = gray_math_ops[self.ie_gray_math_combo.currentIndex()]
        cfg.gray_math.kernel_size = self.ie_gray_math_kernel_spin.value()

        cfg.bc.enabled = True
        cfg.bc.alpha = self.ie_alpha_spin.value()
        cfg.bc.beta  = self.ie_beta_spin.value()

        cfg.arithmetic.enabled = True
        cfg.arithmetic.operation = ("add" if self.ie_arith_combo.currentIndex() == 0
                                    else "subtract")
        cfg.arithmetic.scalar_value = self.ie_scalar_spin.value()
        cfg.arithmetic.operand_path = (self.ie_operand_path
                                        if self.ie_operand_src_combo.currentIndex() == 1
                                        else "")

        cfg.threshold.enabled = True
        mode_map = ["global", "otsu", "adaptive_mean", "adaptive_gaussian"]
        cfg.threshold.mode = mode_map[self.ie_thr_mode_combo.currentIndex()]
        cfg.threshold.threshold_value = self.ie_thr_val_spin.value()
        cfg.threshold.max_value = self.ie_thr_max_spin.value()
        cfg.threshold.block_size = self.ie_thr_block_spin.value()
        cfg.threshold.C = self.ie_thr_c_spin.value()

    def _ie_do_preview(self):
        """实时预览（由 QTimer 触发）。"""
        import cv2 as _cv2
        if not self.ie_single_path or not os.path.isfile(self.ie_single_path):
            return

        step_order = self._ie_get_step_order()
        if not step_order:
            # 未勾选任何步骤，显示原图
            # No steps selected; show original image
            self._ie_current_preview = None
            return

        self._ie_sync_config_from_ui()
        try:
            img = _cv2.imread(self.ie_single_path, _cv2.IMREAD_UNCHANGED)
            op_img = None
            if self.ie_config.arithmetic.operand_path:
                op_img = _cv2.imread(self.ie_config.arithmetic.operand_path,
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
            steps_str = " -> ".join(
                ImageEditor.STEP_LABELS.get(s, s) for s in step_order)
            self.ie_log.append(f"预览: {steps_str}  输出: {w}x{h}")
        except Exception as e:
            self.ie_log.append(f"预览失败: {e}")

    def _ie_save_single(self):
        """保存单张处理结果"""
        import cv2 as _cv2
        if self._ie_current_preview is None:
            QMessageBox.warning(self, "提示", "请先加载图像并调整参数（预览生成后再保存）")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "保存处理结果", "",
            "PNG (*.png);;JPEG (*.jpg);;BMP (*.bmp);;TIFF (*.tif)")
        if path:
            _cv2.imwrite(path, self._ie_current_preview)
            self.ie_log.append(f"已保存 {path}")
            QMessageBox.information(self, "完成", f"已保存至:\n{path}")

    def _ie_run_batch(self):
        """启动批量处理。"""
        if not self.ie_src_dir or not os.path.isdir(self.ie_src_dir):
            QMessageBox.warning(self, "提示", "请先选择输入目录")
            return
        if not self.ie_dst_dir:
            QMessageBox.warning(self, "提示", "请先选择输出目录")
            return
        os.makedirs(self.ie_dst_dir, exist_ok=True)

        step_order = self._ie_get_step_order()
        if not step_order:
            QMessageBox.warning(self, "提示", "请至少勾选一个处理步骤")
            return

        self._ie_sync_config_from_ui()

        self.ie_batch_run_btn.setEnabled(False)
        self.ie_stop_btn.setEnabled(True)
        self.ie_progress_bar.setValue(0)
        self.ie_progress_bar.setVisible(True)
        self.ie_log.append(f"开始批量处理: {self.ie_src_dir}")
        self.ie_log.append(f"   步骤: {' -> '.join(ImageEditor.STEP_LABELS.get(s, s) for s in step_order)}")

        self.ie_worker_thread = self._IEBatchWorker(
            self.ie_src_dir, self.ie_dst_dir, self.ie_config, step_order)
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
        """numpy 数组转 QPixmap。"""
        import cv2 as _cv2
        if img is None:
            return None
        if img.dtype != np.uint8:
            img = np.clip(img.astype(np.float64), 0, 255).astype(np.uint8)
        if img.ndim == 2:
            img_bgr = _cv2.cvtColor(img, _cv2.COLOR_GRAY2BGR)
        else:
            img_bgr = img
        img_rgb = _cv2.cvtColor(img_bgr, _cv2.COLOR_BGR2RGB)
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
            "bubble_segment": 4,
            "particle": 5,
            "ptv": 6,
            "piv2d": 7,
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
            5: 6,
            6: 0,
        }
        current_idx = legacy_to_current.get(page_idx, page_idx)
        self.content_stack.setCurrentIndex(current_idx)
        nav_keys = [
            "image_editor",
            "calibration",
            "reconstruction",
            "raytrace",
            "bubble_segment",
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

    class _IEBatchWorker(QThread):
        progress = pyqtSignal(int, int, str)   # done, total, filename
        finished = pyqtSignal(int, int)         # success, total
        error    = pyqtSignal(str)

        def __init__(self, src_dir, dst_dir, config: "ImageEditConfig",
                     step_order=None):
            super().__init__()
            self.src_dir = src_dir
            self.dst_dir = dst_dir
            self.config  = config
            self.step_order = step_order or []
            self._stop   = False

        def stop(self):
            self._stop = True

        def run(self):
            try:
                from utils.image_editor import ImageEditor, SUPPORTED_EXTS
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
                    op_img = _cv2.imread(self.config.arithmetic.operand_path,
                                         _cv2.IMREAD_UNCHANGED)

                editor = ImageEditor(self.config)
                for i, f in enumerate(files):
                    if self._stop:
                        break
                    dst = str(Path(self.dst_dir) / f.name)
                    img = _cv2.imread(str(f), _cv2.IMREAD_UNCHANGED)
                    if img is not None:
                        result = editor.process(img, op_img,
                                                step_order=self.step_order)
                        _cv2.imwrite(dst, result)
                        success += 1
                    self.progress.emit(i + 1, total, f.name)

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
