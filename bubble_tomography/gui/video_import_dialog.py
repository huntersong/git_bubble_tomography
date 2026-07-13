"""Video import dialog for the general image-processing module."""

import json
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from utils.video_importer import (
    VIDEO_EXTS,
    VideoAdjustmentConfig,
    convert_frame_bit_depth,
    export_video_frames,
    open_video_reader,
    probe_video,
    render_adjusted_display,
)


class VideoExportWorker(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(int, str)
    error = pyqtSignal(str)

    def __init__(
        self,
        video_path: str,
        output_dir: str,
        bit_depth: int,
        image_format: str,
        start_frame: int,
        end_frame: int,
        step: int,
        adjustment_config: Optional[VideoAdjustmentConfig] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.video_path = video_path
        self.output_dir = output_dir
        self.bit_depth = bit_depth
        self.image_format = image_format
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.step = step
        self.adjustment_config = adjustment_config

    def run(self):
        try:
            count = export_video_frames(
                self.video_path,
                self.output_dir,
                bit_depth=self.bit_depth,
                image_format=self.image_format,
                start=self.start_frame,
                end=self.end_frame,
                step=self.step,
                adjustment_config=self.adjustment_config,
                progress=lambda pct, idx, dst: self.progress.emit(
                    pct, f"导出帧 {idx}: {Path(dst).name}"
                ),
            )
            self.finished.emit(count, self.output_dir)
        except Exception as exc:
            self.error.emit(str(exc))


class VideoImportWidget(QWidget):
    """CINE/MP4/MOV/AVI/MKV video import and frame export UI."""

    export_finished = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("videoImportPage")
        self.setWindowTitle("视频导入")
        self.resize(1260, 760)
        self._items: List[dict] = []
        self._reader = None
        self._worker: Optional[VideoExportWorker] = None
        self._current_frame = 0
        self._play_direction = 1
        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._advance_playback)
        self.last_output_dir = ""
        self._build_ui()
        self._sync_enabled(False)

    def _build_ui(self):
        self.setStyleSheet("""
            QWidget#videoImportPage { background: #11161D; color: #E6EDF7; }
            QLabel { color: #E6EDF7; }
            QGroupBox {
                color: #E6EDF7; font-weight: 700;
                border: 1px solid #2C3440; border-radius: 8px;
                margin-top: 12px; padding-top: 16px;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
            QPushButton {
                background: #252E39; color: #E6EDF7;
                border: 1px solid #334255; border-radius: 7px;
                padding: 7px 12px; font-weight: 600;
            }
            QPushButton:hover { background: #2D68E8; border-color: #4F86F5; }
            QPushButton:disabled { color: #6D7683; background: #1A2029; }
            QLineEdit, QComboBox, QSpinBox {
                background: #202833; color: #E6EDF7;
                border: 1px solid #303B49; border-radius: 6px; padding: 5px 8px;
            }
            QListWidget, QTextEdit {
                background: #171D25; color: #DCE7FF;
                border: 1px solid #2C3440; border-radius: 8px;
            }
            QListWidget::item { padding: 8px; border-radius: 6px; }
            QListWidget::item:selected { background: #2D68E8; color: white; }
            QSlider::groove:horizontal { background: #344052; height: 5px; border-radius: 2px; }
            QSlider::handle:horizontal { background: #4F86F5; width: 14px; margin: -5px 0; border-radius: 7px; }
            QProgressBar { background: #202833; border: none; border-radius: 5px; height: 14px; color: white; text-align: center; }
            QProgressBar::chunk { background: #2EB867; border-radius: 5px; }
        """)
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 16, 20, 16)
        root.setSpacing(14)

        title_row = QHBoxLayout()
        title = QLabel("视频导入")
        title.setStyleSheet("font-size: 24px; font-weight: 800; color: #FFFFFF;")
        title_row.addWidget(title)
        subtitle = QLabel("媒体 / 导入")
        subtitle.setStyleSheet("font-size: 13px; color: #8D98A8;")
        title_row.addWidget(subtitle)
        title_row.addStretch()
        badge = QLabel("CINE · MP4 · MOV · AVI · MKV")
        badge.setStyleSheet(
            "background:#202833; color:#CFE0FF; border-radius:7px; padding:6px 12px;"
        )
        title_row.addWidget(badge)
        root.addLayout(title_row)

        body = QHBoxLayout()
        body.setSpacing(14)
        root.addLayout(body, stretch=1)

        left = QVBoxLayout()
        body.addLayout(left, stretch=0)
        source_group = QGroupBox("来源")
        source_layout = QVBoxLayout(source_group)
        add_btn = QPushButton("打开视频文件...")
        add_btn.clicked.connect(self._add_files)
        source_layout.addWidget(add_btn)
        add_dir_btn = QPushButton("打开文件夹...")
        add_dir_btn.clicked.connect(self._add_folder)
        source_layout.addWidget(add_dir_btn)
        left.addWidget(source_group)

        queue_group = QGroupBox("素材队列")
        queue_layout = QVBoxLayout(queue_group)
        self.queue_list = QListWidget()
        self.queue_list.setMinimumWidth(260)
        self.queue_list.currentRowChanged.connect(self._on_selected_video_changed)
        queue_layout.addWidget(self.queue_list, stretch=1)
        left.addWidget(queue_group, stretch=1)
        hint = QLabel("支持 CINE RAW、MP4、MOV、AVI、MKV、MXF 等视频。")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#8D98A8; font-size:12px;")
        left.addWidget(hint)

        center = QVBoxLayout()
        body.addLayout(center, stretch=1)
        self.video_name_label = QLabel("尚未选择视频")
        self.video_name_label.setStyleSheet("font-size: 15px; font-weight: 700;")
        center.addWidget(self.video_name_label)
        self.preview_label = QLabel("视频预览")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(620, 390)
        self.preview_label.setStyleSheet(
            "background:#0B1118; border:1px solid #2C3440; border-radius:8px; color:#6D7683;"
        )
        center.addWidget(self.preview_label, stretch=1)

        controls = QHBoxLayout()
        self.prev_btn = QPushButton("◀")
        self.prev_btn.clicked.connect(lambda: self._seek_relative(-1))
        controls.addWidget(self.prev_btn)
        self.reverse_play_btn = QPushButton("倒退播放")
        self.reverse_play_btn.clicked.connect(lambda: self._start_playback(-1))
        controls.addWidget(self.reverse_play_btn)
        self.play_btn = QPushButton("播放")
        self.play_btn.clicked.connect(lambda: self._toggle_playback(1))
        controls.addWidget(self.play_btn)
        self.stop_btn = QPushButton("停止")
        self.stop_btn.clicked.connect(self._stop_playback)
        controls.addWidget(self.stop_btn)
        self.next_btn = QPushButton("▶")
        self.next_btn.clicked.connect(lambda: self._seek_relative(1))
        controls.addWidget(self.next_btn)
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.valueChanged.connect(self._on_frame_slider_changed)
        controls.addWidget(self.frame_slider, stretch=1)
        self.frame_label = QLabel("帧 -- / --")
        self.frame_label.setMinimumWidth(130)
        controls.addWidget(self.frame_label)
        center.addLayout(controls)

        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setMinimumWidth(330)
        right_scroll.setStyleSheet("QScrollArea { border: none; background: #11161D; }")
        right_widget = QWidget()
        right = QVBoxLayout(right_widget)
        right.setContentsMargins(0, 0, 0, 0)
        right.setSpacing(10)
        right_scroll.setWidget(right_widget)
        body.addWidget(right_scroll, stretch=0)
        info_group = QGroupBox("素材信息")
        info_layout = QVBoxLayout(info_group)
        self.info_label = QLabel("格式: --\n分辨率: --\n位深/帧率: --\n时长: --")
        self.info_label.setWordWrap(True)
        self.info_label.setMinimumWidth(270)
        self.info_label.setStyleSheet("color:#DCE7FF; font-size:12px;")
        info_layout.addWidget(self.info_label)
        right.addWidget(info_group)

        tools_group = QGroupBox("图像工具")
        tools_layout = QGridLayout(tools_group)
        self.hist_label = QLabel("直方图")
        self.hist_label.setAlignment(Qt.AlignCenter)
        self.hist_label.setMinimumSize(280, 150)
        self.hist_label.setStyleSheet("background:#F4F5F8; color:#333; border-radius:4px;")
        tools_layout.addWidget(self.hist_label, 0, 0, 1, 4)
        self.avg_label = QLabel("Avg: --")
        self.avg_label.setStyleSheet("color:#DCE7FF; font-weight:700;")
        tools_layout.addWidget(self.avg_label, 1, 0, 1, 2)
        tools_layout.addWidget(QLabel("显示"), 1, 2)
        self.color_combo = QComboBox()
        self.color_combo.addItems(["灰", "Jet", "Hot", "Turbo"])
        tools_layout.addWidget(self.color_combo, 1, 3)

        tools_layout.addWidget(QLabel("标准曲线"), 2, 0)
        self.curve_combo = QComboBox()
        self.curve_combo.addItems(["线性", "加马"])
        tools_layout.addWidget(self.curve_combo, 2, 1, 1, 3)
        tools_layout.addWidget(QLabel("Bit slider"), 3, 0, 1, 4)
        self.bit_min_spin = QSpinBox()
        self.bit_max_spin = QSpinBox()
        self.bit_min_slider = QSlider(Qt.Horizontal)
        self.bit_max_slider = QSlider(Qt.Horizontal)
        for slider in (self.bit_min_slider, self.bit_max_slider):
            slider.setRange(0, 4095)
        for spin in (self.bit_min_spin, self.bit_max_spin):
            spin.setRange(0, 4095)
        self.bit_min_spin.setValue(0)
        self.bit_max_spin.setValue(4095)
        self.bit_max_slider.setValue(4095)
        tools_layout.addWidget(self.bit_min_spin, 4, 0)
        tools_layout.addWidget(self.bit_min_slider, 4, 1)
        tools_layout.addWidget(self.bit_max_slider, 4, 2)
        tools_layout.addWidget(self.bit_max_spin, 4, 3)

        self.brightness_spin = QDoubleSpinBox()
        self.gain_spin = QDoubleSpinBox()
        self.gamma_spin = QDoubleSpinBox()
        self.knee_spin = QDoubleSpinBox()
        self.brightness_slider = QSlider(Qt.Horizontal)
        self.gain_slider = QSlider(Qt.Horizontal)
        self.gamma_slider = QSlider(Qt.Horizontal)
        self.knee_slider = QSlider(Qt.Horizontal)
        self._add_adjust_row(tools_layout, 5, "亮度 (%)", self.brightness_spin, self.brightness_slider, -100, 100, 0.0, 100.0)
        self._add_adjust_row(tools_layout, 6, "增益", self.gain_spin, self.gain_slider, 0, 400, 1.0, 100.0)
        self._add_adjust_row(tools_layout, 7, "加马", self.gamma_spin, self.gamma_slider, 5, 500, 1.0, 100.0)
        self._add_adjust_row(tools_layout, 8, "膝部", self.knee_spin, self.knee_slider, 5, 500, 1.0, 100.0)

        tools_layout.addWidget(QLabel("滤波器"), 9, 0)
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["无", "中值 3x3", "高斯 3x3"])
        tools_layout.addWidget(self.filter_combo, 9, 1, 1, 3)
        self.flip_h_check = QCheckBox("水平翻转")
        self.flip_v_check = QCheckBox("垂直翻转")
        self.rotate_ccw_check = QCheckBox("逆时针")
        self.rotate_cw_check = QCheckBox("顺时针")
        tools_layout.addWidget(self.flip_h_check, 10, 0)
        tools_layout.addWidget(self.flip_v_check, 10, 1)
        tools_layout.addWidget(self.rotate_ccw_check, 10, 2)
        tools_layout.addWidget(self.rotate_cw_check, 10, 3)
        self.grid_check = QCheckBox("网格")
        self.cross_check = QCheckBox("十字")
        tools_layout.addWidget(QLabel("叠加显示:"), 11, 0)
        tools_layout.addWidget(self.grid_check, 11, 1)
        tools_layout.addWidget(self.cross_check, 11, 2)

        crop_group = QGroupBox("裁切和一次采样")
        crop_layout = QGridLayout(crop_group)
        self.crop_check = QCheckBox("裁切")
        self.hide_crop_check = QCheckBox("隐藏裁切框")
        self.hide_crop_check.setChecked(True)
        crop_layout.addWidget(self.crop_check, 0, 0)
        crop_layout.addWidget(self.hide_crop_check, 0, 1, 1, 3)
        self.crop_x_spin = QSpinBox()
        self.crop_y_spin = QSpinBox()
        self.crop_w_spin = QSpinBox()
        self.crop_h_spin = QSpinBox()
        for spin in (self.crop_x_spin, self.crop_y_spin, self.crop_w_spin, self.crop_h_spin):
            spin.setRange(0, 100000)
        for col, (label, spin) in enumerate((("X", self.crop_x_spin), ("Y", self.crop_y_spin), ("宽", self.crop_w_spin), ("高", self.crop_h_spin))):
            crop_layout.addWidget(QLabel(label), 1, col)
            crop_layout.addWidget(spin, 2, col)
        self.resample_check = QCheckBox("一次采样")
        crop_layout.addWidget(self.resample_check, 3, 0, 1, 2)
        self.resample_w_spin = QSpinBox()
        self.resample_h_spin = QSpinBox()
        for spin in (self.resample_w_spin, self.resample_h_spin):
            spin.setRange(1, 100000)
        crop_layout.addWidget(QLabel("宽"), 4, 0)
        crop_layout.addWidget(QLabel("高"), 4, 1)
        crop_layout.addWidget(self.resample_w_spin, 5, 0)
        crop_layout.addWidget(self.resample_h_spin, 5, 1)
        tools_layout.addWidget(crop_group, 12, 0, 1, 4)

        button_row = QHBoxLayout()
        self.adjust_disable_btn = QPushButton("禁用")
        self.adjust_load_btn = QPushButton("载入")
        self.adjust_save_btn = QPushButton("保存")
        self.adjust_default_btn = QPushButton("默认")
        for btn in (self.adjust_disable_btn, self.adjust_load_btn, self.adjust_save_btn, self.adjust_default_btn):
            button_row.addWidget(btn)
        tools_layout.addLayout(button_row, 13, 0, 1, 4)
        right.addWidget(tools_group)

        export_group = QGroupBox("视频转图片")
        export_layout = QGridLayout(export_group)
        export_layout.addWidget(QLabel("输出位深"), 0, 0)
        self.bit_depth_combo = QComboBox()
        self.bit_depth_combo.addItems(["8 位灰度", "12 位灰度", "24 位彩色"])
        export_layout.addWidget(self.bit_depth_combo, 0, 1)
        export_layout.addWidget(QLabel("图片格式"), 1, 0)
        self.format_combo = QComboBox()
        self.format_combo.addItems(["png", "tif", "bmp"])
        export_layout.addWidget(self.format_combo, 1, 1)
        export_layout.addWidget(QLabel("起始帧"), 2, 0)
        self.start_spin = QSpinBox()
        export_layout.addWidget(self.start_spin, 2, 1)
        export_layout.addWidget(QLabel("结束帧"), 3, 0)
        self.end_spin = QSpinBox()
        export_layout.addWidget(self.end_spin, 3, 1)
        export_layout.addWidget(QLabel("步长"), 4, 0)
        self.step_spin = QSpinBox()
        self.step_spin.setRange(1, 100000)
        self.step_spin.setValue(1)
        export_layout.addWidget(self.step_spin, 4, 1)
        export_layout.addWidget(QLabel("输出目录"), 5, 0)
        self.output_edit = QLineEdit()
        export_layout.addWidget(self.output_edit, 5, 1)
        pick_output = QPushButton("选择...")
        pick_output.clicked.connect(self._pick_output_dir)
        export_layout.addWidget(pick_output, 6, 1)
        self.export_btn = QPushButton("导出图片")
        self.export_btn.clicked.connect(self._export_selected)
        export_layout.addWidget(self.export_btn, 7, 0, 1, 2)
        self.progress = QProgressBar()
        self.progress.setValue(0)
        export_layout.addWidget(self.progress, 8, 0, 1, 2)
        right.addWidget(export_group)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(130)
        self.log.setPlaceholderText("导入和导出日志")
        right.addWidget(self.log)
        right.addStretch()
        self._wire_adjustment_controls()

    def _add_adjust_row(self, layout, row, label, spin, slider, minimum, maximum, value, scale):
        layout.addWidget(QLabel(label), row, 0)
        spin.setDecimals(3 if scale == 100.0 else 2)
        spin.setRange(float(minimum) / scale, float(maximum) / scale)
        spin.setSingleStep(1.0 / scale)
        spin.setValue(float(value))
        slider.setRange(int(minimum), int(maximum))
        slider.setValue(int(round(float(value) * scale)))
        layout.addWidget(spin, row, 1)
        layout.addWidget(slider, row, 2, 1, 2)

    def _wire_adjustment_controls(self):
        self._adjust_widgets = [
            self.hist_label, self.color_combo, self.curve_combo,
            self.bit_min_spin, self.bit_max_spin, self.bit_min_slider, self.bit_max_slider,
            self.brightness_spin, self.gain_spin, self.gamma_spin, self.knee_spin,
            self.brightness_slider, self.gain_slider, self.gamma_slider, self.knee_slider,
            self.filter_combo, self.flip_h_check, self.flip_v_check,
            self.rotate_ccw_check, self.rotate_cw_check, self.grid_check, self.cross_check,
            self.crop_check, self.hide_crop_check, self.crop_x_spin, self.crop_y_spin,
            self.crop_w_spin, self.crop_h_spin, self.resample_check,
            self.resample_w_spin, self.resample_h_spin, self.adjust_disable_btn,
            self.adjust_load_btn, self.adjust_save_btn, self.adjust_default_btn,
        ]
        self.bit_min_slider.valueChanged.connect(self.bit_min_spin.setValue)
        self.bit_max_slider.valueChanged.connect(self.bit_max_spin.setValue)
        self.bit_min_spin.valueChanged.connect(self.bit_min_slider.setValue)
        self.bit_max_spin.valueChanged.connect(self.bit_max_slider.setValue)
        for spin, slider, scale in (
            (self.brightness_spin, self.brightness_slider, 100.0),
            (self.gain_spin, self.gain_slider, 100.0),
            (self.gamma_spin, self.gamma_slider, 100.0),
            (self.knee_spin, self.knee_slider, 100.0),
        ):
            spin.valueChanged.connect(lambda value, s=slider, sc=scale: s.setValue(int(round(value * sc))))
            slider.valueChanged.connect(lambda value, sp=spin, sc=scale: sp.setValue(value / sc))
        for widget in self._adjust_widgets:
            if widget in (self.hist_label, self.adjust_load_btn, self.adjust_save_btn, self.adjust_default_btn, self.adjust_disable_btn):
                continue
            if hasattr(widget, "valueChanged"):
                widget.valueChanged.connect(self._on_adjustment_changed)
            elif hasattr(widget, "currentIndexChanged"):
                widget.currentIndexChanged.connect(self._on_adjustment_changed)
            elif hasattr(widget, "toggled"):
                widget.toggled.connect(self._on_adjustment_changed)
        self.adjust_default_btn.clicked.connect(self._reset_adjustments)
        self.adjust_disable_btn.clicked.connect(self._disable_adjustments)
        self.adjust_save_btn.clicked.connect(self._save_adjustments)
        self.adjust_load_btn.clicked.connect(self._load_adjustments)

    def _add_files(self):
        exts = " ".join(f"*{ext}" for ext in sorted(VIDEO_EXTS))
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择视频文件", "", f"视频文件 ({exts});;所有文件 (*.*)"
        )
        self._append_files(files)

    def _add_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择视频文件夹")
        if not folder:
            return
        paths = [
            str(p) for p in sorted(Path(folder).iterdir())
            if p.is_file() and p.suffix.lower() in VIDEO_EXTS
        ]
        self._append_files(paths)

    def _append_files(self, files: List[str]):
        for path in files:
            if not path or not Path(path).exists():
                continue
            try:
                info = probe_video(path)
            except Exception as exc:
                self.log.append(f"无法导入 {Path(path).name}: {exc}")
                continue
            self._items.append({"path": path, "info": info})
            self.queue_list.addItem(f"{Path(path).name}\n{info.summary()}")
            self.log.append(f"已导入: {info.summary()}")
        if self.queue_list.count() and self.queue_list.currentRow() < 0:
            self.queue_list.setCurrentRow(0)

    def _on_selected_video_changed(self, row: int):
        self._stop_playback()
        self._close_reader()
        if row < 0 or row >= len(self._items):
            self._sync_enabled(False)
            return
        item = self._items[row]
        info = item["info"]
        try:
            self._reader = open_video_reader(item["path"])
        except Exception as exc:
            self.log.append(f"打开视频失败: {exc}")
            self._sync_enabled(False)
            return
        self.video_name_label.setText(Path(item["path"]).name)
        self.info_label.setText(
            f"格式: {info.format_name}\n"
            f"分辨率: {info.width} x {info.height}\n"
            f"存储位深/帧率: {info.bit_depth}-bit / {info.fps:.3g} fps\n"
            f"原始采集数位深度: {info.acquisition_bit_depth or info.bit_depth}-bit\n"
            f"Bits per pixel: {info.compression_label or (str(info.bit_depth) + '-bit')}\n"
            f"时长: {info.duration_s:.2f} s\n"
            f"路径: {item['path']}"
        )
        last = max(0, info.frame_count - 1)
        self.frame_slider.blockSignals(True)
        self.frame_slider.setRange(0, last)
        self.frame_slider.setValue(0)
        self.frame_slider.blockSignals(False)
        for spin in (self.start_spin, self.end_spin):
            spin.setRange(0, last)
        self.start_spin.setValue(0)
        self.end_spin.setValue(last)
        self.output_edit.setText(str(Path(item["path"]).with_suffix("")) + "_frames")
        self._reset_adjustments_for_info(info)
        self._current_frame = 0
        self._sync_enabled(True)
        self._refresh_preview()

    def _sync_enabled(self, enabled: bool):
        for widget in (
            self.prev_btn, self.reverse_play_btn, self.play_btn, self.stop_btn,
            self.next_btn, self.frame_slider,
            self.start_spin, self.end_spin, self.step_spin, self.output_edit,
            self.bit_depth_combo, self.format_combo, self.export_btn,
        ):
            widget.setEnabled(enabled)
        for widget in getattr(self, "_adjust_widgets", []):
            widget.setEnabled(enabled)

    def _close_reader(self):
        self._stop_playback()
        if self._reader is not None:
            try:
                self._reader.close()
            except Exception:
                pass
        self._reader = None

    def _on_frame_slider_changed(self, value: int):
        self._current_frame = int(value)
        self._refresh_preview()

    def _seek_relative(self, delta: int):
        if self._play_timer.isActive():
            self._stop_playback()
        self.frame_slider.setValue(
            max(
                self.frame_slider.minimum(),
                min(self.frame_slider.maximum(), self.frame_slider.value() + delta),
            )
        )

    def _play_interval_ms(self) -> int:
        row = self.queue_list.currentRow()
        fps = 0.0
        if 0 <= row < len(self._items):
            fps = float(self._items[row]["info"].fps or 0.0)
        if fps <= 0.0 or fps > 60.0:
            fps = 30.0
        return max(15, int(round(1000.0 / fps)))

    def _start_playback(self, direction: int = 1):
        if self._reader is None:
            return
        self._play_direction = 1 if direction >= 0 else -1
        if self._play_direction > 0 and self.frame_slider.value() >= self.frame_slider.maximum():
            self.frame_slider.setValue(self.frame_slider.minimum())
        elif self._play_direction < 0 and self.frame_slider.value() <= self.frame_slider.minimum():
            self.frame_slider.setValue(self.frame_slider.maximum())
        self.play_btn.setText("暂停")
        self.reverse_play_btn.setText("倒退中")
        self._play_timer.start(self._play_interval_ms())

    def _toggle_playback(self, direction: int = 1):
        if self._play_timer.isActive() and self._play_direction == (1 if direction >= 0 else -1):
            self._stop_playback()
            return
        self._start_playback(direction)

    def _advance_playback(self):
        next_value = self.frame_slider.value() + self._play_direction
        if next_value < self.frame_slider.minimum() or next_value > self.frame_slider.maximum():
            self._stop_playback()
            return
        self.frame_slider.setValue(next_value)

    def _stop_playback(self):
        self._play_timer.stop()
        if hasattr(self, "play_btn"):
            self.play_btn.setText("播放")
        if hasattr(self, "reverse_play_btn"):
            self.reverse_play_btn.setText("倒退播放")

    def _source_max(self) -> int:
        row = self.queue_list.currentRow()
        if 0 <= row < len(self._items):
            info = self._items[row]["info"]
            depth = int(info.acquisition_bit_depth or info.bit_depth or 12)
            return int((1 << min(max(depth, 1), 16)) - 1)
        return 4095

    def _reset_adjustments_for_info(self, info):
        source_max = int((1 << min(max(int(info.acquisition_bit_depth or info.bit_depth or 12), 1), 16)) - 1)
        for widget in (self.bit_min_spin, self.bit_max_spin, self.bit_min_slider, self.bit_max_slider):
            widget.blockSignals(True)
            widget.setRange(0, source_max)
            widget.blockSignals(False)
        self.bit_min_spin.setValue(0)
        self.bit_min_slider.setValue(0)
        self.bit_max_spin.setValue(source_max)
        self.bit_max_slider.setValue(source_max)
        for spin in (self.crop_x_spin, self.crop_y_spin):
            spin.setRange(0, max(info.width, info.height))
            spin.setValue(0)
        self.crop_w_spin.setRange(1, info.width)
        self.crop_h_spin.setRange(1, info.height)
        self.crop_w_spin.setValue(info.width)
        self.crop_h_spin.setValue(info.height)
        self.resample_w_spin.setValue(info.width)
        self.resample_h_spin.setValue(info.height)
        self._reset_adjustments(refresh=False, keep_bit_range=True)

    def _reset_adjustments(self, _checked=False, refresh=True, keep_bit_range=False):
        if not keep_bit_range:
            self.bit_min_spin.setValue(0)
            self.bit_max_spin.setValue(self._source_max())
        self.brightness_spin.setValue(0.0)
        self.gain_spin.setValue(1.0)
        self.gamma_spin.setValue(1.0)
        self.knee_spin.setValue(1.0)
        self.color_combo.setCurrentIndex(0)
        self.filter_combo.setCurrentIndex(0)
        for check in (
            self.flip_h_check, self.flip_v_check, self.rotate_ccw_check,
            self.rotate_cw_check, self.grid_check, self.cross_check,
            self.crop_check, self.resample_check,
        ):
            check.setChecked(False)
        self.hide_crop_check.setChecked(True)
        if refresh:
            self._refresh_preview()

    def _disable_adjustments(self):
        self._reset_adjustments()

    def _adjustment_to_dict(self) -> dict:
        config = self._current_adjustment_config()
        return dict(config.__dict__)

    def _apply_adjustment_dict(self, data: dict):
        self.bit_min_spin.setValue(int(data.get("bit_min", 0)))
        self.bit_max_spin.setValue(int(data.get("bit_max", self._source_max())))
        self.brightness_spin.setValue(float(data.get("brightness", 0.0)))
        self.gain_spin.setValue(float(data.get("gain", 1.0)))
        self.gamma_spin.setValue(float(data.get("gamma", 1.0)))
        self.knee_spin.setValue(float(data.get("knee", 1.0)))
        color_index = {"gray": 0, "jet": 1, "hot": 2, "turbo": 3}.get(str(data.get("color_mode", "gray")), 0)
        filter_index = {"none": 0, "median": 1, "gaussian": 2}.get(str(data.get("filter_mode", "none")), 0)
        self.color_combo.setCurrentIndex(color_index)
        self.filter_combo.setCurrentIndex(filter_index)
        self.flip_h_check.setChecked(bool(data.get("flip_horizontal", False)))
        self.flip_v_check.setChecked(bool(data.get("flip_vertical", False)))
        self.rotate_ccw_check.setChecked(bool(data.get("rotate_ccw", False)))
        self.rotate_cw_check.setChecked(bool(data.get("rotate_cw", False)))
        self.crop_check.setChecked(bool(data.get("crop_enabled", False)))
        self.crop_x_spin.setValue(int(data.get("crop_x", 0)))
        self.crop_y_spin.setValue(int(data.get("crop_y", 0)))
        self.crop_w_spin.setValue(int(data.get("crop_w", self.crop_w_spin.maximum())))
        self.crop_h_spin.setValue(int(data.get("crop_h", self.crop_h_spin.maximum())))
        self.resample_check.setChecked(bool(data.get("resample_enabled", False)))
        self.resample_w_spin.setValue(int(data.get("resample_w", self.resample_w_spin.value())))
        self.resample_h_spin.setValue(int(data.get("resample_h", self.resample_h_spin.value())))
        self._refresh_preview()

    def _save_adjustments(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "保存视频调整配置", "video_adjustment.json", "JSON (*.json)"
        )
        if not path:
            return
        Path(path).write_text(json.dumps(self._adjustment_to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        self.log.append(f"已保存调整配置: {path}")

    def _load_adjustments(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "载入视频调整配置", "", "JSON (*.json);;所有文件 (*.*)"
        )
        if not path:
            return
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self._apply_adjustment_dict(data)
        self.log.append(f"已载入调整配置: {path}")

    def _current_adjustment_config(self) -> VideoAdjustmentConfig:
        color_map = {"灰": "gray", "Jet": "jet", "Hot": "hot", "Turbo": "turbo"}
        filter_map = {"无": "none", "中值 3x3": "median", "高斯 3x3": "gaussian"}
        return VideoAdjustmentConfig(
            bit_min=self.bit_min_spin.value(),
            bit_max=self.bit_max_spin.value(),
            brightness=self.brightness_spin.value(),
            gain=self.gain_spin.value(),
            gamma=self.gamma_spin.value(),
            knee=self.knee_spin.value(),
            color_mode=color_map.get(self.color_combo.currentText(), "gray"),
            filter_mode=filter_map.get(self.filter_combo.currentText(), "none"),
            flip_horizontal=self.flip_h_check.isChecked(),
            flip_vertical=self.flip_v_check.isChecked(),
            rotate_ccw=self.rotate_ccw_check.isChecked(),
            rotate_cw=self.rotate_cw_check.isChecked(),
            crop_enabled=self.crop_check.isChecked(),
            crop_x=self.crop_x_spin.value(),
            crop_y=self.crop_y_spin.value(),
            crop_w=self.crop_w_spin.value(),
            crop_h=self.crop_h_spin.value(),
            resample_enabled=self.resample_check.isChecked(),
            resample_w=self.resample_w_spin.value(),
            resample_h=self.resample_h_spin.value(),
            source_max=self._source_max(),
        )

    def _on_adjustment_changed(self, *_args):
        if self.bit_min_spin.value() >= self.bit_max_spin.value():
            sender = self.sender()
            if sender in (self.bit_min_spin, self.bit_min_slider):
                self.bit_max_spin.setValue(min(self._source_max(), self.bit_min_spin.value() + 1))
            else:
                self.bit_min_spin.setValue(max(0, self.bit_max_spin.value() - 1))
        self._refresh_preview()

    def _draw_overlays(self, image: np.ndarray) -> np.ndarray:
        out = image.copy()
        h, w = out.shape[:2]
        if self.grid_check.isChecked():
            step = max(32, min(w, h) // 8)
            for x in range(step, w, step):
                cv2.line(out, (x, 0), (x, h - 1), (70, 110, 160), 1)
            for y in range(step, h, step):
                cv2.line(out, (0, y), (w - 1, y), (70, 110, 160), 1)
        if self.cross_check.isChecked():
            cv2.line(out, (w // 2, 0), (w // 2, h - 1), (0, 255, 255), 1)
            cv2.line(out, (0, h // 2), (w - 1, h // 2), (0, 255, 255), 1)
        if self.crop_check.isChecked() and not self.hide_crop_check.isChecked():
            x, y = self.crop_x_spin.value(), self.crop_y_spin.value()
            cw, ch = self.crop_w_spin.value(), self.crop_h_spin.value()
            cv2.rectangle(out, (x, y), (min(w - 1, x + cw), min(h - 1, y + ch)), (0, 255, 255), 2)
        return out

    def _update_histogram(self, frame: np.ndarray):
        source_max = max(1, self._source_max())
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([np.clip(gray, 0, source_max).astype(np.float32)], [0], None, [256], [0, source_max])
        hist = hist.ravel()
        canvas = np.full((150, 280, 3), 245, dtype=np.uint8)
        if hist.max() > 0:
            hist = hist / hist.max()
            for x, value in enumerate(hist):
                x0 = int(x * canvas.shape[1] / 256)
                x1 = int((x + 1) * canvas.shape[1] / 256)
                y = int(canvas.shape[0] - 1 - value * (canvas.shape[0] - 20))
                cv2.rectangle(canvas, (x0, y), (max(x0, x1), canvas.shape[0] - 1), (145, 145, 145), -1)
        lo = int(self.bit_min_spin.value() * (canvas.shape[1] - 1) / source_max)
        hi = int(self.bit_max_spin.value() * (canvas.shape[1] - 1) / source_max)
        cv2.line(canvas, (lo, canvas.shape[0] - 1), (hi, 0), (0, 255, 255), 2)
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888).copy()
        self.hist_label.setPixmap(QPixmap.fromImage(qimg))
        self.avg_label.setText(f"Avg: {float(np.mean(gray)):.0f}")

    def _refresh_preview(self):
        if self._reader is None:
            return
        try:
            frame = self._reader.read_frame(self._current_frame)
            config = self._current_adjustment_config()
            preview = render_adjusted_display(frame, config)
            preview = self._draw_overlays(preview)
            self._update_histogram(frame)
            rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            qimg = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888).copy()
            pixmap = QPixmap.fromImage(qimg).scaled(
                self.preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.preview_label.setPixmap(pixmap)
            info = self._items[self.queue_list.currentRow()]["info"]
            self.frame_label.setText(
                f"帧 {self._current_frame:06d} / {max(0, info.frame_count - 1):06d}"
            )
        except Exception as exc:
            self.preview_label.setText(f"预览失败: {exc}")

    def _pick_output_dir(self):
        folder = QFileDialog.getExistingDirectory(
            self, "选择图片输出目录", self.output_edit.text()
        )
        if folder:
            self.output_edit.setText(folder)

    def _export_selected(self):
        self._stop_playback()
        row = self.queue_list.currentRow()
        if row < 0 or row >= len(self._items):
            return
        output_dir = self.output_edit.text().strip()
        if not output_dir:
            QMessageBox.warning(self, "视频导入", "请先设置输出目录。")
            return
        bit_depth = [8, 12, 24][self.bit_depth_combo.currentIndex()]
        self.export_btn.setEnabled(False)
        self.progress.setValue(0)
        self._worker = VideoExportWorker(
            self._items[row]["path"],
            output_dir,
            bit_depth,
            self.format_combo.currentText(),
            self.start_spin.value(),
            self.end_spin.value(),
            self.step_spin.value(),
            self._current_adjustment_config(),
            self,
        )
        self._worker.progress.connect(self._on_export_progress)
        self._worker.finished.connect(self._on_export_finished)
        self._worker.error.connect(self._on_export_error)
        self._worker.start()
        self.log.append(f"开始导出: {Path(self._items[row]['path']).name} -> {output_dir}")

    def _on_export_progress(self, pct: int, message: str):
        self.progress.setValue(pct)
        self.log.append(message)

    def _on_export_finished(self, count: int, output_dir: str):
        self.last_output_dir = output_dir
        self.progress.setValue(100)
        self.export_btn.setEnabled(True)
        self.log.append(f"导出完成: {count} 张 -> {output_dir}")
        self.export_finished.emit(output_dir)
        QMessageBox.information(self, "视频导入", f"导出完成：{count} 张图片\n{output_dir}")

    def _on_export_error(self, message: str):
        self.export_btn.setEnabled(True)
        self.log.append(f"导出失败: {message}")
        QMessageBox.critical(self, "视频导入", message)

    def closeEvent(self, event):
        if self._worker is not None and self._worker.isRunning():
            QMessageBox.warning(self, "视频导入", "正在导出图片，请等待导出完成。")
            event.ignore()
            return
        self._close_reader()
        super().closeEvent(event)


class VideoImportDialog(QDialog):
    """Compatibility wrapper for using the video importer as a dialog."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("视频导入")
        self.resize(1260, 760)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.widget = VideoImportWidget(self)
        layout.addWidget(self.widget)

    @property
    def last_output_dir(self) -> str:
        return self.widget.last_output_dir
