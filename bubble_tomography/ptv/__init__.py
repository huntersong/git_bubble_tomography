"""
PTV (Particle Tracking Velocimetry) 模块

基于PTV_report.docx中描述的算法原理，实现多帧粒子跟踪测速功能。

核心功能:
1. 多帧粒子检测与2D坐标提取
2. 四帧前后跟踪算法 (4-Frame Forward-Backward Tracking)
3. 最近邻跟踪算法 (Nearest Neighbor Tracking)
4. 松弛法跟踪算法 (Relaxation Method)
5. Shake-The-Box (STB) 高密度3D-PTV优化
6. 拉格朗日速度场计算

与现有模块的关系:
- 粒子检测复用 particles.particle_reconstructor.ParticleDetector
- 标定参数复用 calibration.camera_calibrator.MultiCameraCalibrator
- 3D重建复用 particles.particle_reconstructor.Particle3DReconstructor
- 可视化复用 visualization.visualizer.ResultVisualizer
"""

from .tracker import (
    PTVTracker,
    PTVConfig,
    TrackingResult,
    ForwardBackwardTracker,
    NearestNeighborTracker,
    RelaxationTracker,
    ShakeTheBoxTracker,
)
from .velocity import PTVVelocityCalculator, VelocityProfile

__all__ = [
    "PTVTracker",
    "PTVConfig",
    "TrackingResult",
    "ForwardBackwardTracker",
    "NearestNeighborTracker",
    "RelaxationTracker",
    "ShakeTheBoxTracker",
    "PTVVelocityCalculator",
    "VelocityProfile",
]
