"""
PTV 速度场计算模块

基于PTV跟踪得到的粒子轨迹，计算拉格朗日速度场。

功能:
1. 轨迹速度计算（中心差分）
2. 拉格朗日速度剖面提取
3. 散点速度插值到规则网格（用于与PIV对比）
4. 湍流统计量计算（雷诺应力、TKE等）
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from scipy.interpolate import griddata, RegularGridInterpolator
from scipy.spatial import KDTree
import logging

logger = logging.getLogger(__name__)


@dataclass
class VelocityProfile:
    """速度场结果"""
    # 散点数据
    positions: np.ndarray = field(default_factory=lambda: np.array([]).reshape(0, 3))  # (N, 3) mm
    velocities: np.ndarray = field(default_factory=lambda: np.array([]).reshape(0, 3))  # (N, 3) mm/s
    track_ids: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))       # 轨迹ID

    # 网格插值数据（可选）
    grid_positions: Optional[np.ndarray] = None      # (Nx, Ny, Nz, 3)
    grid_velocities: Optional[np.ndarray] = None     # (Nx, Ny, Nz, 3)
    grid_shape: Optional[Tuple[int, int, int]] = None

    # 统计
    mean_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))    # 平均速度 mm/s
    std_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))     # 速度标准差 mm/s
    max_speed: float = 0.0                                                     # 最大速度 mm/s
    mean_speed: float = 0.0                                                    # 平均速度大小 mm/s

    # 湍流统计（如果有多条轨迹）
    reynolds_stress: Optional[np.ndarray] = None   # (3, 3) 雷诺应力张量
    tke: float = 0.0                               # 湍流动能 TKE

    # 配置
    dt: float = 0.001
    domain_size: Tuple[float, float, float] = (20.0, 20.0, 20.0)

    def summary(self) -> str:
        """生成速度场统计摘要"""
        lines = [
            "PTV速度场统计",
            f"  粒子数: {len(self.positions)}",
            f"  dt: {self.dt * 1000:.2f} ms",
            f"  平均速度: ({self.mean_velocity[0]:.2f}, {self.mean_velocity[1]:.2f}, {self.mean_velocity[2]:.2f}) mm/s",
            f"  平均速度大小: {self.mean_speed:.2f} mm/s",
            f"  最大速度: {self.max_speed:.2f} mm/s",
            f"  速度标准差: ({self.std_velocity[0]:.2f}, {self.std_velocity[1]:.2f}, {self.std_velocity[2]:.2f}) mm/s",
        ]
        if self.tke > 0:
            lines.append(f"  湍流动能 (TKE): {self.tke:.4f} mm^2/s^2")
        if self.grid_shape is not None:
            lines.append(f"  插值网格: {self.grid_shape}")
        return "\n".join(lines)


class PTVVelocityCalculator:
    """
    PTV速度场计算器

    从跟踪轨迹计算速度场，支持:
    - 中心差分速度计算
    - 拉格朗日速度统计
    - 欧拉网格插值
    - 湍流统计量
    """

    def __init__(self, dt: float = 0.001,
                 domain_size: Tuple[float, float, float] = (20.0, 20.0, 20.0)):
        """
        Parameters
        ----------
        dt : float
            帧间时间间隔 (s)
        domain_size : Tuple[float, float, float]
            重建域物理尺寸 (mm)
        """
        self.dt = dt
        self.domain_size = domain_size

    def compute_from_tracks(self, tracks: list) -> VelocityProfile:
        """
        从轨迹列表计算速度场

        Parameters
        ----------
        tracks : List[Track]
            PTV跟踪得到的轨迹列表

        Returns
        -------
        profile : VelocityProfile
        """
        all_positions = []
        all_velocities = []
        all_track_ids = []

        for track in tracks:
            positions, velocities, _ = track.to_arrays()

            if len(positions) < 2:
                continue

            # 如果轨迹没有速度，用中心差分计算
            if len(velocities) == 0:
                velocities = self._centered_difference(positions, self.dt)

            # 对长轨迹使用中心差分重新计算更精确的速度
            if len(positions) >= 3:
                velocities = self._centered_difference(positions, self.dt)

            for j in range(len(positions)):
                if j < len(velocities):
                    all_positions.append(positions[j])
                    all_velocities.append(velocities[j])
                    all_track_ids.append(track.particle_id)

        if not all_positions:
            return VelocityProfile(dt=self.dt, domain_size=self.domain_size)

        positions = np.array(all_positions)
        velocities = np.array(all_velocities)
        track_ids = np.array(all_track_ids)

        # 统计
        speeds = np.linalg.norm(velocities, axis=1)
        mean_vel = np.mean(velocities, axis=0)
        std_vel = np.std(velocities, axis=0)

        # 湍流统计（需要多条轨迹）
        reynolds_stress = None
        tke = 0.0

        unique_tracks = list(set(all_track_ids))
        if len(unique_tracks) >= 10:
            # 计算每个轨迹的平均速度
            track_mean_vels = {}
            track_counts = {}

            for j in range(len(positions)):
                tid = int(track_ids[j])
                if tid not in track_mean_vels:
                    track_mean_vels[tid] = np.zeros(3)
                    track_counts[tid] = 0
                track_mean_vels[tid] += velocities[j]
                track_counts[tid] += 1

            for tid in track_mean_vels:
                track_mean_vels[tid] /= track_counts[tid]

            # 雷诺应力
            fluctuations = np.zeros_like(velocities)
            for j in range(len(positions)):
                tid = int(track_ids[j])
                fluctuations[j] = velocities[j] - track_mean_vels[tid]

            reynolds_stress = np.zeros((3, 3))
            for i in range(3):
                for j in range(3):
                    reynolds_stress[i, j] = np.mean(fluctuations[:, i] * fluctuations[:, j])

            tke = 0.5 * np.trace(reynolds_stress)

        profile = VelocityProfile(
            positions=positions,
            velocities=velocities,
            track_ids=track_ids,
            mean_velocity=mean_vel,
            std_velocity=std_vel,
            max_speed=float(np.max(speeds)),
            mean_speed=float(np.mean(speeds)),
            reynolds_stress=reynolds_stress,
            tke=tke,
            dt=self.dt,
            domain_size=self.domain_size,
        )

        logger.info(f"PTV速度场计算完成:")
        logger.info(f"  数据点: {len(positions)}")
        logger.info(f"  平均速度: {np.linalg.norm(mean_vel):.2f} mm/s")
        logger.info(f"  最大速度: {np.max(speeds):.2f} mm/s")
        if tke > 0:
            logger.info(f"  TKE: {tke:.4f} mm^2/s^2")

        return profile

    def interpolate_to_grid(self,
                            positions: np.ndarray,
                            velocities: np.ndarray,
                            grid_resolution: Tuple[int, int, int] = (16, 16, 16),
                            method: str = 'linear') -> Tuple[np.ndarray, np.ndarray, Tuple]:
        """
        将散点速度插值到规则欧拉网格

        Parameters
        ----------
        positions : np.ndarray (N, 3)
        velocities : np.ndarray (N, 3)
        grid_resolution : Tuple[int, int, int]
        method : str
            'linear', 'nearest', 'cubic'

        Returns
        -------
        grid_positions : np.ndarray (Nx, Ny, Nz, 3)
        grid_velocities : np.ndarray (Nx, Ny, Nz, 3)
        grid_shape : Tuple[int, int, int]
        """
        origin = -np.array(self.domain_size) / 2

        # 生成网格坐标
        axes = [
            np.linspace(origin[d], origin[d] + self.domain_size[d], grid_resolution[d])
            for d in range(3)
        ]

        GX, GY, GZ = np.meshgrid(axes[0], axes[1], axes[2], indexing='ij')
        grid_points = np.column_stack([GX.ravel(), GY.ravel(), GZ.ravel()])

        # 三维插值
        grid_vel_flat = np.zeros((grid_points.shape[0], 3))

        for d in range(3):
            try:
                grid_vel_flat[:, d] = griddata(
                    positions, velocities[:, d], grid_points,
                    method=method, fill_value=0.0
                )
            except Exception as e:
                logger.warning(f"插值维度 {d} 失败: {e}")
                grid_vel_flat[:, d] = 0.0

        grid_positions = grid_points.reshape(grid_resolution + (3,))
        grid_velocities = grid_vel_flat.reshape(grid_resolution + (3,))

        return grid_positions, grid_velocities, grid_resolution

    def compute_streamlines(self,
                            positions: np.ndarray,
                            velocities: np.ndarray,
                            n_lines: int = 20) -> List[np.ndarray]:
        """
        从散点速度数据提取流线

        Parameters
        ----------
        positions : np.ndarray (N, 3)
        velocities : np.ndarray (N, 3)
        n_lines : int
            流线数量

        Returns
        -------
        streamlines : List[np.ndarray]
            每条流线是 (M, 3) 的坐标数组
        """
        if len(positions) < 3:
            return []

        tree = KDTree(positions)

        # 选择种子点（在速度大小较大的区域）
        speeds = np.linalg.norm(velocities, axis=1)
        seed_indices = np.argsort(speeds)[-n_lines:]

        streamlines = []
        max_steps = 100
        step_size = 0.5  # mm

        for si in seed_indices:
            line = [positions[si].copy()]

            pos = positions[si].copy()
            for _ in range(max_steps):
                # 插值速度
                dists, idxs = tree.query(pos, k=min(5, len(positions)))

                weights = 1.0 / (dists + 1e-10)
                weights /= weights.sum()

                vel = np.sum(
                    [w * velocities[idx] for w, idx in zip(weights, idxs)],
                    axis=0
                )

                speed = np.linalg.norm(vel)
                if speed < 0.01:
                    break

                # 归一化步进
                pos = pos + vel / speed * step_size

                # 检查是否超出域
                half = np.array(self.domain_size) / 2
                if np.any(np.abs(pos) > half * 1.2):
                    break

                line.append(pos.copy())

            if len(line) > 3:
                streamlines.append(np.array(line))

        return streamlines

    @staticmethod
    def _centered_difference(positions: np.ndarray, dt: float) -> np.ndarray:
        """
        中心差分法计算速度

        Parameters
        ----------
        positions : np.ndarray (N, 3)
        dt : float

        Returns
        -------
        velocities : np.ndarray (N-1, 3) or (N, 3)
        """
        n = len(positions)
        if n < 2:
            return np.array([])

        if n == 2:
            return np.array([(positions[1] - positions[0]) / dt])

        velocities = np.zeros((n, 3))
        # 两端用前向/后向差分
        velocities[0] = (positions[1] - positions[0]) / dt
        velocities[-1] = (positions[-1] - positions[-2]) / dt
        # 中间用中心差分
        for i in range(1, n - 1):
            velocities[i] = (positions[i + 1] - positions[i - 1]) / (2 * dt)

        return velocities
