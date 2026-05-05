"""
三维互相关速度场计算模块

基于两帧时刻的示踪粒子三维位置，通过互相关算法获得三维速度场。

算法流程:
1. 将重建域划分为规则 interrogation volume（查询体）
2. 在每个查询体内计算两帧粒子的相关函数
3. 从相关峰位置计算位移向量
4. 亚像素精化（三点高斯拟合）
5. 可选: 递归窗口优化 + 中值滤波后处理
6. 空间插值获得规则网格上的速度场

适用场景: Tomographic PIV、3D-PTV
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from scipy.signal import correlate
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import median_filter, uniform_filter
import logging

logger = logging.getLogger(__name__)


@dataclass
class CorrelationConfig:
    """互相关算法配置"""
    # 查询体 (interrogation volume) 尺寸 (mm)
    interrogation_size: Tuple[float, float, float] = (2.0, 2.0, 2.0)
    # 查询体重叠率 (0~0.75)
    overlap_ratio: float = 0.5
    # 是否启用亚像素精化
    subpixel_refinement: bool = True
    # 相关峰最小值（信噪比阈值）
    peak_threshold: float = 1.2
    # 是否启用递归窗口优化
    recursive_refinement: bool = False
    # 递归级别
    recursive_levels: int = 2
    # 是否启用中值滤波去噪
    median_filter: bool = True
    # 中值滤波核大小
    median_kernel_size: int = 3
    # 位移范围限制 (mm) - 超过此范围认为无效
    max_displacement: float = 5.0
    # 置信度阈值
    q_threshold: float = 0.3


@dataclass
class VelocityVector:
    """单个网格点的速度向量"""
    position: np.ndarray              # (x, y, z) 网格中心
    displacement: np.ndarray          # (dx, dy, dz) 位移 mm
    velocity: np.ndarray              # (vx, vy, vz) mm/s
    correlation_peak: float           # 相关峰值
    signal_to_noise: float            # 信噪比
    quality: float                    # 质量标记 (0~1)


class VolumeCorrelator:
    """
    三维体互相关计算器
    
    将两帧粒子位置光栅化为三维体强度场，
    然后在查询体内计算三维互相关函数。
    """
    
    def __init__(self, config: CorrelationConfig):
        self.config = config
    
    def particles_to_volume(self,
                            particles: List,
                            domain_size: Tuple[float, float, float],
                            grid_size: Tuple[int, int, int],
                            weights: Optional[List[float]] = None) -> np.ndarray:
        """
        将粒子位置光栅化为三维体强度场
        
        Parameters
        ----------
        particles : List[Particle3D]
            粒子列表
        domain_size : Tuple[float, float, float]
            重建域尺寸 (mm)
        grid_size : Tuple[int, int, int]
            光栅化网格分辨率
        weights : List[float], optional
            粒子权重（如亮度、质量等）
            
        Returns
        -------
        volume : np.ndarray (Nx, Ny, Nz)
        """
        volume = np.zeros(grid_size, dtype=np.float64)
        origin = -np.array(domain_size) / 2
        voxel_size = np.array(domain_size) / np.array(grid_size)
        
        for i, p in enumerate(particles):
            pos = p.position if hasattr(p, 'position') else p
            
            # 体素索引
            idx = ((pos - origin) / voxel_size).astype(int)
            
            if np.any(idx < 0) or np.any(idx >= grid_size):
                continue
            
            w = weights[i] if weights else 1.0
            
            # 三线性插值分配到周围体素
            for dx in range(2):
                for dy in range(2):
                    for dz in range(2):
                        ix = min(idx[0] + dx, grid_size[0] - 1)
                        iy = min(idx[1] + dy, grid_size[1] - 1)
                        iz = min(idx[2] + dz, grid_size[2] - 1)
                        
                        # 简单分配
                        volume[ix, iy, iz] += w
        
        return volume
    
    def cross_correlate_3d(self,
                           volume1: np.ndarray,
                           volume2: np.ndarray,
                           center_ijk: Tuple[int, int, int],
                           half_window: int) -> Tuple[np.ndarray, float, float]:
        """
        三维互相关计算
        
        Parameters
        ----------
        volume1, volume2 : np.ndarray
            两帧的体强度场
        center_ijk : Tuple[int, int, int]
            查询体中心
        half_window : int
            查询体半宽
            
        Returns
        -------
        displacement : np.ndarray (3,)
            位移 (体素索引偏移)
        peak_value : float
            相关峰高度
        snr : float
            信噪比
        """
        Nx, Ny, Nz = volume1.shape
        ci, cj, ck = center_ijk
        
        i0 = max(0, ci - half_window)
        i1 = min(Nx, ci + half_window + 1)
        j0 = max(0, cj - half_window)
        j1 = min(Ny, cj + half_window + 1)
        k0 = max(0, ck - half_window)
        k1 = min(Nz, ck + half_window + 1)
        
        win1 = volume1[i0:i1, j0:j1, k0:k1].copy()
        win2 = volume2[i0:i1, j0:j1, k0:k1].copy()
        
        if win1.size == 0 or win2.size == 0:
            return np.array([0.0, 0.0, 0.0]), 0.0, 0.0
        
        # 归一化
        win1 = (win1 - win1.mean()) / (win1.std() + 1e-10)
        win2 = (win2 - win2.mean()) / (win2.std() + 1e-10)
        
        # 三维互相关 (FFT加速)
        correlation = correlate(win1, win2, mode='full')
        
        # 找到相关峰
        peak_idx = np.unravel_index(np.argmax(correlation), correlation.shape)
        
        # 将相关函数坐标转换为位移
        win_size = np.array(win1.shape)
        displacement = np.array(peak_idx) - (win_size - 1)
        
        peak_value = float(correlation[peak_idx])
        
        # 计算信噪比（峰值 / 次峰）
        # 简单方法：排除中心附近区域后的最大值
        mask = np.ones(correlation.shape, dtype=bool)
        margin = 1
        peak_range = tuple(
            slice(max(0, pi - margin), min(s, pi + margin + 1))
            for pi, s in zip(peak_idx, correlation.shape)
        )
        mask[peak_range] = False
        
        if mask.any():
            second_peak = correlation[mask].max()
        else:
            second_peak = 0.0
        
        snr = peak_value / (second_peak + 1e-10)
        
        # 亚像素精化
        if self.config.subpixel_refinement:
            displacement = self._subpixel_refine(
                correlation, peak_idx, displacement
            )
        
        return displacement, peak_value, snr
    
    def _subpixel_refine(self,
                         correlation: np.ndarray,
                         peak_idx: Tuple[int, int, int],
                         displacement: np.ndarray) -> np.ndarray:
        """三点高斯亚像素精化"""
        sub_disp = displacement.copy().astype(float)
        
        for dim in range(3):
            pi = peak_idx[dim]
            shape_dim = correlation.shape[dim]
            
            if pi > 0 and pi < shape_dim - 1:
                # 三个点
                idx_prev = list(peak_idx)
                idx_prev[dim] -= 1
                idx_next = list(peak_idx)
                idx_next[dim] += 1
                
                y_prev = float(correlation[tuple(idx_prev)])
                y_peak = float(correlation[peak_idx])
                y_next = float(correlation[tuple(idx_next)])
                
                # 高斯拟合
                denom = 2 * (2 * y_peak - y_prev - y_next)
                if abs(denom) > 1e-10:
                    sub_disp[dim] += (y_prev - y_next) / denom
                # 防止偏移超过 ±0.5
                sub_disp[dim] = np.clip(sub_disp[dim],
                                        displacement[dim] - 0.5,
                                        displacement[dim] + 0.5)
        
        return sub_disp


class VelocityFieldCalculator:
    """
    三维速度场计算器
    
    输入: 两帧时刻的三维粒子位置
    输出: 规则网格上的三维速度场
    """
    
    def __init__(self, config: Optional[CorrelationConfig] = None,
                 domain_size: Tuple[float, float, float] = (20, 20, 20),
                 dt: float = 1.0):
        """
        Parameters
        ----------
        config : CorrelationConfig
        domain_size : Tuple[float, float, float]
            重建域物理尺寸 (mm)
        dt : float
            两帧之间的时间间隔 (s)
        """
        self.config = config or CorrelationConfig()
        self.domain_size = domain_size
        self.dt = dt
        self.correlator = VolumeCorrelator(self.config)
    
    def compute_velocity_field(
            self,
            particles_frame1: List,
            particles_frame2: List,
            grid_resolution: Optional[Tuple[int, int, int]] = None
    ) -> Dict:
        """
        计算三维速度场
        
        Parameters
        ----------
        particles_frame1 : List[Particle3D]
            第一帧粒子
        particles_frame2 : List[Particle3D]
            第二帧粒子
        grid_resolution : Tuple[int, int, int], optional
            输出网格分辨率。默认根据查询体大小自动计算。
            
        Returns
        -------
        result : Dict
            {
                'velocity_field': np.ndarray (Nx, Ny, Nz, 3),
                'grid_positions': np.ndarray (Nx, Ny, Nz, 3),
                'snr_field': np.ndarray (Nx, Ny, Nz),
                'displacement_field': np.ndarray (Nx, Ny, Nz, 3),
                'config': dict
            }
        """
        cfg = self.config
        
        # 自动计算网格分辨率
        if grid_resolution is None:
            voxels_per_dim = int(cfg.interrogation_size[0] / 
                                 (self.domain_size[0] / 32))
            voxels_per_dim = max(8, min(64, voxels_per_dim))
            grid_resolution = (voxels_per_dim, voxels_per_dim, voxels_per_dim)
        
        # Step 1: 光栅化为体强度场
        logger.info("Step 1: 光栅化粒子为体强度场...")
        vol1 = self.correlator.particles_to_volume(
            particles_frame1, self.domain_size, grid_resolution
        )
        vol2 = self.correlator.particles_to_volume(
            particles_frame2, self.domain_size, grid_resolution
        )
        
        # 高斯平滑（模拟粒子图像扩散）
        from scipy.ndimage import gaussian_filter
        vol1 = gaussian_filter(vol1, sigma=0.5)
        vol2 = gaussian_filter(vol2, sigma=0.5)
        
        # Step 2: 计算查询体参数
        vox_size = np.array(self.domain_size) / np.array(grid_resolution)
        half_win = max(1, int(cfg.interrogation_size[0] / (2 * vox_size[0])))
        
        step = max(1, int(half_win * 2 * (1 - cfg.overlap_ratio)))
        if step < 1:
            step = 1
        
        # Step 3: 逐查询体计算互相关
        logger.info("Step 2: 计算三维互相关...")
        Nx, Ny, Nz = grid_resolution
        
        # 输出网格
        out_nx = max(1, (Nx - 2 * half_win) // step + 1)
        out_ny = max(1, (Ny - 2 * half_win) // step + 1)
        out_nz = max(1, (Nz - 2 * half_win) // step + 1)
        
        displacement_field = np.zeros((out_nx, out_ny, out_nz, 3))
        snr_field = np.zeros((out_nx, out_ny, out_nz))
        peak_field = np.zeros((out_nx, out_ny, out_nz))
        
        count = 0
        total = out_nx * out_ny * out_nz
        
        for i_out in range(out_nx):
            for j_out in range(out_ny):
                for k_out in range(out_nz):
                    i_in = half_win + i_out * step
                    j_in = half_win + j_out * step
                    k_in = half_win + k_out * step
                    
                    disp, peak, snr = self.correlator.cross_correlate_3d(
                        vol1, vol2, (i_in, j_in, k_in), half_win
                    )
                    
                    displacement_field[i_out, j_out, k_out] = disp
                    snr_field[i_out, j_out, k_out] = snr
                    peak_field[i_out, j_out, k_out] = peak
                    
                    count += 1
        
        logger.info(f"  完成 {count}/{total} 个查询体")
        
        # Step 4: 转换为物理位移 (体素 → mm)
        displacement_mm = displacement_field * vox_size
        
        # 位移范围限制
        max_disp = cfg.max_displacement
        disp_magnitude = np.linalg.norm(displacement_mm, axis=-1, keepdims=True)
        mask = disp_magnitude > max_disp
        displacement_mm = np.where(mask, displacement_mm * max_disp / (disp_magnitude + 1e-10),
                                    displacement_mm)
        
        # Step 5: 中值滤波后处理
        if cfg.median_filter:
            logger.info("Step 3: 中值滤波后处理...")
            for d in range(3):
                displacement_mm[:, :, :, d] = median_filter(
                    displacement_mm[:, :, :, d],
                    size=cfg.median_kernel_size
                )
        
        # Step 6: 计算速度场
        velocity_field = displacement_mm / self.dt
        
        # 质量标记
        quality = snr_field.copy()
        quality = (quality - quality.min()) / (quality.max() - quality.min() + 1e-10)
        quality[snr_field < cfg.peak_threshold] = 0
        
        # 生成网格坐标
        origin = -np.array(self.domain_size) / 2
        step_mm = np.array(self.domain_size) / np.array(grid_resolution) * step
        gx = np.arange(out_nx) * step_mm[0] + origin[0] + half_win * vox_size[0]
        gy = np.arange(out_ny) * step_mm[1] + origin[1] + half_win * vox_size[1]
        gz = np.arange(out_nz) * step_mm[2] + origin[2] + half_win * vox_size[2]
        
        grid_positions = np.zeros((out_nx, out_ny, out_nz, 3))
        GX, GY, GZ = np.meshgrid(gx, gy, gz, indexing='ij')
        grid_positions[:, :, :, 0] = GX
        grid_positions[:, :, :, 1] = GY
        grid_positions[:, :, :, 2] = GZ
        
        result = {
            'velocity_field': velocity_field,
            'grid_positions': grid_positions,
            'displacement_field': displacement_mm,
            'snr_field': snr_field,
            'peak_field': peak_field,
            'quality_field': quality,
            'grid_shape': (out_nx, out_ny, out_nz),
            'voxel_size': vox_size,
            'config': {
                'domain_size': self.domain_size,
                'dt': self.dt,
                'interrogation_size': cfg.interrogation_size,
                'overlap_ratio': cfg.overlap_ratio,
                'grid_resolution': grid_resolution
            }
        }
        
        # 统计信息
        valid_mask = quality > 0
        n_valid = np.sum(valid_mask)
        avg_velocity = np.mean(np.linalg.norm(
            velocity_field[valid_mask], axis=-1
        )) if n_valid > 0 else 0
        max_velocity = np.max(np.linalg.norm(
            velocity_field, axis=-1
        )) if velocity_field.size > 0 else 0
        
        logger.info(f"速度场计算完成:")
        logger.info(f"  网格: {out_nx} x {out_ny} x {out_nz}")
        logger.info(f"  有效向量: {n_valid}/{out_nx * out_ny * out_nz}")
        logger.info(f"  平均速度: {avg_velocity:.2f} mm/s")
        logger.info(f"  最大速度: {max_velocity:.2f} mm/s")
        logger.info(f"  信噪比均值: {np.mean(snr_field[valid_mask]):.2f}" 
                     if n_valid > 0 else "  无有效数据")
        
        return result
    
    def compute_velocity_field_direct(
            self,
            particles_frame1: List,
            particles_frame2: List,
            max_match_distance: float = 2.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        直接粒子匹配法（PTV风格）
        
        通过最近邻匹配两帧粒子，直接计算位移。
        适用于粒子浓度较低的场合。
        
        Parameters
        ----------
        particles_frame1, particles_frame2 : List[Particle3D]
        max_match_distance : float
            最大匹配距离 (mm)
            
        Returns
        -------
        matched_positions : np.ndarray (N, 3)
            匹配上的粒子位置
        velocities : np.ndarray (N, 3)
            速度向量 (mm/s)
        """
        if len(particles_frame1) == 0 or len(particles_frame2) == 0:
            return np.array([]).reshape(0, 3), np.array([]).reshape(0, 3)
        
        # 构建位置数组
        pos1 = np.array([p.position for p in particles_frame1])
        pos2 = np.array([p.position for p in particles_frame2])
        
        # KDTree快速最近邻搜索
        tree = KDTree(pos2)
        
        matched_pos = []
        matched_vel = []
        used = set()
        
        for i, p1 in enumerate(particles_frame1):
            dist, idx = tree.query(p1.position, k=min(3, len(pos2)))
            
            if len(dist) == 0 or dist[0] > max_match_distance:
                continue
            
            # 选择最近的未被匹配的粒子
            best_idx = -1
            for j, d in enumerate(dist):
                if d > max_match_distance:
                    break
                if idx[j] not in used:
                    best_idx = idx[j]
                    break
            
            if best_idx < 0:
                continue
            
            used.add(best_idx)
            displacement = pos2[best_idx] - p1.position
            velocity = displacement / self.dt
            
            matched_pos.append(p1.position)
            matched_vel.append(velocity)
        
        if len(matched_pos) == 0:
            return np.array([]).reshape(0, 3), np.array([]).reshape(0, 3)
        
        return np.array(matched_pos), np.array(matched_vel)


class VelocityPostProcessor:
    """速度场后处理器"""
    
    @staticmethod
    def remove_outliers(velocity_field: np.ndarray,
                        snr_field: np.ndarray,
                        snr_threshold: float = 1.2,
                        max_velocity: Optional[float] = None) -> np.ndarray:
        """移除低信噪比和异常速度向量"""
        cleaned = velocity_field.copy()
        
        # SNR阈值
        mask = snr_field < snr_threshold
        cleaned[mask] = 0
        
        # 速度幅值阈值
        if max_velocity is not None:
            speed = np.linalg.norm(cleaned, axis=-1)
            mask_speed = speed > max_velocity
            cleaned[mask_speed] = 0
        
        return cleaned
    
    @staticmethod
    def smooth_velocity_field(velocity_field: np.ndarray,
                               method: str = 'median',
                               kernel_size: int = 3) -> np.ndarray:
        """平滑速度场"""
        from scipy.ndimage import median_filter, uniform_filter
        
        smoothed = np.zeros_like(velocity_field)
        for d in range(3):
            if method == 'median':
                smoothed[:, :, :, d] = median_filter(
                    velocity_field[:, :, :, d], size=kernel_size
                )
            elif method == 'uniform':
                smoothed[:, :, :, d] = uniform_filter(
                    velocity_field[:, :, :, d], size=kernel_size
                )
            elif method == 'gaussian':
                from scipy.ndimage import gaussian_filter
                smoothed[:, :, :, d] = gaussian_filter(
                    velocity_field[:, :, :, d], sigma=kernel_size / 2
                )
        
        return smoothed
    
    @staticmethod
    def interpolate_to_grid(positions: np.ndarray,
                             velocities: np.ndarray,
                             grid_size: Tuple[int, int, int],
                             domain_size: Tuple[float, float, float],
                             method: str = 'linear') -> np.ndarray:
        """
        将散点速度插值到规则网格
        
        Parameters
        ----------
        positions : np.ndarray (N, 3)
        velocities : np.ndarray (N, 3)
        grid_size : Tuple[int, int, int]
        domain_size : Tuple[float, float, float]
        method : str
            'linear', 'nearest', 'cubic'
            
        Returns
        -------
        grid_velocity : np.ndarray (Nx, Ny, Nz, 3)
        """
        origin = -np.array(domain_size) / 2
        grid_axes = [
            np.linspace(origin[d], origin[d] + domain_size[d], grid_size[d])
            for d in range(3)
        ]
        
        grid_velocity = np.zeros(grid_size + (3,))
        
        for d in range(3):
            interpolator = RegularGridInterpolator(
                grid_axes,
                np.zeros(grid_size),
                method=method,
                bounds_error=False,
                fill_value=0.0
            )
            
            # 将散点数据光栅化
            vol = np.zeros(grid_size)
            vox_size = np.array(domain_size) / np.array(grid_size)
            
            for i, pos in enumerate(positions):
                idx = ((pos - origin) / vox_size).astype(int)
                if np.all(idx >= 0) and np.all(idx < grid_size):
                    vol[idx[0], idx[1], idx[2]] = velocities[i, d]
            
            # 插值
            interpolator.values = vol
            coords = np.mgrid[0:grid_size[0],
                              0:grid_size[1],
                              0:grid_size[2]].reshape(3, -1).T
            pts = np.array([
                grid_axes[0][coords[:, 0]],
                grid_axes[1][coords[:, 1]],
                grid_axes[2][coords[:, 2]]
            ]).T
            
            interp_vals = interpolator(pts)
            grid_velocity[:, :, :, d] = interp_vals.reshape(grid_size)
        
        return grid_velocity
