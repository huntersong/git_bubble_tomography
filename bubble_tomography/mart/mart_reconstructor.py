"""
层析重建算法模块

提供三种层析重建算法：
  - MART  (Multiplicative Algebraic Reconstruction Technique)
  - SMART (Simultaneous MART)
  - Convolutional-SMART (卷积加速SMART)

统一接口：TomographicReconstructor.reconstruct()
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import logging
from tqdm import tqdm
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


# ============================================================
#  配置参数
# ============================================================

@dataclass
class ReconstructionConfig:
    """层析重建通用配置参数"""
    # 重建体素网格
    grid_size: Tuple[int, int, int] = (64, 64, 64)
    # 重建区域物理尺寸 (mm)
    domain_size: Tuple[float, float, float] = (20.0, 20.0, 20.0)
    # 松弛因子 mu (0 < mu <= 1)
    relaxation_factor: float = 0.5
    # 最大迭代次数
    max_iterations: int = 50
    # 收敛阈值（体积变化率）
    convergence_threshold: float = 1e-4
    # 体素阈值（用于提取气泡表面）
    voxel_threshold: float = 0.1
    # 权重下限（避免数值不稳定）
    weight_epsilon: float = 1e-10
    # 每条光线的采样点数
    rays_per_pixel: int = 1
    # 光线采样间距 (mm)
    ray_sample_step: float = 0.2
    # 算法类型: "MART", "SMART", "ConvSMART"
    algorithm: str = "MART"

    # --- SMART 专用 ---
    # SMART 同步更新时是否归一化权重
    smart_normalize_weights: bool = True

    # --- Conv-SMART 专用 ---
    # 卷积核尺寸（体素数），用于 PSF 近似
    conv_kernel_size: int = 5
    # 是否使用 FFT 加速卷积
    use_fft_convolution: bool = True
    # PSF 类型: "gaussian", "tophat", "empirical"
    psf_type: str = "gaussian"
    # 高斯 PSF 的 sigma (体素单位)
    psf_sigma: float = 1.0


# 保留向后兼容别名
MARTConfig = ReconstructionConfig


# ============================================================
#  光线追踪器
# ============================================================

class RayTracer:
    """
    光线追踪器
    从相机像素坐标发射光线，计算其与重建域内体素的交点及权重
    """

    def __init__(self, grid_size: Tuple[int, int, int],
                 domain_size: Tuple[float, float, float]):
        self.grid_size = np.array(grid_size, dtype=np.int32)
        self.domain_size = np.array(domain_size, dtype=np.float64)

        # 体素尺寸
        self.voxel_size = self.domain_size / self.grid_size

        # 重建域边界
        self.origin = -self.domain_size / 2  # 中心对称

    def get_voxel_index(self, point: np.ndarray) -> Optional[Tuple[int, int, int]]:
        """将物理坐标转换为体素索引"""
        idx = ((point - self.origin) / self.voxel_size).astype(np.int32)

        if np.any(idx < 0) or np.any(idx >= self.grid_size):
            return None
        return tuple(idx)

    def cast_ray(self,
                 ray_origin: np.ndarray,
                 ray_direction: np.ndarray,
                 step: float = 0.2) -> List[Tuple[int, int, int, float]]:
        """
        从相机发射一条光线，返回与重建域相交的体素列表及其权重

        Parameters
        ----------
        ray_origin : np.ndarray (3,)
            光线起点（相机光心在世界坐标中的位置）
        ray_direction : np.ndarray (3,)
            光线方向（归一化）
        step : float
            采样步长 (mm)

        Returns
        -------
        List[Tuple[voxel_i, voxel_j, voxel_k, weight]]
        """
        direction = ray_direction / (np.linalg.norm(ray_direction) + 1e-15)

        # 计算光线与重建域AABB的交点
        t_min, t_max = self._ray_aabb_intersection(ray_origin, direction)

        if t_min is None or t_max is None or t_min > t_max:
            return []

        # 沿光线采样
        t_start = max(t_min, 0)
        t_end = t_max

        if t_end - t_start < step * 0.1:
            return []

        t_values = np.arange(t_start, t_end, step)
        if len(t_values) == 0:
            return []

        sample_points = ray_origin + np.outer(t_values, direction)

        # 统计各体素的权重（穿过的采样点数）
        voxel_weights = {}

        for point in sample_points:
            idx = self.get_voxel_index(point)
            if idx is not None:
                if idx in voxel_weights:
                    voxel_weights[idx] += step
                else:
                    voxel_weights[idx] = step

        # 转换为列表
        result = []
        for (i, j, k), w in voxel_weights.items():
            result.append((i, j, k, w))

        return result

    def _ray_aabb_intersection(self,
                                origin: np.ndarray,
                                direction: np.ndarray
                                ) -> Tuple[Optional[float], Optional[float]]:
        """计算光线与轴对齐包围盒(AABB)的交点参数t"""
        bmin = self.origin
        bmax = self.origin + self.domain_size

        t_min = -np.inf
        t_max = np.inf

        for i in range(3):
            if abs(direction[i]) < 1e-12:
                if origin[i] < bmin[i] or origin[i] > bmax[i]:
                    return None, None
            else:
                t1 = (bmin[i] - origin[i]) / direction[i]
                t2 = (bmax[i] - origin[i]) / direction[i]

                if t1 > t2:
                    t1, t2 = t2, t1

                t_min = max(t_min, t1)
                t_max = min(t_max, t2)

                if t_min > t_max:
                    return None, None

        return t_min, t_max


# ============================================================
#  投影工具函数
# ============================================================

def pixel_to_ray(P: np.ndarray, u: float, v: float,
                 K_inv: Optional[np.ndarray] = None
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """
    将像素坐标转换为世界坐标中的光线

    Parameters
    ----------
    P : np.ndarray (3x4)
        投影矩阵 [K[R|t]]
    u, v : float
        像素坐标
    K_inv : np.ndarray (3x3), optional
        内参矩阵的逆

    Returns
    -------
    ray_origin : np.ndarray (3,) - 相机光心
    ray_direction : np.ndarray (3,) - 光线方向
    """
    M = P[:, :3]  # 3x3
    p4 = P[:, 3]  # (3,)

    # 光心 = -M^{-1} * p4
    ray_origin = -np.linalg.solve(M, p4)

    # 从投影矩阵分解 K, R, t
    K_from_P, R_from_P, _ = _decompose_P(P)
    pixel_hom = np.array([u, v, 1.0])
    ray_dir_cam = np.linalg.inv(K_from_P) @ pixel_hom
    ray_direction = R_from_P.T @ ray_dir_cam

    ray_direction = ray_direction / (np.linalg.norm(ray_direction) + 1e-15)

    return ray_origin, ray_direction


def _decompose_P(P: np.ndarray):
    """分解投影矩阵 P = K[R|t]"""
    M = P[:, :3]

    # K = RQ分解 (右正交分解)
    K, R = np.linalg.qr(M.T)
    K = K.T
    R = R.T

    # 确保K的对角元素为正
    for i in range(3):
        if K[i, i] < 0:
            K[:, i] *= -1
            R[i, :] *= -1

    t = np.linalg.solve(M, P[:, 3])

    return K, R, t


def build_ray_voxel_matrix(projections: Dict[str, np.ndarray],
                            camera_params: Dict[str, dict],
                            ray_tracer: RayTracer,
                            ray_sample_step: float = 0.2,
                            mask_threshold: float = 0.05
                            ) -> Tuple[dict, dict]:
    """
    构建光线-体素权重矩阵（稀疏表示）

    Parameters
    ----------
    projections : Dict[str, np.ndarray]
        {相机ID: 投影图像 (灰度, 归一化到[0,1])}
    camera_params : Dict[str, dict]
        {相机ID: {'P': 投影矩阵, 'K_inv': 内参逆矩阵}}
    ray_tracer : RayTracer
        光线追踪器实例
    ray_sample_step : float
        光线追踪采样步长
    mask_threshold : float
        投影值低于此阈值的光线被跳过

    Returns
    -------
    rays : dict - 光线数据
    weights : dict - 体素权重
    """
    rays = {}
    weights = {}

    for cam_id, proj_img in projections.items():
        cam_param = camera_params[cam_id]
        P = cam_param['P']
        K_inv = cam_param.get('K_inv')

        img_h, img_w = proj_img.shape

        # 下采样以提高速度
        downsample = max(1, min(img_w, img_h) // 128)
        ds_h = img_h // downsample
        ds_w = img_w // downsample

        if downsample > 1:
            import cv2
            proj_ds = cv2.resize(proj_img, (ds_w, ds_h))
        else:
            proj_ds = proj_img

        cam_rays = []
        cam_weights = {}

        logger.info(f"相机 {cam_id}: 构建光线矩阵 "
                    f"(下采样到 {ds_w}x{ds_h})")

        for v in tqdm(range(ds_h), desc=f"  光线追踪 {cam_id}",
                      leave=False):
            for u in range(ds_w):
                proj_val = proj_ds[v, u]

                if proj_val < mask_threshold:
                    continue

                ray_origin, ray_dir = pixel_to_ray(P, u, v, K_inv)

                voxels = ray_tracer.cast_ray(
                    ray_origin, ray_dir,
                    ray_sample_step
                )

                if len(voxels) == 0:
                    continue

                ray_idx = len(cam_rays)
                cam_rays.append({
                    'pixel': (u * downsample, v * downsample),
                    'projection': float(proj_val),
                    'camera_id': cam_id
                })

                for (i, j, k, w) in voxels:
                    key = (i, j, k)
                    if key not in cam_weights:
                        cam_weights[key] = {}
                    cam_weights[key][ray_idx] = w

        rays[cam_id] = cam_rays
        weights[cam_id] = cam_weights

        logger.info(f"相机 {cam_id}: {len(cam_rays)} 条有效光线")

    return rays, weights


def compute_projection_error(volume: np.ndarray,
                              rays: dict,
                              weights: dict,
                              projections: dict,
                              threshold: float = 0.05) -> float:
    """计算当前重建体素场的投影误差"""
    total_error = 0.0
    total_count = 0

    for cam_id in rays:
        cam_rays = rays[cam_id]
        cam_weights = weights[cam_id]

        for ray_idx, ray_info in enumerate(cam_rays):
            p_measured = ray_info['projection']
            weighted_sum = 0.0

            for (vi, vj, vk), w_dict in cam_weights.items():
                if ray_idx in w_dict:
                    weighted_sum += w_dict[ray_idx] * volume[vi, vj, vk]

            if p_measured > threshold:
                error = (weighted_sum - p_measured) ** 2
                total_error += error
                total_count += 1

    return np.sqrt(total_error / (total_count + 1e-15))


# ============================================================
#  基类：层析重建器
# ============================================================

class TomographicReconstructor(ABC):
    """层析重建器基类，定义统一接口"""

    def __init__(self, config: Optional[ReconstructionConfig] = None):
        self.config = config or ReconstructionConfig()
        self.ray_tracer = RayTracer(
            self.config.grid_size,
            self.config.domain_size
        )
        self.volume: Optional[np.ndarray] = None
        self._projection_matrices: Dict[str, np.ndarray] = {}

    def set_projection_matrix(self, camera_id: str, P: np.ndarray):
        """设置相机的投影矩阵 (3x4)"""
        self._projection_matrices[camera_id] = P

    @abstractmethod
    def reconstruct(self,
                    projections: Dict[str, np.ndarray],
                    camera_params: Dict[str, dict],
                    callback=None) -> np.ndarray:
        """
        执行层析重建

        Parameters
        ----------
        projections : Dict[str, np.ndarray]
            {相机ID: 投影图像}
        camera_params : Dict[str, dict]
            {相机ID: {'P': 投影矩阵, 'K_inv': 内参逆矩阵}}
        callback : callable, optional
            迭代回调函数 callback(iteration, volume, error)

        Returns
        -------
        volume : np.ndarray (Nx, Ny, Nz)
            重建的三维体素场
        """
        pass

    def extract_bubble_surface(self,
                                threshold: Optional[float] = None) -> np.ndarray:
        """
        从重建体素场中提取气泡点云

        Parameters
        ----------
        threshold : float, optional
            体素阈值，默认使用配置中的值

        Returns
        -------
        point_cloud : np.ndarray (N, 3)
            气泡表面点云坐标 (mm)
        """
        if self.volume is None:
            raise ValueError("请先执行重建")

        threshold = threshold or self.config.voxel_threshold

        # 找到阈值以上的体素
        mask = self.volume > threshold
        indices = np.argwhere(mask)

        # 转换为物理坐标
        voxel_size = self.ray_tracer.voxel_size
        origin = self.ray_tracer.origin

        points = indices.astype(np.float64) * voxel_size + origin + voxel_size / 2

        return points

    def extract_bubble_point_cloud(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        提取气泡等值面点云（仅表面体素）

        Returns
        -------
        points : np.ndarray (N, 3)
            表面点云坐标 (mm)
        normals : np.ndarray (N, 3)
            表面法向量（近似）
        """
        if self.volume is None:
            raise ValueError("请先执行重建")

        cfg = self.config
        volume = self.volume

        # 使用marching cubes提取等值面
        from skimage.measure import marching_cubes

        # 转置为 skimage 需要的顺序 (z, y, x)
        vol_t = volume.transpose(2, 1, 0)

        spacing = self.ray_tracer.voxel_size

        try:
            verts, faces, normals, _ = marching_cubes(
                vol_t,
                level=cfg.voxel_threshold,
                spacing=(spacing[2], spacing[1], spacing[0])
            )

            # 偏移到世界坐标
            origin = self.ray_tracer.origin
            verts[:, 0] += origin[2]
            verts[:, 1] += origin[1]
            verts[:, 2] += origin[0]

            # 重新排列坐标顺序为 (x, y, z)
            points = verts[:, [2, 1, 0]]
            normals_out = normals[:, [2, 1, 0]]

            return points, normals_out

        except Exception as e:
            logger.warning(f"Marching cubes失败: {e}，使用阈值法提取")
            points = self.extract_bubble_surface()
            return points, np.zeros_like(points)

    def get_volume_stats(self) -> dict:
        """获取重建体素场的统计信息"""
        if self.volume is None:
            return {}

        return {
            'grid_size': self.config.grid_size,
            'domain_size_mm': self.config.domain_size,
            'voxel_size_mm': self.ray_tracer.voxel_size.tolist(),
            'volume_min': float(np.min(self.volume)),
            'volume_max': float(np.max(self.volume)),
            'volume_mean': float(np.mean(self.volume)),
            'volume_std': float(np.std(self.volume)),
            'nonzero_voxels': int(np.sum(self.volume > self.config.voxel_threshold)),
            'total_voxels': int(np.prod(self.config.grid_size)),
            'fill_fraction': float(
                np.sum(self.volume > self.config.voxel_threshold) /
                np.prod(self.config.grid_size)
            )
        }


# ============================================================
#  MART 重建器
# ============================================================

class MARTReconstructor(TomographicReconstructor):
    """
    MART (Multiplicative Algebraic Reconstruction Technique) 层析重建器

    算法原理:
    对每个体素j和每个投影射线i:
        f_j^{new} = f_j^{old} * (p_i / sum_k(w_ik * f_k))^{w_ij * mu}

    其中:
    - f_j: 体素j的值
    - p_i: 射线i的测量投影值
    - w_ij: 射线i与体素j的几何权重
    - mu: 松弛因子

    特点：逐光线更新（sequential），收敛快但可能不稳定
    """

    def reconstruct(self,
                    projections: Dict[str, np.ndarray],
                    camera_params: Dict[str, dict],
                    callback=None) -> np.ndarray:
        """执行MART重建"""
        cfg = self.config
        Nx, Ny, Nz = cfg.grid_size

        # 初始化体素场为均匀分布
        self.volume = np.ones((Nx, Ny, Nz), dtype=np.float64) * 0.5

        # 构建光线-体素权重
        logger.info("构建光线-体素权重矩阵...")
        rays, weights = build_ray_voxel_matrix(
            projections, camera_params, self.ray_tracer,
            ray_sample_step=cfg.ray_sample_step
        )

        # 预计算每条光线的权重和
        logger.info("开始MART迭代重建...")

        prev_volume_norm = np.sum(self.volume ** 2)

        for iteration in range(cfg.max_iterations):
            for cam_id in rays:
                cam_rays = rays[cam_id]
                cam_weights = weights[cam_id]

                for ray_idx, ray_info in enumerate(cam_rays):
                    p_measured = ray_info['projection']

                    # 计算光线穿过体素的加权和
                    weighted_sum = 0.0
                    voxel_list = []

                    for (vi, vj, vk), w_dict in cam_weights.items():
                        if ray_idx in w_dict:
                            w = w_dict[ray_idx]
                            f = self.volume[vi, vj, vk]
                            weighted_sum += w * f
                            voxel_list.append((vi, vj, vk, w))

                    if weighted_sum < cfg.weight_epsilon:
                        continue

                    # MART更新: f_j = f_j * (p / sum(w*f))^{w_ij * mu}
                    ratio = p_measured / weighted_sum

                    # 限制ratio范围，避免数值发散
                    ratio = np.clip(ratio, 0.1, 10.0)

                    for (vi, vj, vk, w) in voxel_list:
                        exponent = w * cfg.relaxation_factor
                        # 归一化权重使指数有界
                        w_max = max(w_dict.get(ray_idx, 0)
                                    for _, w_dict in cam_weights.items()
                                    if ray_idx in w_dict) if voxel_list else 1.0
                        if w_max > 0:
                            exponent = (w / w_max) * cfg.relaxation_factor

                        self.volume[vi, vj, vk] *= ratio ** exponent

                        # 限制体素值范围
                        self.volume[vi, vj, vk] = np.clip(
                            self.volume[vi, vj, vk], 0.0, 1.0
                        )

            # 计算收敛性
            current_norm = np.sum(self.volume ** 2)
            change = abs(current_norm - prev_volume_norm) / (prev_volume_norm + 1e-15)
            prev_volume_norm = current_norm

            # 计算投影误差
            error = compute_projection_error(
                self.volume, rays, weights, projections
            )

            logger.info(f"迭代 {iteration + 1}/{cfg.max_iterations}, "
                        f"体积变化率: {change:.6f}, 投影误差: {error:.4f}")

            if callback:
                callback(iteration, self.volume.copy(), error)

            if change < cfg.convergence_threshold:
                logger.info(f"收敛于第 {iteration + 1} 次迭代")
                break

        return self.volume


# ============================================================
#  SMART 重建器
# ============================================================

class SMARTReconstructor(TomographicReconstructor):
    """
    SMART (Simultaneous MART) 层析重建器

    算法原理:
    与MART的区别在于——所有射线的修正因子同时计算后再统一更新体素场，
    而非逐条光线更新。数学上等价于：

        修正因子 C_i = (p_i / sum_k(w_ik * f_k))^{mu}
        f_j^{new} = f_j^{old} * prod_i(C_i)^{w_ij / sum_i(w_ij)}

    优势：
    - 收敛更稳定（不会因光线处理顺序引入偏差）
    - 适合多相机场景
    - 易于并行化（所有光线修正可同时计算）
    """

    def reconstruct(self,
                    projections: Dict[str, np.ndarray],
                    camera_params: Dict[str, dict],
                    callback=None) -> np.ndarray:
        """执行SMART重建"""
        cfg = self.config
        Nx, Ny, Nz = cfg.grid_size

        # 初始化体素场为均匀分布
        self.volume = np.ones((Nx, Ny, Nz), dtype=np.float64) * 0.5

        # 构建光线-体素权重
        logger.info("构建光线-体素权重矩阵...")
        rays, weights = build_ray_voxel_matrix(
            projections, camera_params, self.ray_tracer,
            ray_sample_step=cfg.ray_sample_step
        )

        logger.info("开始SMART迭代重建...")

        # 预计算每个体素涉及的所有光线及其权重（加速查找）
        voxel_ray_map = self._build_voxel_ray_map(rays, weights)

        prev_volume_norm = np.sum(self.volume ** 2)

        for iteration in range(cfg.max_iterations):
            # ---- Step 1: 计算所有光线的修正因子 ----
            correction_factors = {}  # (cam_id, ray_idx) -> float

            for cam_id in rays:
                cam_rays = rays[cam_id]
                cam_weights = weights[cam_id]

                for ray_idx, ray_info in enumerate(cam_rays):
                    p_measured = ray_info['projection']

                    # 计算加权投影
                    weighted_sum = 0.0
                    for (vi, vj, vk), w_dict in cam_weights.items():
                        if ray_idx in w_dict:
                            weighted_sum += w_dict[ray_idx] * self.volume[vi, vj, vk]

                    if weighted_sum < cfg.weight_epsilon:
                        continue

                    # 修正因子 C_i = (p_i / w_sum)^{mu}
                    ratio = p_measured / weighted_sum
                    ratio = np.clip(ratio, 0.1, 10.0)

                    correction_factors[(cam_id, ray_idx)] = ratio ** cfg.relaxation_factor

            # ---- Step 2: 同时更新所有体素 ----
            for (vi, vj, vk), ray_list in voxel_ray_map.items():
                if not ray_list:
                    continue

                # 计算该体素的累积修正
                # f_j^{new} = f_j * prod_i(C_i^{w_ij})  归一化
                log_correction = 0.0
                total_weight = 0.0

                for (cam_id, ray_idx, w_ij) in ray_list:
                    key = (cam_id, ray_idx)
                    if key in correction_factors:
                        C_i = correction_factors[key]
                        log_correction += w_ij * np.log(max(C_i, 1e-15))
                        total_weight += w_ij

                if total_weight > 0:
                    # 归一化：几何平均
                    correction = np.exp(log_correction / total_weight)
                    self.volume[vi, vj, vk] *= correction
                    self.volume[vi, vj, vk] = np.clip(
                        self.volume[vi, vj, vk], 0.0, 1.0
                    )

            # 计算收敛性
            current_norm = np.sum(self.volume ** 2)
            change = abs(current_norm - prev_volume_norm) / (prev_volume_norm + 1e-15)
            prev_volume_norm = current_norm

            # 计算投影误差
            error = compute_projection_error(
                self.volume, rays, weights, projections
            )

            logger.info(f"迭代 {iteration + 1}/{cfg.max_iterations}, "
                        f"体积变化率: {change:.6f}, 投影误差: {error:.4f}")

            if callback:
                callback(iteration, self.volume.copy(), error)

            if change < cfg.convergence_threshold:
                logger.info(f"SMART收敛于第 {iteration + 1} 次迭代")
                break

        return self.volume

    def _build_voxel_ray_map(self, rays: dict, weights: dict) -> dict:
        """
        构建体素→光线映射表（反向索引）

        Returns
        -------
        voxel_ray_map : dict
            {(i,j,k): [(cam_id, ray_idx, w_ij), ...]}
        """
        voxel_ray_map = {}

        for cam_id in weights:
            cam_weights = weights[cam_id]
            for (vi, vj, vk), w_dict in cam_weights.items():
                key = (vi, vj, vk)
                if key not in voxel_ray_map:
                    voxel_ray_map[key] = []
                for ray_idx, w_ij in w_dict.items():
                    voxel_ray_map[key].append((cam_id, ray_idx, w_ij))

        return voxel_ray_map


# ============================================================
#  Convolutional-SMART 重建器
# ============================================================

class ConvSMARTReconstructor(TomographicReconstructor):
    """
    Convolutional-SMART (卷积加速SMART) 层析重建器

    算法原理 (Yang & He, 2025):
    将SMART中的权重矩阵乘法重新诠释为三维卷积运算。

    传统SMART中：
        投影:  p_i = sum_j(w_ij * f_j)    ← 矩阵乘法
        回投影: f_j += w_ij * correction   ← 矩阵乘法

    卷积SMART中：
        投影:  p = f * K_proj               ← 三维卷积
        回投影: f += K_back * correction      ← 三维卷积

    其中 K_proj 和 K_back 为等效卷积核（PSF近似），尺寸远小于权重矩阵。

    优势：
    - 内存占用从 O(N*M) 降至 O(N)（N=体素数, M=像素数）
    - FFT卷积实现，计算加速 20-50 倍（GPU可选）
    - 精度与SMART基本持平

    实现策略：
    1. 将各相机投影图像映射到体素网格上（反投影初始化）
    2. 用 PSF 卷积核近似光线-体素权重
    3. 迭代：正向卷积投影 → 计算修正 → 反向卷积更新
    """

    def reconstruct(self,
                    projections: Dict[str, np.ndarray],
                    camera_params: Dict[str, dict],
                    callback=None) -> np.ndarray:
        """执行Convolutional-SMART重建"""
        cfg = self.config
        Nx, Ny, Nz = cfg.grid_size

        # 初始化体素场
        self.volume = np.ones((Nx, Ny, Nz), dtype=np.float64) * 0.5

        # Step 1: 将各相机投影图像反投影到体素空间
        logger.info("Conv-SMART: 构建反投影投影场...")
        backproj_fields = self._build_backprojection_fields(
            projections, camera_params
        )

        # Step 2: 生成卷积核（PSF）
        logger.info(f"Conv-SMART: 生成{cfg.psf_type} PSF卷积核 "
                    f"(尺寸={cfg.conv_kernel_size}, sigma={cfg.psf_sigma})")
        psf_kernel = self._create_psf_kernel()

        # Step 3: 预计算各相机的投影参考场
        proj_ref_fields = {}
        for cam_id, bp_field in backproj_fields.items():
            # 正向投影：对反投影场做 PSF 卷积（模糊）
            if cfg.use_fft_convolution:
                proj_ref_fields[cam_id] = self._fft_convolve3d(bp_field, psf_kernel)
            else:
                proj_ref_fields[cam_id] = self._direct_convolve3d(bp_field, psf_kernel)

        logger.info("开始Conv-SMART迭代重建...")

        prev_volume_norm = np.sum(self.volume ** 2)

        for iteration in range(cfg.max_iterations):
            # ---- 计算当前体素场的正投影（卷积近似）----
            # 正投影 = volume * PSF
            if cfg.use_fft_convolution:
                forward_proj = self._fft_convolve3d(self.volume, psf_kernel)
            else:
                forward_proj = self._direct_convolve3d(self.volume, psf_kernel)

            # ---- 计算修正场 ----
            total_correction = np.zeros_like(self.volume)
            total_weight = np.zeros_like(self.volume) + cfg.weight_epsilon

            for cam_id, bp_field in backproj_fields.items():
                # 修正比 = 参考投影 / 当前正投影（在体素空间中）
                proj_ref = proj_ref_fields[cam_id]
                current_proj = forward_proj

                # 避免除零
                safe_current = np.where(
                    np.abs(current_proj) > cfg.weight_epsilon,
                    current_proj,
                    1.0
                )

                # 修正因子（取对数，后续指数恢复）
                ratio = np.clip(proj_ref / safe_current, 0.1, 10.0)
                log_ratio = np.log(ratio)

                # 加权回投影修正
                weight = np.abs(bp_field) + cfg.weight_epsilon
                total_correction += weight * log_ratio
                total_weight += weight

            # ---- SMART 更新 ----
            # f_new = f * exp(mu * sum_i(w_i * log(ratio_i)) / sum_i(w_i))
            avg_log_correction = total_correction / total_weight
            correction = np.exp(cfg.relaxation_factor * avg_log_correction)

            self.volume *= correction
            self.volume = np.clip(self.volume, 0.0, 1.0)

            # ---- 收敛判断 ----
            current_norm = np.sum(self.volume ** 2)
            change = abs(current_norm - prev_volume_norm) / (prev_volume_norm + 1e-15)
            prev_volume_norm = current_norm

            # 近似投影误差（用卷积投影代替逐光线计算）
            error = self._approx_projection_error(
                forward_proj, proj_ref_fields
            )

            logger.info(f"迭代 {iteration + 1}/{cfg.max_iterations}, "
                        f"体积变化率: {change:.6f}, 近似投影误差: {error:.4f}")

            if callback:
                callback(iteration, self.volume.copy(), error)

            if change < cfg.convergence_threshold:
                logger.info(f"Conv-SMART收敛于第 {iteration + 1} 次迭代")
                break

        return self.volume

    def _build_backprojection_fields(
            self,
            projections: Dict[str, np.ndarray],
            camera_params: Dict[str, dict]
    ) -> Dict[str, np.ndarray]:
        """
        将各相机的2D投影图像反投影到3D体素空间

        返回: {cam_id: 3D array (Nx, Ny, Nz)}
        """
        cfg = self.config
        Nx, Ny, Nz = cfg.grid_size
        fields = {}

        for cam_id, proj_img in projections.items():
            cam_param = camera_params[cam_id]
            P = cam_param['P']

            img_h, img_w = proj_img.shape

            # 下采样到与体素网格匹配的分辨率
            target_w = min(img_w, Nx)
            target_h = min(img_h, Ny)

            if target_w != img_w or target_h != img_h:
                import cv2
                proj_ds = cv2.resize(proj_img, (target_w, target_h))
            else:
                proj_ds = proj_img

            # 沿深度方向扩展为3D场
            # 简化方法：将2D投影沿光轴方向复制
            # 更精确的方法需要逐光线反投影，这里用快速近似
            field = np.zeros((Nx, Ny, Nz), dtype=np.float64)

            # 将投影图像分配到xz平面，沿y轴复制
            # 使用投影矩阵计算相机的观察方向来决定分配方式
            M = P[:, :3]
            p4 = P[:, 3]
            cam_center = -np.linalg.solve(M, p4)

            # 简化：沿z方向平均分配
            for iz in range(Nz):
                # 插值投影图像到当前z对应的像素行
                z_frac = iz / max(Nz - 1, 1)
                row = int(z_frac * (target_h - 1))
                row = min(row, target_h - 1)

                for ix in range(min(Nx, target_w)):
                    for iy in range(min(Ny, target_h)):
                        field[ix, iy, iz] = proj_ds[
                            min(iy, target_h - 1),
                            min(ix, target_w - 1)
                        ]

            fields[cam_id] = field
            logger.info(f"  相机 {cam_id}: 反投影场已构建")

        return fields

    def _create_psf_kernel(self) -> np.ndarray:
        """
        创建3D PSF卷积核

        Returns
        -------
        kernel : np.ndarray (K, K, K)
            归一化的3D卷积核
        """
        cfg = self.config
        K = cfg.conv_kernel_size
        center = K // 2

        if cfg.psf_type == "gaussian":
            # 3D 高斯 PSF
            coords = np.arange(K) - center
            x, y, z = np.meshgrid(coords, coords, coords, indexing='ij')
            r2 = x**2 + y**2 + z**2
            sigma = cfg.psf_sigma
            kernel = np.exp(-r2 / (2 * sigma**2))

        elif cfg.psf_type == "tophat":
            # 3D 顶帽（均匀圆盘）PSF
            coords = np.arange(K) - center
            x, y, z = np.meshgrid(coords, coords, coords, indexing='ij')
            r = np.sqrt(x**2 + y**2 + z**2)
            radius = center
            kernel = np.where(r <= radius, 1.0, 0.0)

        elif cfg.psf_type == "empirical":
            # 经验 PSF：从标定数据估计（这里用高斯近似）
            coords = np.arange(K) - center
            x, y, z = np.meshgrid(coords, coords, coords, indexing='ij')
            r2 = x**2 + y**2 + z**2
            sigma = cfg.psf_sigma
            kernel = np.exp(-r2 / (2 * sigma**2))
            logger.warning("empirical PSF currently falls back to Gaussian; "
                          "provide measured PSF data for better results.")
        else:
            raise ValueError(f"未知PSF类型: {cfg.psf_type}")

        # 归一化
        kernel = kernel / np.sum(kernel)

        return kernel

    def _fft_convolve3d(self, volume: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """
        使用FFT进行3D卷积（快速实现）

        Parameters
        ----------
        volume : np.ndarray (Nx, Ny, Nz)
        kernel : np.ndarray (K, K, K) - 已归一化

        Returns
        -------
        result : np.ndarray (Nx, Ny, Nz)
        """
        from scipy.signal import fftconvolve

        result = fftconvolve(volume, kernel, mode='same')
        return result

    def _direct_convolve3d(self, volume: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """
        直接3D卷积（空间域，适用于小核）

        Parameters
        ----------
        volume : np.ndarray (Nx, Ny, Nz)
        kernel : np.ndarray (K, K, K) - 已归一化

        Returns
        -------
        result : np.ndarray (Nx, Ny, Nz)
        """
        from scipy.ndimage import convolve

        result = convolve(volume, kernel, mode='constant', cval=0.0)
        return result

    def _approx_projection_error(self,
                                   forward_proj: np.ndarray,
                                   proj_ref_fields: Dict[str, np.ndarray]
                                   ) -> float:
        """
        计算近似投影误差（基于卷积投影场）

        Parameters
        ----------
        forward_proj : np.ndarray
            当前体素场的正投影（卷积结果）
        proj_ref_fields : Dict[str, np.ndarray]
            各相机参考投影场

        Returns
        -------
        error : float
        """
        total_error = 0.0
        count = 0

        for cam_id, ref in proj_ref_fields.items():
            diff = forward_proj - ref
            mask = ref > 0.01  # 只计算有信号的区域
            if np.any(mask):
                total_error += np.sum(diff[mask] ** 2)
                count += np.sum(mask)

        return np.sqrt(total_error / (count + 1e-15))


# ============================================================
#  工厂函数：根据配置创建对应的重建器
# ============================================================

def create_reconstructor(config: Optional[ReconstructionConfig] = None
                         ) -> TomographicReconstructor:
    """
    根据配置中的算法类型创建对应的重建器实例

    Parameters
    ----------
    config : ReconstructionConfig, optional
        重建配置，默认使用 MART

    Returns
    -------
    reconstructor : TomographicReconstructor
        对应算法的重建器实例
    """
    if config is None:
        config = ReconstructionConfig()

    algo = config.algorithm.upper()

    if algo == "MART":
        return MARTReconstructor(config)
    elif algo == "SMART":
        return SMARTReconstructor(config)
    elif algo == "CONVSMART" or algo == "CONV-SMART" or algo == "CONVOLUTIONAL-SMART":
        return ConvSMARTReconstructor(config)
    else:
        raise ValueError(
            f"未知重建算法: {algo}。支持的算法: MART, SMART, ConvSMART"
        )
