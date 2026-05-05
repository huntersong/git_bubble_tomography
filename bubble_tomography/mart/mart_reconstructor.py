"""
MART (Multiplicative Algebraic Reconstruction Technique) 层析重建算法

用于多相机视角下的气泡三维重建。
基于光线追踪的代数重建方法，通过多视角投影数据迭代重建体素场。
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class MARTConfig:
    """MART算法配置参数"""
    # 重建体素网格
    grid_size: Tuple[int, int, int] = (64, 64, 64)
    # 重建区域物理尺寸 (mm)
    domain_size: Tuple[float, float, float] = (20.0, 20.0, 20.0)
    # MART松弛因子 mu (0 < mu <= 1)
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
    # 是否使用SART改进（Simultaneous ART）
    use_sart_mode: bool = False


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


class MARTReconstructor:
    """
    MART层析重建器
    
    算法原理:
    对每个体素j和每个投影射线i:
        f_j^{new} = f_j^{old} * (p_i / sum_k(w_ik * f_k))^{w_ij * mu}
    
    其中:
    - f_j: 体素j的值（折射率/透射率）
    - p_i: 射线i的测量投影值
    - w_ij: 射线i与体素j的几何权重
    - mu: 松弛因子
    """

    def __init__(self, config: Optional[MARTConfig] = None):
        self.config = config or MARTConfig()
        self.ray_tracer = RayTracer(
            self.config.grid_size,
            self.config.domain_size
        )
        
        # 重建结果
        self.volume: Optional[np.ndarray] = None
        
        # 投影矩阵缓存
        self._projection_matrices: Dict[str, np.ndarray] = {}
        
    def set_projection_matrix(self, camera_id: str, P: np.ndarray):
        """设置相机的投影矩阵 (3x4)"""
        self._projection_matrices[camera_id] = P
    
    def _pixel_to_ray(self,
                      P: np.ndarray,
                      u: float,
                      v: float,
                      K_inv: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        将像素坐标转换为世界坐标中的光线
        
        Parameters
        ----------
        P : np.ndarray (3x4)
            投影矩阵 [K[R|t]]
        u, v : float
            像素坐标
        K_inv : np.ndarray (3x3), optional
            内参矩阵的逆。如果提供，使用直接反投影。
            
        Returns
        -------
        ray_origin : np.ndarray (3,) - 相机光心
        ray_direction : np.ndarray (3,) - 光线方向
        """
        # 光心: P矩阵左3x3的伪逆 乘 -P的第4列
        M = P[:, :3]  # 3x3
        p4 = P[:, 3]  # (3,)
        
        # 光心 = -M^{-1} * p4
        ray_origin = -np.linalg.solve(M, p4)
        
        # 从投影矩阵分解 K, R, t
        K_from_P, R_from_P, t_from_P = self._decompose_P(P)
        pixel_hom = np.array([u, v, 1.0])
        ray_dir_cam = np.linalg.inv(K_from_P) @ pixel_hom
        ray_direction = R_from_P.T @ ray_dir_cam
        
        ray_direction = ray_direction / (np.linalg.norm(ray_direction) + 1e-15)
        
        return ray_origin, ray_direction
    
    def _decompose_P(self, P: np.ndarray):
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
    
    def build_ray_voxel_matrix(self,
                               projections: Dict[str, np.ndarray],
                               camera_params: Dict[str, dict],
                               mask_threshold: float = 0.05) -> Tuple[dict, dict]:
        """
        构建光线-体素权重矩阵（稀疏表示）
        
        Parameters
        ----------
        projections : Dict[str, np.ndarray]
            {相机ID: 投影图像 (灰度, 归一化到[0,1])}
        camera_params : Dict[str, dict]
            {相机ID: {'P': 投影矩阵, 'K_inv': 内参逆矩阵}}
        mask_threshold : float
            投影值低于此阈值的光线被跳过
            
        Returns
        -------
        ray_data : dict - 光线数据
        ray_weights : dict - 体素权重
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
            
            proj_ds = cv2_resize(proj_img, (ds_w, ds_h)) if downsample > 1 else proj_img
            
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
                    
                    ray_origin, ray_dir = self._pixel_to_ray(P, u, v, K_inv)
                    
                    voxels = self.ray_tracer.cast_ray(
                        ray_origin, ray_dir,
                        self.config.ray_sample_step
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
    
    def reconstruct(self,
                    projections: Dict[str, np.ndarray],
                    camera_params: Dict[str, dict],
                    callback=None) -> np.ndarray:
        """
        执行MART重建
        
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
        cfg = self.config
        Nx, Ny, Nz = cfg.grid_size
        
        # 初始化体素场为均匀分布
        self.volume = np.ones((Nx, Ny, Nz), dtype=np.float64) * 0.5
        
        # 构建光线-体素权重
        logger.info("构建光线-体素权重矩阵...")
        rays, weights = self.build_ray_voxel_matrix(projections,
                                                     camera_params)
        
        # 预计算每条光线的权重和 (w_ik * f_k)
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
            error = self._compute_projection_error(rays, weights, projections)
            
            logger.info(f"迭代 {iteration + 1}/{cfg.max_iterations}, "
                        f"体积变化率: {change:.6f}, 投影误差: {error:.4f}")
            
            if callback:
                callback(iteration, self.volume.copy(), error)
            
            if change < cfg.convergence_threshold:
                logger.info(f"收敛于第 {iteration + 1} 次迭代")
                break
        
        return self.volume
    
    def _compute_projection_error(self,
                                   rays: dict,
                                   weights: dict,
                                   projections: dict) -> float:
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
                        weighted_sum += w_dict[ray_idx] * self.volume[vi, vj, vk]
                
                if p_measured > 0.05:
                    error = (weighted_sum - p_measured) ** 2
                    total_error += error
                    total_count += 1
        
        return np.sqrt(total_error / (total_count + 1e-15))
    
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


def cv2_resize(img, size):
    """简单的numpy双线性插值缩放（避免强依赖cv2）"""
    import cv2
    return cv2.resize(img, size)
