"""
示踪粒子三维位置重建模块

基于多相机视角的粒子图像，通过以下步骤重建粒子的三维空间位置：
1. 各视角粒子检测与图像坐标提取
2. 跨视角粒子匹配（光束交叉 / 外极线约束）
3. 三角测量（最小二乘 / SVD分解）
4. 粗差剔除（RANSAC）

物理背景：Tomographic PIV 实验中，在流场中散布微小示踪粒子，
利用多相机同步拍摄，通过三角测量恢复每个粒子的三维坐标。
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from itertools import combinations
from scipy.optimize import least_squares
from scipy.spatial import KDTree
import logging

logger = logging.getLogger(__name__)


@dataclass
class TriangulationConfig:
    """粒子三角测量配置"""
    # 粒子检测参数
    blob_min_threshold: int = 30          # 粒子检测最低亮度
    blob_max_threshold: int = 255
    blob_filter_by_area: bool = True
    blob_min_area: float = 2.0            # 最小粒子面积 (像素²)
    blob_max_area: float = 200.0          # 最大粒子面积 (像素²)
    blob_circularity_threshold: float = 0.5
    
    # 匹配参数
    epipolar_threshold: float = 3.0       # 外极线匹配容差 (像素)
    triangulation_method: str = 'lstsq'   # 'lstsq' (SVD最小二乘) 或 'optimal' (Hartley最优)
    max_particle_distance: float = 50.0   # 粒子最大三维距离 (mm)
    ransac_confidence: float = 0.99       # RANSAC置信度
    ransac_iterations: int = 1000
    ransac_reproj_threshold: float = 2.0  # 重投影阈值 (像素)
    
    # 光束交叉法参数
    use_beam_crossing: bool = False       # 使用光束交叉法而非纯三角测量
    beam_intersection_tolerance: float = 1.0  # mm


@dataclass
class Particle2D:
    """单个相机的二维粒子检测结果"""
    pixel: Tuple[float, float]        # 像素坐标 (u, v) 亚像素精度
    area: float                       # 粒子面积
    intensity: float                  # 峰值亮度
    circularity: float                # 圆形度
    camera_id: str                    # 所属相机


@dataclass
class Particle3D:
    """三维重建后的粒子"""
    position: np.ndarray              # 三维坐标 (x, y, z) mm
    n_views: int                      # 被多少个相机观测到
    reprojection_errors: Dict[str, float] = field(default_factory=dict)  # 各视角重投影误差
    quality: float = 0.0              # 重构质量评分


class ParticleDetector:
    """粒子检测器 - 从各视角图像中提取示踪粒子坐标"""
    
    def __init__(self, config: TriangulationConfig):
        self.config = config
    
    def detect_particles(self,
                         image: np.ndarray,
                         camera_id: str,
                         undistort_params: Optional[dict] = None) -> List[Particle2D]:
        """
        检测单张图像中的示踪粒子
        
        Parameters
        ----------
        image : np.ndarray
            粒子图像 (灰度或彩色)
        camera_id : str
            相机ID
        undistort_params : dict, optional
            {'camera_matrix': K, 'dist_coeffs': D}
            
        Returns
        -------
        particles : List[Particle2D]
        """
        # 去畸变
        if undistort_params is not None:
            K = np.array(undistort_params['camera_matrix'])
            D = np.array(undistort_params['dist_coeffs'])
            h, w = image.shape[:2]
            new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1, (w, h))
            image = cv2.undistort(image, K, D, None, new_K)
        
        # 转灰度
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 高斯模糊去噪（粒子很小，用小核）
        blurred = cv2.GaussianBlur(gray, (3, 3), 0.5)
        
        # 自适应阈值增强粒子对比度
        # 方法1: 局部自适应阈值
        local_thresh = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=21, C=5
        )
        
        # 方法2: Top-hat变换增强亮点
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        tophat = cv2.morphologyEx(blurred, cv2.MORPH_TOPHAT, kernel)
        
        # 综合两种方法
        combined = cv2.bitwise_and(tophat, local_thresh)
        
        # 粒子检测 - 使用SimpleBlobDetector
        params = cv2.SimpleBlobDetector_Params()
        
        # 阈值
        params.minThreshold = self.config.blob_min_threshold
        params.maxThreshold = self.config.blob_max_threshold
        
        # 面积过滤
        params.filterByArea = self.config.blob_filter_by_area
        params.minArea = self.config.blob_min_area
        params.maxArea = self.config.blob_max_area
        
        # 圆形度过滤
        params.filterByCircularity = True
        params.minCircularity = self.config.blob_circularity_threshold
        
        # 凸度过滤
        params.filterByConvexity = True
        params.minConvexity = 0.3
        
        # 惯性比过滤
        params.filterByInertia = True
        params.minInertiaRatio = 0.2
        
        # 颜色过滤（白色粒子，深色背景）
        params.filterByColor = True
        params.blobColor = 255
        
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(combined)
        
        # 如果blob检测器找不到粒子，回退到连通域检测
        if len(keypoints) == 0:
            keypoints = self._fallback_detection(tophat)
        
        # 转换为Particle2D列表
        particles = []
        for kp in keypoints:
            # 亚像素精化
            x, y = int(kp.pt[0]), int(kp.pt[1])
            x, y, size = self._subpixel_refine(blurred, x, y, 
                                                int(kp.size) + 2)
            
            # 计算圆形度
            circularity = 4 * np.pi * kp.size / (2 * np.pi * (kp.size / 2)) ** 2 \
                if kp.size > 0 else 0
            circularity = min(circularity, 1.0)
            
            particles.append(Particle2D(
                pixel=(x, y),
                area=kp.size,
                intensity=float(blurred[int(round(y)), int(round(x))])
                           if 0 <= int(round(y)) < blurred.shape[0]
                           and 0 <= int(round(x)) < blurred.shape[1]
                           else 0.0,
                circularity=circularity,
                camera_id=camera_id
            ))
        
        logger.info(f"相机 {camera_id}: 检测到 {len(particles)} 个粒子")
        return particles
    
    def _subpixel_refine(self,
                         image: np.ndarray,
                         x: int, y: int,
                         half_size: int = 3) -> Tuple[float, float, float]:
        """亚像素级粒子中心定位"""
        h, w = image.shape
        x0 = max(0, x - half_size)
        x1 = min(w, x + half_size + 1)
        y0 = max(0, y - half_size)
        y1 = min(h, y + half_size + 1)
        
        patch = image[y0:y1, x0:x1].astype(np.float64)
        
        if patch.shape[0] < 3 or patch.shape[1] < 3:
            return float(x), float(y), 1.0
        
        # 质心法
        yy, xx = np.mgrid[:patch.shape[0], :patch.shape[1]]
        total = patch.sum() + 1e-15
        cx = (xx * patch).sum() / total + x0
        cy = (yy * patch).sum() / total + y0
        
        return cx, cy, float(half_size * 2)
    
    def _fallback_detection(self, image: np.ndarray):
        """回退到连通域检测"""
        _, binary = cv2.threshold(image, 0, 255,
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        
        keypoints = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if (self.config.blob_min_area < area < self.config.blob_max_area):
                M = cv2.moments(cnt)
                if M['m00'] > 0:
                    cx = M['m10'] / M['m00']
                    cy = M['m01'] / M['m00']
                    size = np.sqrt(area / np.pi) * 2
                    keypoints.append(cv2.KeyPoint(cx, cy, size))
        
        return keypoints


class Particle3DReconstructor:
    """
    示踪粒子三维重建器
    
    核心算法流程:
    1. 多视角粒子检测 → 2D坐标
    2. 外极线约束 + 光束交叉法 → 粒子匹配
    3. SVD三角测量 → 3D坐标
    4. RANSAC粗差剔除
    """
    
    def __init__(self, config: Optional[TriangulationConfig] = None):
        self.config = config or TriangulationConfig()
        self.detector = ParticleDetector(self.config)
        
        # 缓存基础矩阵和本质矩阵
        self._F_matrices: Dict[Tuple[str, str], np.ndarray] = {}
        self._E_matrices: Dict[Tuple[str, str], np.ndarray] = {}
        
        # 预计算外极线
        self._epipolar_lines: Dict[str, List[np.ndarray]] = {}
    
    def compute_epipolar_geometry(self,
                                   calibrator) -> None:
        """
        预计算所有相机对之间的外极几何
        
        Parameters
        ----------
        calibrator : MultiCameraCalibrator
            已完成标定的标定器
        """
        cam_ids = list(calibrator.camera_params.keys())
        
        for cam1_id, cam2_id in combinations(cam_ids, 2):
            P1 = calibrator.compute_projection_matrix(cam1_id)
            P2 = calibrator.compute_projection_matrix(cam2_id)
            
            # 基础矩阵 F = K2^{-T} E K1^{-1}
            K1 = np.array(calibrator.camera_params[cam1_id].camera_matrix)
            K2 = np.array(calibrator.camera_params[cam2_id].camera_matrix)
            
            # 从投影矩阵计算本质矩阵
            E = self._compute_essential_matrix(P1, P2)
            F = np.linalg.inv(K2).T @ E @ np.linalg.inv(K1)
            
            self._F_matrices[(cam1_id, cam2_id)] = F
            self._E_matrices[(cam1_id, cam2_id)] = E
        
        logger.info(f"已计算 {len(cam_ids)} 个相机之间的外极几何 "
                     f"({len(self._F_matrices)} 对)")
    
    def _compute_essential_matrix(self, P1: np.ndarray,
                                   P2: np.ndarray) -> np.ndarray:
        """从两个投影矩阵计算本质矩阵"""
        # 投影矩阵分解 P = K[R|t]
        M1 = P1[:, :3]
        c1 = -np.linalg.solve(M1, P1[:, 3])
        
        M2 = P2[:, :3]
        c2 = -np.linalg.solve(M2, P2[:, 3])
        
        # 平移向量
        t = c2 - c1
        
        # 旋转矩阵 (从P1到P2)
        K1, R1, _ = self._decompose_projection(M1)
        K2, R2, _ = self._decompose_projection(M2)
        R = R2 @ R1.T
        
        # 反对称矩阵 [t]_x
        t_skew = np.array([
            [0, -t[2], t[1]],
            [t[2], 0, -t[0]],
            [-t[1], t[0], 0]
        ])
        
        E = t_skew @ R
        return E
    
    def _decompose_projection(self, M: np.ndarray):
        """分解 M = KR"""
        K, R = np.linalg.qr(M.T)
        K = K.T
        R = R.T
        for i in range(3):
            if K[i, i] < 0:
                K[:, i] *= -1
                R[i, :] *= -1
        return K, R, None
    
    def get_epipolar_line(self, point: Tuple[float, float],
                           cam_from: str, cam_to: str) -> np.ndarray:
        """
        计算从 cam_from 中一个点到 cam_to 的外极线
        
        Returns
        -------
        line : np.ndarray (3,) - ax + by + c = 0
        """
        key = (cam_from, cam_to)
        if key in self._F_matrices:
            F = self._F_matrices[key]
        else:
            # 反向
            F = self._F_matrices.get((cam_to, cam_from))
            if F is not None:
                point_h = np.array([point[0], point[1], 1.0])
                line = F.T @ point_h
                return line
            raise ValueError(f"未计算相机对 {cam_from}-{cam_to} 的外极几何")
        
        point_h = np.array([point[0], point[1], 1.0])
        line = F @ point_h
        return line
    
    def point_to_epipolar_distance(self,
                                    point: Tuple[float, float],
                                    line: np.ndarray) -> float:
        """计算点到外极线的距离"""
        a, b, c = line
        return abs(a * point[0] + b * point[1] + c) / (np.sqrt(a**2 + b**2) + 1e-15)
    
    def match_particles_across_views(
            self,
            particles_2d: Dict[str, List[Particle2D]],
            calibrator
    ) -> List[List[Tuple[str, Particle2D]]]:
        """
        跨视角粒子匹配
        
        使用外极线约束 + 光束交叉法将不同相机中的同一粒子关联起来。
        
        Parameters
        ----------
        particles_2d : Dict[str, List[Particle2D]]
            {相机ID: 粒子列表}
        calibrator : MultiCameraCalibrator
            
        Returns
        -------
        matched_groups : List[List[Tuple[str, Particle2D]]]
            匹配的粒子组，每组包含多个相机的观测
            e.g., [('cam1', Particle2D), ('cam2', Particle2D), ('cam3', Particle2D)]
        """
        if not self._F_matrices:
            self.compute_epipolar_geometry(calibrator)
        
        cam_ids = list(particles_2d.keys())
        
        # Step 1: 两两匹配
        pairwise_matches = {}
        for cam1_id, cam2_id in combinations(cam_ids, 2):
            matches = self._match_pair(
                cam1_id, particles_2d[cam1_id],
                cam2_id, particles_2d[cam2_id]
            )
            pairwise_matches[(cam1_id, cam2_id)] = matches
        
        # Step 2: 多视角一致性传播
        # 从第一个相机出发，通过传递闭包构建多视角匹配组
        matched_groups = self._propagate_matches(
            cam_ids, particles_2d, pairwise_matches, calibrator
        )
        
        # Step 3: RANSAC剔除粗差
        matched_groups = self._ransac_filter(matched_groups, calibrator)
        
        logger.info(f"跨视角匹配完成: {len(matched_groups)} 个粒子组")
        for i, group in enumerate(matched_groups[:5]):
            views = [f"{cid}" for cid, _ in group]
            logger.info(f"  粒子 {i+1}: {len(group)} 视角 {views}")
        if len(matched_groups) > 5:
            logger.info(f"  ... (共 {len(matched_groups)} 个)")
        
        return matched_groups
    
    def _match_pair(self,
                    cam1_id: str,
                    particles1: List[Particle2D],
                    cam2_id: str,
                    particles2: List[Particle2D]) -> List[Tuple[int, int]]:
        """
        两两相机之间的粒子匹配
        
        Returns
        -------
        matches : List[Tuple[idx1, idx2]]
        """
        # 获取外极线容差
        threshold = self.config.epipolar_threshold
        
        matches = []
        
        for i, p1 in enumerate(particles1):
            # 计算p1在cam2上的外极线
            try:
                line = self.get_epipolar_line(p1.pixel, cam1_id, cam2_id)
            except ValueError:
                continue
            
            # 找cam2中外极线附近的粒子
            candidates = []
            for j, p2 in enumerate(particles2):
                dist = self.point_to_epipolar_distance(p2.pixel, line)
                if dist < threshold:
                    # 额外的强度相似性约束（可选）
                    intensity_diff = abs(p1.intensity - p2.intensity) / \
                                     (max(p1.intensity, p2.intensity) + 1e-10)
                    candidates.append((j, dist, intensity_diff))
            
            # 选择最近的候选
            if candidates:
                candidates.sort(key=lambda x: x[1])  # 按外极线距离排序
                matches.append((i, candidates[0][0]))
        
        return matches
    
    def _propagate_matches(self,
                           cam_ids: List[str],
                           particles_2d: Dict[str, List[Particle2D]],
                           pairwise_matches: Dict,
                           calibrator) -> List[List[Tuple[str, Particle2D]]]:
        """多视角匹配传播"""
        if len(cam_ids) < 3:
            # 只有两个相机，直接使用两两匹配
            groups = []
            for cam1_id, cam2_id in combinations(cam_ids, 2):
                key = (cam1_id, cam2_id)
                if key in pairwise_matches:
                    for i, j in pairwise_matches[key]:
                        groups.append([
                            (cam1_id, particles_2d[cam1_id][i]),
                            (cam2_id, particles_2d[particles_2d.keys().__iter__().__next__()][j])
                                if cam2_id in particles_2d else None
                        ])
            return groups
        
        # 多视角传播：从cam1出发逐步扩展
        groups = []
        used_cam2 = set()
        
        cam1_id = cam_ids[0]
        for cam2_id in cam_ids[1:]:
            key = (cam1_id, cam2_id)
            if key not in pairwise_matches:
                continue
            
            for i, j in pairwise_matches[key]:
                # 初始组：cam1 和 cam2 的匹配
                group = [
                    (cam1_id, particles_2d[cam1_id][i]),
                    (cam2_id, particles_2d[cam2_id][j])
                ]
                
                # 尝试添加更多视角
                for cam_k in cam_ids[2:]:
                    if cam_k == cam2_id:
                        continue
                    
                    # 检查cam_k中是否有匹配
                    # 三角测量得到3D位置
                    P1 = calibrator.compute_projection_matrix(cam1_id)
                    P2 = calibrator.compute_projection_matrix(cam2_id)
                    pt3d = self.triangulate(
                        P1, particles_2d[cam1_id][i].pixel,
                        P2, particles_2d[cam2_id][j].pixel
                    )
                    
                    if pt3d is None:
                        continue
                    
                    # 投影到cam_k，找最近粒子
                    Pk = calibrator.compute_projection_matrix(cam_k)
                    K_k = np.array(calibrator.camera_params[cam_k].camera_matrix)
                    proj_pt, _ = cv2.projectPoints(
                        pt3d.reshape(1, 1, 3).astype(np.float64),
                        np.zeros(3), np.zeros(3), K_k, np.zeros(5)
                    )
                    proj_u, proj_v = proj_pt[0, 0, 0], proj_pt[0, 0, 1]
                    
                    # 找cam_k中最近的粒子
                    best_dist = float('inf')
                    best_idx = -1
                    for k_idx, pk in enumerate(particles_2d[cam_k]):
                        d = np.sqrt((pk.pixel[0] - proj_u)**2 +
                                    (pk.pixel[1] - proj_v)**2)
                        if d < best_dist and d < threshold * 2:
                            best_dist = d
                            best_idx = k_idx
                    
                    threshold = self.config.epipolar_threshold
                    if best_idx >= 0:
                        group.append((cam_k, particles_2d[cam_k][best_idx]))
                
                groups.append(group)
        
        return groups
    
    def triangulate(self,
                    P1: np.ndarray,
                    pt1: Tuple[float, float],
                    P2: np.ndarray,
                    pt2: Tuple[float, float]) -> Optional[np.ndarray]:
        """
        三角测量 - 从两个视角的2D点计算3D坐标
        
        使用DLT (Direct Linear Transform) 方法
        
        Parameters
        ----------
        P1, P2 : np.ndarray (3x4)
            投影矩阵
        pt1, pt2 : Tuple[float, float]
            像素坐标 (u, v)
            
        Returns
        -------
        point_3d : np.ndarray (3,) or None
        """
        if self.config.triangulation_method == 'lstsq':
            return self._triangulate_lstsq(P1, pt1, P2, pt2)
        else:
            return self._triangulate_optimal(P1, pt1, P2, pt2)
    
    def _triangulate_lstsq(self, P1, pt1, P2, pt2):
        """SVD线性三角测量"""
        # 构造 A 矩阵: A * X = 0
        # u*P[2] - P[0] = 0, v*P[2] - P[1] = 0 (对每个视角)
        A = np.array([
            pt1[0] * P1[2] - P1[0],
            pt1[1] * P1[2] - P1[1],
            pt2[0] * P2[2] - P2[0],
            pt2[1] * P2[2] - P2[1],
        ], dtype=np.float64)
        
        # SVD求解
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        
        if abs(X[3]) < 1e-10:
            return None
        
        X = X / X[3]
        return X[:3]
    
    def _triangulate_optimal(self, P1, pt1, P2, pt2):
        """Hartley最优三角测量（考虑图像噪声协方差）"""
        # 先用线性方法得到初始解
        X_init = self._triangulate_lstsq(P1, pt1, P2, pt2)
        if X_init is None:
            return None
        
        # 优化重投影误差
        def residuals(x):
            X_h = np.array([x[0], x[1], x[2], 1.0])
            proj1 = P1 @ X_h
            proj2 = P2 @ X_h
            
            if abs(proj1[2]) < 1e-10 or abs(proj2[2]) < 1e-10:
                return [1e6, 1e6, 1e6, 1e6]
            
            u1, v1 = proj1[0] / proj1[2], proj1[1] / proj1[2]
            u2, v2 = proj2[0] / proj2[2], proj2[1] / proj2[2]
            
            return [u1 - pt1[0], v1 - pt1[1], u2 - pt2[0], v2 - pt2[1]]
        
        try:
            result = least_squares(residuals, X_init, method='lm')
            return result.x
        except Exception:
            return X_init
    
    def _ransac_filter(self,
                       matched_groups: List[List[Tuple[str, Particle2D]]],
                       calibrator) -> List[List[Tuple[str, Particle2D]]]:
        """RANSAC粗差剔除"""
        if len(matched_groups) == 0:
            return matched_groups
        
        valid_groups = []
        threshold = self.config.ransac_reproj_threshold
        
        for group in matched_groups:
            if len(group) < 2:
                continue
            
            # 取前两个视角进行三角测量
            cam1_id, p1 = group[0]
            cam2_id, p2 = group[1]
            
            P1 = calibrator.compute_projection_matrix(cam1_id)
            P2 = calibrator.compute_projection_matrix(cam2_id)
            
            pt3d = self.triangulate(P1, p1.pixel, P2, p2.pixel)
            if pt3d is None:
                continue
            
            # 检查3D点距离是否合理
            dist = np.linalg.norm(pt3d)
            if dist > self.config.max_particle_distance:
                continue
            
            # 验证所有视角的重投影误差
            max_error = 0.0
            valid = True
            error_dict = {}
            
            for cam_id, particle in group:
                P = calibrator.compute_projection_matrix(cam_id)
                K = np.array(calibrator.camera_params[cam_id].camera_matrix)
                
                X_h = np.array([pt3d[0], pt3d[1], pt3d[2], 1.0])
                proj = P @ X_h
                
                if abs(proj[2]) < 1e-10:
                    valid = False
                    break
                
                u_proj = proj[0] / proj[2]
                v_proj = proj[1] / proj[2]
                
                error = np.sqrt((u_proj - particle.pixel[0])**2 +
                                (v_proj - particle.pixel[1])**2)
                error_dict[cam_id] = float(error)
                
                if error > threshold:
                    valid = False
                    break
                
                max_error = max(max_error, error)
            
            if valid:
                valid_groups.append(group)
        
        return valid_groups
    
    def reconstruct_particles(
            self,
            images: Dict[str, np.ndarray],
            calibrator,
            reference_images: Optional[Dict[str, np.ndarray]] = None
    ) -> List[Particle3D]:
        """
        完整的示踪粒子三维重建流程
        
        Parameters
        ----------
        images : Dict[str, np.ndarray]
            {相机ID: 粒子图像}
        calibrator : MultiCameraCalibrator
            已标定的多相机标定器
        reference_images : Dict[str, np.ndarray], optional
            {相机ID: 背景参考图}
            
        Returns
        -------
        particles_3d : List[Particle3D]
        """
        # Step 1: 粒子检测
        logger.info("Step 1: 多视角粒子检测...")
        particles_2d = {}
        camera_params_undist = {}
        
        for cam_id, image in images.items():
            cp = calibrator.camera_params[cam_id]
            camera_params_undist[cam_id] = {
                'camera_matrix': cp.camera_matrix,
                'dist_coeffs': cp.dist_coeffs
            }
            
            detected = self.detector.detect_particles(
                image, cam_id, camera_params_undist[cam_id]
            )
            particles_2d[cam_id] = detected
        
        # Step 2: 跨视角匹配
        logger.info("Step 2: 跨视角粒子匹配...")
        matched_groups = self.match_particles_across_views(
            particles_2d, calibrator
        )
        
        # Step 3: 三角测量
        logger.info("Step 3: 三角测量...")
        particles_3d = []
        
        for group in matched_groups:
            if len(group) < 2:
                continue
            
            # 使用所有可用视角进行多视角三角测量
            pt3d = self._multiview_triangulate(group, calibrator)
            
            if pt3d is not None:
                # 计算重投影误差
                error_dict = {}
                total_error = 0.0
                
                for cam_id, particle in group:
                    P = calibrator.compute_projection_matrix(cam_id)
                    X_h = np.array([pt3d[0], pt3d[1], pt3d[2], 1.0])
                    proj = P @ X_h
                    
                    if abs(proj[2]) > 1e-10:
                        u_proj = proj[0] / proj[2]
                        v_proj = proj[1] / proj[2]
                        err = np.sqrt((u_proj - particle.pixel[0])**2 +
                                      (v_proj - particle.pixel[1])**2)
                        error_dict[cam_id] = float(err)
                        total_error += err
                
                avg_error = total_error / len(group)
                quality = 1.0 / (1.0 + avg_error)
                
                particles_3d.append(Particle3D(
                    position=pt3d,
                    n_views=len(group),
                    reprojection_errors=error_dict,
                    quality=quality
                ))
        
        logger.info(f"三维重建完成: {len(particles_3d)} 个粒子")
        return particles_3d
    
    def _multiview_triangulate(self,
                                group: List[Tuple[str, Particle2D]],
                                calibrator) -> Optional[np.ndarray]:
        """
        多视角三角测量（超过2个视角时使用所有视角）
        """
        if len(group) < 2:
            return None
        
        # 两两视角分别三角测量
        points_3d = []
        
        pairs = list(combinations(range(len(group)), 2))
        
        for idx1, idx2 in pairs:
            cam1_id, p1 = group[idx1]
            cam2_id, p2 = group[idx2]
            
            P1 = calibrator.compute_projection_matrix(cam1_id)
            P2 = calibrator.compute_projection_matrix(cam2_id)
            
            pt = self.triangulate(P1, p1.pixel, P2, p2.pixel)
            if pt is not None:
                points_3d.append(pt)
        
        if len(points_3d) == 0:
            return None
        
        # 取中位数作为最终3D位置（鲁棒估计）
        points_3d = np.array(points_3d)
        median_point = np.median(points_3d, axis=0)
        
        return median_point
