"""
气泡图像预处理与投影计算模块

处理流程:
1. 背景去除（帧差法或参考图法）
2. 去畸变（使用标定参数）
3. 二值化/阈值分割
4. 投影图像计算（透射投影或消光投影）
5. 图像增强
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class BubbleImageProcessor:
    """
    气泡图像预处理器
    
    支持多种背景去除方法和投影计算方式
    """

    def __init__(self,
                 background_method: str = 'reference',
                 threshold_method: str = 'otsu',
                 morph_operations: bool = True):
        """
        Parameters
        ----------
        background_method : str
            背景去除方法:
            - 'reference': 使用无气泡参考图
            - 'median': 中值滤波背景估计
            - 'mog': 混合高斯模型（视频序列）
        threshold_method : str
            阈值分割方法:
            - 'otsu': Otsu自适应阈值
            - 'adaptive': 局部自适应阈值
            - 'manual': 手动指定阈值
            - 'li': Li's迭代最小交叉熵
        morph_operations : bool
            是否执行形态学操作（去噪）
        """
        self.background_method = background_method
        self.threshold_method = threshold_method
        self.morph_operations = morph_operations
        
        # 背景图像缓存
        self._backgrounds: Dict[str, np.ndarray] = {}

    def remove_background(self,
                          image: np.ndarray,
                          camera_id: str,
                          reference_image: Optional[np.ndarray] = None,
                          kernel_size: int = 15) -> np.ndarray:
        """
        去除背景，提取气泡前景
        
        Parameters
        ----------
        image : np.ndarray
            输入气泡图像
        camera_id : str
            相机ID（用于缓存背景）
        reference_image : np.ndarray, optional
            无气泡参考图像
        kernel_size : int
            中值滤波核大小
            
        Returns
        -------
        foreground : np.ndarray
            前景图像（气泡区域）
        """
        gray = self._to_gray(image)
        
        if self.background_method == 'reference':
            if reference_image is not None:
                bg = self._to_gray(reference_image)
            elif camera_id in self._backgrounds:
                bg = self._backgrounds[camera_id]
            else:
                logger.warning(f"相机 {camera_id}: 未提供参考图像，"
                               "使用中值滤波估计背景")
                bg = cv2.medianBlur(gray, kernel_size)
            
            if gray.shape != bg.shape:
                bg = cv2.resize(bg, (gray.shape[1], gray.shape[0]))
            
            foreground = cv2.absdiff(gray, bg)
            
        elif self.background_method == 'median':
            bg = cv2.medianBlur(gray, kernel_size)
            foreground = cv2.absdiff(gray, bg)
            
        elif self.background_method == 'mog':
            # 单帧退化为中值滤波
            bg = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
            foreground = cv2.absdiff(gray, bg)
        else:
            foreground = gray.copy()
        
        return foreground

    def segment_bubbles(self,
                        foreground: np.ndarray,
                        manual_threshold: Optional[float] = None,
                        blur_ksize: int = 5) -> np.ndarray:
        """
        气泡二值化分割
        
        Parameters
        ----------
        foreground : np.ndarray
            前景图像
        manual_threshold : float, optional
            手动阈值 (0-255)
        blur_ksize : int
            预处理模糊核大小
            
        Returns
        -------
        binary : np.ndarray
            二值化结果 (0=背景, 255=气泡)
        """
        # 预处理: 高斯模糊降噪
        blurred = cv2.GaussianBlur(foreground, (blur_ksize, blur_ksize), 0)
        
        if self.threshold_method == 'otsu':
            _, binary = cv2.threshold(blurred, 0, 255,
                                       cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
        elif self.threshold_method == 'adaptive':
            binary = cv2.adaptiveThreshold(
                blurred, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
            
        elif self.threshold_method == 'li':
            threshold = self._li_threshold(blurred)
            _, binary = cv2.threshold(blurred, threshold, 255,
                                       cv2.THRESH_BINARY)
            
        elif self.threshold_method == 'manual':
            if manual_threshold is None:
                raise ValueError("手动阈值模式需要提供 manual_threshold")
            _, binary = cv2.threshold(blurred, manual_threshold, 255,
                                       cv2.THRESH_BINARY)
        else:
            _, binary = cv2.threshold(blurred, 0, 255,
                                       cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 形态学操作去噪
        if self.morph_operations:
            binary = self._morphological_cleanup(binary)
        
        return binary

    def compute_projection(self,
                           binary_image: np.ndarray,
                           projection_type: str = 'silhouette') -> np.ndarray:
        """
        计算投影图像（用于MART重建）
        
        Parameters
        ----------
        binary_image : np.ndarray
            气泡二值图像
        projection_type : str
            投影类型:
            - 'silhouette': 阴影投影（0/1，气泡遮挡区域）
            - 'soft_edge': 柔化边缘的投影
            - 'distance': 距离变换投影（近似厚度信息）
            
        Returns
        -------
        projection : np.ndarray (float64)
            归一化投影图像 [0, 1]
        """
        if projection_type == 'silhouette':
            # 直接二值投影
            proj = binary_image.astype(np.float64) / 255.0
            
        elif projection_type == 'soft_edge':
            # 使用高斯模糊柔化边缘
            soft = cv2.GaussianBlur(binary_image.astype(np.float64), (5, 5), 2)
            proj = soft / 255.0
            
        elif projection_type == 'distance':
            # 距离变换作为投影（包含厚度信息）
            dist = cv2.distanceTransform(binary_image, cv2.DIST_L2, 5)
            max_dist = dist.max() + 1e-10
            proj = dist / max_dist
        else:
            proj = binary_image.astype(np.float64) / 255.0
        
        return proj

    def undistort_image(self,
                        image: np.ndarray,
                        camera_matrix: np.ndarray,
                        dist_coeffs: np.ndarray) -> np.ndarray:
        """
        使用标定参数对图像去畸变
        """
        h, w = image.shape[:2]
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            camera_matrix, dist_coeffs, (w, h), 1, (w, h)
        )
        undistorted = cv2.undistort(image, camera_matrix, dist_coeffs,
                                     None, new_camera_matrix)
        return undistorted

    def prepare_projection_data(self,
                                 camera_images: Dict[str, np.ndarray],
                                 camera_params: Dict[str, dict],
                                 reference_images: Optional[Dict[str, np.ndarray]] = None,
                                 projection_type: str = 'soft_edge',
                                 manual_threshold: Optional[float] = None
                                 ) -> Dict[str, np.ndarray]:
        """
        完整的图像预处理流程：去背景 → 分割 → 计算投影
        
        Parameters
        ----------
        camera_images : Dict[str, np.ndarray]
            {相机ID: 气泡图像}
        camera_params : Dict[str, dict]
            {相机ID: {'camera_matrix': ..., 'dist_coeffs': ...}}
        reference_images : Dict[str, np.ndarray], optional
            {相机ID: 背景参考图像}
        projection_type : str
            投影计算方式
        manual_threshold : float, optional
            手动分割阈值
            
        Returns
        -------
        projections : Dict[str, np.ndarray]
            {相机ID: 投影图像}
        """
        projections = {}
        
        for cam_id, image in camera_images.items():
            logger.info(f"预处理相机 {cam_id} 的图像...")
            
            # 1. 去畸变
            if cam_id in camera_params:
                cp = camera_params[cam_id]
                K = np.array(cp['camera_matrix'])
                D = np.array(cp['dist_coeffs'])
                image = self.undistort_image(image, K, D)
            
            # 2. 去背景
            ref_img = None
            if reference_images and cam_id in reference_images:
                ref_img = reference_images[cam_id]
            
            foreground = self.remove_background(image, cam_id, ref_img)
            
            # 3. 分割
            binary = self.segment_bubbles(foreground, manual_threshold)
            
            # 4. 计算投影
            proj = self.compute_projection(binary, projection_type)
            
            projections[cam_id] = proj
            
            # 保存中间结果用于可视化
            logger.info(f"  相机 {cam_id}: 气泡像素占比 "
                        f"{np.mean(binary > 0) * 100:.1f}%")
        
        return projections

    def set_background(self, camera_id: str, background: np.ndarray):
        """设置相机背景图像"""
        self._backgrounds[camera_id] = self._to_gray(background)

    # ---- 内部方法 ----

    def _to_gray(self, image: np.ndarray) -> np.ndarray:
        """转灰度"""
        if len(image.shape) == 2:
            return image.copy()
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def _morphological_cleanup(self, binary: np.ndarray) -> np.ndarray:
        """形态学去噪"""
        # 开运算（去除小噪点）
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # 闭运算（填充小孔洞）
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        
        return cleaned

    def _li_threshold(self, image: np.ndarray) -> float:
        """
        Li's Minimum Cross Entropy 阈值
        """
        img = image.flatten().astype(np.float64)
        mean = np.mean(img)
        
        threshold = mean
        for _ in range(100):
            fg = img[img > threshold]
            bg = img[img <= threshold]
            
            if len(fg) == 0 or len(bg) == 0:
                break
            
            new_threshold = (np.mean(fg) + np.mean(bg)) / 2
            if abs(new_threshold - threshold) < 0.01:
                break
            threshold = new_threshold
        
        return threshold
