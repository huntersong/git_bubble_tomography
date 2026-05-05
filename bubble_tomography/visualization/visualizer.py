"""
三维可视化与点云输出模块

功能:
1. 3D点云可视化（matplotlib交互式）
2. 体素切片视图
3. 点云导出（PLY, PCD, OBJ格式）
4. 投影对比视图
5. 重建过程动画
6. 三维速度场可视化（矢量场、流线、切面）
7. 示踪粒子轨迹可视化
"""

import numpy as np
import cv2
import os
from typing import Dict, Optional, Tuple, List
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
from matplotlib import font_manager
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import logging

logger = logging.getLogger(__name__)


class ResultVisualizer:
    """
    三维重建结果可视化与导出工具
    """

    def __init__(self, output_dir: str = 'results'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self._font_name = self._pick_font([
            'Microsoft YaHei',
            'SimHei',
            'Noto Sans CJK SC',
            'Source Han Sans SC',
            'WenQuanYi Zen Hei',
            'DejaVu Sans',
        ])

        # 设置中文字体，并让文本说明框也使用可显示中文的字体
        plt.rcParams['font.sans-serif'] = [self._font_name]
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.monospace'] = [self._font_name]
        plt.rcParams['axes.unicode_minus'] = False

    @staticmethod
    def _pick_font(candidates: List[str]) -> str:
        available = {font.name for font in font_manager.fontManager.ttflist}
        for name in candidates:
            if name in available:
                return name
        return 'DejaVu Sans'

    def plot_point_cloud(self,
                         points: np.ndarray,
                         normals: Optional[np.ndarray] = None,
                         title: str = '气泡三维点云',
                         colors: Optional[np.ndarray] = None,
                         point_size: float = 1.0,
                         save_path: Optional[str] = None):
        """
        3D点云可视化
        
        Parameters
        ----------
        points : np.ndarray (N, 3)
            点云坐标 (mm)
        normals : np.ndarray (N, 3), optional
            法向量
        title : str
            标题
        colors : np.ndarray (N,), optional
            点颜色值
        point_size : float
            点大小
        save_path : str, optional
            保存路径
        """
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        if colors is None:
            colors = points[:, 2]  # 用Z坐标着色
        
        scatter = ax.scatter(
            points[:, 0], points[:, 1], points[:, 2],
            c=colors, cmap='coolwarm',
            s=point_size, alpha=0.6, edgecolors='none'
        )
        
        if normals is not None and len(normals) == len(points):
            # 绘制法向量（采样以避免过密）
            step = max(1, len(points) // 200)
            scale = 0.3
            ax.quiver(
                points[::step, 0], points[::step, 1], points[::step, 2],
                normals[::step, 0], normals[::step, 1], normals[::step, 2],
                length=scale, color='green', alpha=0.3, linewidth=0.5
            )
        
        ax.set_xlabel('X (mm)', fontsize=12)
        ax.set_ylabel('Y (mm)', fontsize=12)
        ax.set_zlabel('Z (mm)', fontsize=12)
        ax.set_title(title, fontsize=14)
        
        # 等比例坐标轴
        max_range = np.array([
            points[:, 0].max() - points[:, 0].min(),
            points[:, 1].max() - points[:, 1].min(),
            points[:, 2].max() - points[:, 2].min()
        ]).max() / 2
        
        mid = points.mean(axis=0)
        ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
        ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
        ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
        
        plt.colorbar(scatter, ax=ax, shrink=0.5, label='Z (mm)')
        plt.tight_layout()
        
        path = save_path or os.path.join(self.output_dir, 'point_cloud_3d.png')
        plt.savefig(path, dpi=200, bbox_inches='tight')
        plt.close()
        
        logger.info(f"点云图已保存: {path}")
        return path

    def plot_volume_slices(self,
                           volume: np.ndarray,
                           axis: str = 'z',
                           num_slices: int = 5,
                           threshold: float = 0.1,
                           title: str = '体素切片视图',
                           save_path: Optional[str] = None):
        """
        体素场切片视图
        
        Parameters
        ----------
        volume : np.ndarray (Nx, Ny, Nz)
            三维体素场
        axis : str
            切片方向 'x', 'y', 'z'
        num_slices : int
            显示的切片数
        threshold : float
            显示阈值
        """
        axis_map = {'x': 0, 'y': 1, 'z': 2}
        ax_idx = axis_map.get(axis, 2)
        n_slices = volume.shape[ax_idx]
        
        indices = np.linspace(0, n_slices - 1, num_slices, dtype=int)
        
        fig, axes = plt.subplots(1, num_slices, figsize=(4 * num_slices, 4))
        if num_slices == 1:
            axes = [axes]
        
        for i, idx in enumerate(indices):
            if ax_idx == 0:
                slice_data = volume[idx, :, :]
            elif ax_idx == 1:
                slice_data = volume[:, idx, :]
            else:
                slice_data = volume[:, :, idx]
            
            im = axes[i].imshow(slice_data.T, cmap='hot',
                                vmin=threshold, vmax=volume.max(),
                                origin='lower')
            axes[i].set_title(f'{axis.upper()}={idx}', fontsize=10)
            plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
        
        fig.suptitle(title, fontsize=14)
        plt.tight_layout()
        
        path = save_path or os.path.join(self.output_dir, f'volume_slices_{axis}.png')
        plt.savefig(path, dpi=200, bbox_inches='tight')
        plt.close()
        
        logger.info(f"切片图已保存: {path}")
        return path

    def plot_projection_comparison(self,
                                   projections: Dict[str, np.ndarray],
                                   title: str = '多视角投影对比',
                                   save_path: Optional[str] = None):
        """
        多相机投影图像对比视图
        
        Parameters
        ----------
        projections : Dict[str, np.ndarray]
            {相机ID: 投影图像}
        """
        n_cams = len(projections)
        cols = min(n_cams, 4)
        rows = (n_cams + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        axes = axes.flatten()
        
        for i, (cam_id, proj) in enumerate(projections.items()):
            axes[i].imshow(proj, cmap='gray', origin='lower')
            axes[i].set_title(f'Camera {cam_id}', fontsize=12)
        
        # 隐藏多余的子图
        for j in range(n_cams, len(axes)):
            axes[j].axis('off')
        
        fig.suptitle(title, fontsize=14)
        plt.tight_layout()
        
        path = save_path or os.path.join(self.output_dir, 'projection_comparison.png')
        plt.savefig(path, dpi=200, bbox_inches='tight')
        plt.close()
        
        logger.info(f"投影对比图已保存: {path}")
        return path

    def plot_convergence(self,
                         errors: List[float],
                         title: str = 'MART收敛曲线',
                         save_path: Optional[str] = None):
        """绘制迭代收敛曲线"""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(range(1, len(errors) + 1), errors, 'b-o', markersize=4)
        ax.set_xlabel('迭代次数', fontsize=12)
        ax.set_ylabel('投影误差 (RMS)', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        plt.tight_layout()
        
        path = save_path or os.path.join(self.output_dir, 'convergence.png')
        plt.savefig(path, dpi=200, bbox_inches='tight')
        plt.close()
        
        logger.info(f"收敛曲线已保存: {path}")
        return path

    def plot_reprojection(self,
                          camera_params: Dict[str, dict],
                          image_points: Dict[str, np.ndarray],
                          object_points: np.ndarray,
                          save_path: Optional[str] = None):
        """
        绘制重投影结果（标定验证）
        """
        fig, axes = plt.subplots(1, len(camera_params),
                                  figsize=(5 * len(camera_params), 5))
        if len(camera_params) == 1:
            axes = [axes]
        
        for i, (cam_id, cp) in enumerate(camera_params.items()):
            K = np.array(cp['camera_matrix'])
            D = np.array(cp['dist_coeffs'])
            rvec = np.array(cp['rvec'])
            tvec = np.array(cp['tvec'])
            
            # 重投影
            projected, _ = cv2.projectPoints(
                object_points, rvec, tvec, K, D
            )
            projected = projected.reshape(-1, 2)
            
            img_pts = image_points[cam_id].reshape(-1, 2)
            
            ax = axes[i]
            ax.scatter(img_pts[:, 0], img_pts[:, 1],
                       c='blue', s=20, label='检测点', alpha=0.6)
            ax.scatter(projected[:, 0], projected[:, 1],
                       c='red', s=10, label='重投影', alpha=0.8)
            ax.invert_yaxis()
            ax.legend(fontsize=8)
            ax.set_title(f'Camera {cam_id}', fontsize=12)
            ax.set_aspect('equal')
        
        plt.tight_layout()
        
        path = save_path or os.path.join(self.output_dir, 'reprojection.png')
        plt.savefig(path, dpi=200, bbox_inches='tight')
        plt.close()
        
        return path

    def save_point_cloud_ply(self,
                              points: np.ndarray,
                              normals: Optional[np.ndarray] = None,
                              filename: str = 'bubble_point_cloud.ply'):
        """
        导出点云为PLY格式
        
        Parameters
        ----------
        points : np.ndarray (N, 3)
        normals : np.ndarray (N, 3), optional
        filename : str
        """
        path = os.path.join(self.output_dir, filename)
        n = len(points)
        
        has_normals = normals is not None and len(normals) == n
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {n}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            if has_normals:
                f.write("property float nx\n")
                f.write("property float ny\n")
                f.write("property float nz\n")
            f.write("end_header\n")
            
            for i in range(n):
                f.write(f"{points[i, 0]:.6f} {points[i, 1]:.6f} {points[i, 2]:.6f}")
                if has_normals:
                    f.write(f" {normals[i, 0]:.6f} {normals[i, 1]:.6f} {normals[i, 2]:.6f}")
                f.write("\n")
        
        logger.info(f"PLY点云已保存: {path} ({n} 个点)")
        return path

    def save_point_cloud_obj(self,
                              points: np.ndarray,
                              normals: Optional[np.ndarray] = None,
                              filename: str = 'bubble_point_cloud.obj'):
        """导出点云为OBJ格式"""
        path = os.path.join(self.output_dir, filename)
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write("# Bubble Point Cloud\n")
            f.write(f"# Vertices: {len(points)}\n\n")
            
            for i, pt in enumerate(points):
                f.write(f"v {pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f}\n")
            
            if normals is not None:
                for n in normals:
                    f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
        
        logger.info(f"OBJ点云已保存: {path}")
        return path

    def save_point_cloud_pcd(self,
                              points: np.ndarray,
                              filename: str = 'bubble_point_cloud.pcd'):
        """导出点云为PCD格式（兼容CloudCompare/MeshLab）"""
        path = os.path.join(self.output_dir, filename)
        n = len(points)
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write("# .PCD v0.7 - Bubble Point Cloud\n")
            f.write("VERSION 0.7\n")
            f.write("FIELDS x y z\n")
            f.write("SIZE 4 4 4\n")
            f.write("TYPE F F F\n")
            f.write("COUNT 1 1 1\n")
            f.write(f"WIDTH {n}\n")
            f.write("HEIGHT 1\n")
            f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
            f.write(f"POINTS {n}\n")
            f.write("DATA ascii\n")
            
            for pt in points:
                f.write(f"{pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f}\n")
        
        logger.info(f"PCD点云已保存: {path}")
        return path

    def save_volume_npy(self, volume: np.ndarray,
                        filename: str = 'volume_data.npy'):
        """保存体素数据为NPY格式"""
        path = os.path.join(self.output_dir, filename)
        np.save(path, volume)
        logger.info(f"体素数据已保存: {path}")
        return path

    # ========== 速度场可视化 ==========

    def plot_velocity_quiver(self,
                              velocity_field: np.ndarray,
                              grid_positions: np.ndarray,
                              slice_axis: str = 'z',
                              slice_index: Optional[int] = None,
                              title: str = '3D Velocity Field',
                              save_path: Optional[str] = None):
        """
        速度场矢量图（quiver plot）
        
        Parameters
        ----------
        velocity_field : np.ndarray (Nx, Ny, Nz, 3)
        grid_positions : np.ndarray (Nx, Ny, Nz, 3)
        slice_axis : str
            切面方向 'x', 'y', 'z'
        slice_index : int, optional
            切面索引，默认为中间切面
        """
        nx, ny, nz, _ = velocity_field.shape
        axis_map = {'x': 0, 'y': 1, 'z': 2}
        ax_idx = axis_map.get(slice_axis, 2)
        
        if slice_index is None:
            slice_index = velocity_field.shape[ax_idx] // 2
        
        # 提取切面数据
        slices = [slice(None)] * 3
        slices[ax_idx] = slice_index
        
        vel_slice = velocity_field[tuple(slices)]  # (a, b, 3)
        pos_slice = grid_positions[tuple(slices)]  # (a, b, 3)
        
        # 计算速度幅值
        speed = np.linalg.norm(vel_slice, axis=-1)
        
        fig = plt.figure(figsize=(14, 10))
        
        # 3D quiver
        ax3d = fig.add_subplot(121, projection='3d')
        
        # 下采样避免过密
        a, b = vel_slice.shape[:2]
        step = max(1, min(a, b) // 16)
        
        # 根据切面方向选择绘图坐标
        if ax_idx == 0:  # X切面
            X = pos_slice[:, :, 1]
            Y = pos_slice[:, :, 2]
            U = vel_slice[:, :, 1]
            V = vel_slice[:, :, 2]
            W = vel_slice[:, :, 0]
            Z = pos_slice[:, :, 0]
        elif ax_idx == 1:  # Y切面
            X = pos_slice[:, :, 0]
            Y = pos_slice[:, :, 2]
            U = vel_slice[:, :, 0]
            V = vel_slice[:, :, 2]
            W = vel_slice[:, :, 1]
            Z = pos_slice[:, :, 1]
        else:  # Z切面
            X = pos_slice[:, :, 0]
            Y = pos_slice[:, :, 1]
            U = vel_slice[:, :, 0]
            V = vel_slice[:, :, 1]
            W = vel_slice[:, :, 2]
            Z = pos_slice[:, :, 2]
        
        Xs = X[::step, ::step]
        Ys = Y[::step, ::step]
        Us = U[::step, ::step]
        Vs = V[::step, ::step]
        Zs = np.full_like(Xs, Z[0, 0] if isinstance(Z, np.ndarray) else Z)
        Ws = W[::step, ::step]
        
        speed_s = speed[::step, ::step]
        
        q = ax3d.quiver(Xs, Ys, Zs, Us, Vs, Ws,
                         length=0.5, normalize=False,
                         color='steelblue', alpha=0.7, linewidth=0.8)
        ax3d.set_xlabel('X (mm)')
        ax3d.set_ylabel('Y (mm)')
        ax3d.set_zlabel('Z (mm)')
        ax3d.set_title(f'Velocity Vectors ({slice_axis.upper()}={slice_index})')
        plt.colorbar(q, ax=ax3d, shrink=0.5, label='|V| (mm/s)')
        
        # 2D speed contour
        ax2d = fig.add_subplot(122)
        im = ax2d.contourf(X, Y, speed, levels=20, cmap='jet')
        ax2d.quiver(X[::step, ::step], Y[::step, ::step],
                     Us, Vs, color='white', alpha=0.7, scale=None)
        ax2d.set_xlabel(f'{"XYZ"[1 if ax_idx == 0 else 0]} (mm)')
        ax2d.set_ylabel(f'{"XYZ"[2 if ax_idx < 2 else 1]} (mm)')
        ax2d.set_title(f'Speed Contour ({slice_axis.upper()}={slice_index})')
        ax2d.set_aspect('equal')
        plt.colorbar(im, ax=ax2d, label='|V| (mm/s)')
        
        fig.suptitle(title, fontsize=14)
        plt.tight_layout()
        
        path = save_path or os.path.join(self.output_dir,
                                          f'velocity_quiver_{slice_axis}{slice_index}.png')
        plt.savefig(path, dpi=200, bbox_inches='tight')
        plt.close()
        
        logger.info(f"速度矢量图已保存: {path}")
        return path

    def plot_velocity_magnitude_slices(self,
                                        velocity_field: np.ndarray,
                                        grid_positions: np.ndarray,
                                        num_slices: int = 3,
                                        save_path: Optional[str] = None):
        """
        速度幅值多切面视图
        
        Parameters
        ----------
        velocity_field : np.ndarray (Nx, Ny, Nz, 3)
        grid_positions : np.ndarray (Nx, Ny, Nz, 3)
        """
        speed = np.linalg.norm(velocity_field, axis=-1)
        nx, ny, nz = speed.shape
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # X切面
        mid = nx // 2
        slices_idx = np.linspace(max(0, mid - 2), min(nx - 1, mid + 2),
                                  num_slices, dtype=int)
        im0 = axes[0].imshow(speed[slices_idx, :, :].mean(axis=0).T,
                              cmap='jet', origin='lower')
        axes[0].set_title('XY Plane (X avg)')
        plt.colorbar(im0, ax=axes[0], label='|V| (mm/s)')
        
        # Y切面
        mid = ny // 2
        slices_idx = np.linspace(max(0, mid - 2), min(ny - 1, mid + 2),
                                  num_slices, dtype=int)
        im1 = axes[1].imshow(speed[:, slices_idx, :].mean(axis=1).T,
                              cmap='jet', origin='lower')
        axes[1].set_title('XZ Plane (Y avg)')
        plt.colorbar(im1, ax=axes[1], label='|V| (mm/s)')
        
        # Z切面
        mid = nz // 2
        slices_idx = np.linspace(max(0, mid - 2), min(nz - 1, mid + 2),
                                  num_slices, dtype=int)
        im2 = axes[2].imshow(speed[:, :, slices_idx].mean(axis=2).T,
                              cmap='jet', origin='lower')
        axes[2].set_title('YZ Plane (Z avg)')
        plt.colorbar(im2, ax=axes[2], label='|V| (mm/s)')
        
        fig.suptitle('Velocity Magnitude - Orthogonal Slices', fontsize=14)
        plt.tight_layout()
        
        path = save_path or os.path.join(self.output_dir,
                                          'velocity_slices.png')
        plt.savefig(path, dpi=200, bbox_inches='tight')
        plt.close()
        
        logger.info(f"速度切面图已保存: {path}")
        return path

    def plot_particle_positions_3d(self,
                                    particles_3d: list,
                                    velocities: Optional[np.ndarray] = None,
                                    title: str = 'Particle 3D Positions',
                                    save_path: Optional[str] = None):
        """
        示踪粒子三维位置可视化
        
        Parameters
        ----------
        particles_3d : List[Particle3D]
        velocities : np.ndarray (N, 3), optional
            速度向量
        """
        if len(particles_3d) == 0:
            logger.warning("无粒子数据")
            return
        
        positions = np.array([p.position for p in particles_3d])
        quality = np.array([p.quality for p in particles_3d])
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(
            positions[:, 0], positions[:, 1], positions[:, 2],
            c=quality, cmap='hot', s=10, alpha=0.6,
            edgecolors='none'
        )
        
        # 绘制速度箭头
        if velocities is not None and len(velocities) == len(particles_3d):
            step = max(1, len(particles_3d) // 100)
            speed = np.linalg.norm(velocities, axis=-1)
            max_speed = speed.max() + 1e-10
            
            ax.quiver(
                positions[::step, 0],
                positions[::step, 1],
                positions[::step, 2],
                velocities[::step, 0],
                velocities[::step, 1],
                velocities[::step, 2],
                length=2.0 / max_speed,
                normalize=False,
                color='blue', alpha=0.4, linewidth=0.5
            )
        
        ax.set_xlabel('X (mm)', fontsize=12)
        ax.set_ylabel('Y (mm)', fontsize=12)
        ax.set_zlabel('Z (mm)', fontsize=12)
        ax.set_title(title, fontsize=14)
        plt.colorbar(scatter, ax=ax, shrink=0.5, label='Quality')
        
        # 等比例坐标
        max_range = np.array([
            positions[:, 0].max() - positions[:, 0].min(),
            positions[:, 1].max() - positions[:, 1].min(),
            positions[:, 2].max() - positions[:, 2].min()
        ]).max() / 2
        mid = positions.mean(axis=0)
        ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
        ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
        ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
        
        plt.tight_layout()
        
        path = save_path or os.path.join(self.output_dir,
                                          'particle_positions_3d.png')
        plt.savefig(path, dpi=200, bbox_inches='tight')
        plt.close()
        
        logger.info(f"粒子位置图已保存: {path}")
        return path

    def plot_velocity_statistics(self,
                                  velocity_field: np.ndarray,
                                  snr_field: np.ndarray = None,
                                  save_path: Optional[str] = None):
        """
        速度场统计图
        
        Parameters
        ----------
        velocity_field : np.ndarray (Nx, Ny, Nz, 3)
        snr_field : np.ndarray (Nx, Ny, Nz), optional
        """
        speed = np.linalg.norm(velocity_field, axis=-1).flatten()
        vx = velocity_field[:, :, :, 0].flatten()
        vy = velocity_field[:, :, :, 1].flatten()
        vz = velocity_field[:, :, :, 2].flatten()
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 速度幅值直方图
        axes[0, 0].hist(speed, bins=50, color='steelblue', edgecolor='white')
        axes[0, 0].set_xlabel('|V| (mm/s)')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('Speed Distribution')
        axes[0, 0].axvline(np.mean(speed), color='red', linestyle='--',
                            label=f'Mean: {np.mean(speed):.2f}')
        axes[0, 0].legend()
        
        # 各分量分布
        axes[0, 1].hist(vx, bins=50, alpha=0.5, label='Vx', color='red')
        axes[0, 1].hist(vy, bins=50, alpha=0.5, label='Vy', color='green')
        axes[0, 1].hist(vz, bins=50, alpha=0.5, label='Vz', color='blue')
        axes[0, 1].set_xlabel('Velocity (mm/s)')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('Velocity Component Distribution')
        axes[0, 1].legend()
        
        # 剪切应力（速度梯度）
        if velocity_field.shape[0] > 2:
            dvx_dy = np.gradient(velocity_field[:, :, :, 0], axis=1)
            dvy_dx = np.gradient(velocity_field[:, :, :, 1], axis=0)
            shear = (dvx_dy + dvy_dx) / 2
            axes[1, 0].hist(shear.flatten(), bins=50, color='purple',
                             edgecolor='white')
            axes[1, 0].set_xlabel('Shear Rate (1/s)')
            axes[1, 0].set_ylabel('Count')
            axes[1, 0].set_title('Shear Rate Distribution')
        
        # 湍流强度（如果有SNR）
        if snr_field is not None:
            axes[1, 1].hist(snr_field.flatten(), bins=50, color='orange',
                             edgecolor='white')
            axes[1, 1].set_xlabel('SNR')
            axes[1, 1].set_ylabel('Count')
            axes[1, 1].set_title('Signal-to-Noise Ratio')
            axes[1, 1].axvline(1.2, color='red', linestyle='--',
                                label='Threshold')
            axes[1, 1].legend()
        else:
            # 涡量分布
            if velocity_field.shape[0] > 2:
                dvz_dy = np.gradient(velocity_field[:, :, :, 2], axis=1)
                dvy_dz = np.gradient(velocity_field[:, :, :, 1], axis=2)
                vorticity_x = dvz_dy - dvy_dz
                axes[1, 1].hist(vorticity_x.flatten(), bins=50,
                                 color='darkred', edgecolor='white')
                axes[1, 1].set_xlabel('Vorticity (1/s)')
                axes[1, 1].set_ylabel('Count')
                axes[1, 1].set_title('Vorticity Distribution (X)')
        
        fig.suptitle('Velocity Field Statistics', fontsize=14)
        plt.tight_layout()
        
        path = save_path or os.path.join(self.output_dir,
                                          'velocity_statistics.png')
        plt.savefig(path, dpi=200, bbox_inches='tight')
        plt.close()
        
        logger.info(f"速度统计图已保存: {path}")
        return path

    def create_velocity_report(self,
                                velocity_result: dict,
                                particles_3d: list,
                                title: str = 'Tomographic PIV Report',
                                save_path: Optional[str] = None):
        """
        生成综合PIV报告图
        """
        vf = velocity_result['velocity_field']
        gp = velocity_result['grid_positions']
        snr = velocity_result['snr_field']
        
        speed = np.linalg.norm(vf, axis=-1)
        grid_shape = vf.shape[:3]
        
        fig = plt.figure(figsize=(18, 12))
        
        # 1. 粒子3D位置
        ax1 = fig.add_subplot(2, 3, 1, projection='3d')
        if len(particles_3d) > 0:
            pos = np.array([p.position for p in particles_3d])
            ax1.scatter(pos[:, 0], pos[:, 1], pos[:, 2],
                        c='blue', s=2, alpha=0.3)
        ax1.set_title('Particle Positions')
        ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
        
        # 2. 速度幅值中间切面
        ax2 = fig.add_subplot(2, 3, 2)
        mid_z = speed.shape[2] // 2
        im2 = ax2.contourf(speed[:, :, mid_z].T, levels=20, cmap='jet')
        plt.colorbar(im2, ax=ax2, label='|V|')
        ax2.set_title(f'Speed (Z={mid_z})')
        ax2.set_aspect('equal')
        
        # 3. 矢量叠加
        ax3 = fig.add_subplot(2, 3, 3)
        step_q = max(1, min(speed.shape[0], speed.shape[1]) // 12)
        im3 = ax3.contourf(speed[:, :, mid_z].T, levels=20, cmap='jet')
        ax3.quiver(gp[::step_q, ::step_q, mid_z, 0],
                    gp[::step_q, ::step_q, mid_z, 1],
                    vf[::step_q, ::step_q, mid_z, 0],
                    vf[::step_q, ::step_q, mid_z, 1],
                    color='white', alpha=0.8)
        plt.colorbar(im3, ax=ax3, label='|V|')
        ax3.set_title(f'Vectors (Z={mid_z})')
        ax3.set_aspect('equal')
        
        # 4. SNR分布
        ax4 = fig.add_subplot(2, 3, 4)
        mid_x = snr.shape[0] // 2
        im4 = ax4.imshow(snr[:, :, mid_z].T, cmap='hot', origin='lower')
        plt.colorbar(im4, ax=ax4, label='SNR')
        ax4.set_title(f'SNR (Z={mid_z})')
        
        # 5. 速度统计
        ax5 = fig.add_subplot(2, 3, 5)
        ax5.hist(speed.flatten(), bins=50, color='steelblue', edgecolor='white')
        ax5.set_xlabel('|V| (mm/s)'); ax5.set_ylabel('Count')
        ax5.set_title(f'Speed Distribution')
        ax5.axvline(np.mean(speed), color='red', ls='--')
        
        # 6. 统计信息
        ax6 = fig.add_subplot(2, 3, 6)
        ax6.axis('off')
        
        valid = speed > 0
        stats_text = (
            f"Tomographic PIV Statistics\n"
            f"{'=' * 35}\n"
            f"Grid: {grid_shape}\n"
            f"Valid vectors: {np.sum(valid)}/{speed.size}\n"
            f"Mean speed: {np.mean(speed[valid]):.2f} mm/s\n"
            f"Max speed: {np.max(speed):.2f} mm/s\n"
            f"RMS speed: {np.sqrt(np.mean(speed**2)):.2f} mm/s\n"
            f"Mean SNR: {np.mean(snr[valid]):.2f}\n"
            f"Particles (frame1): {len(particles_3d)}\n"
            f"dt: {velocity_result['config']['dt']:.4f} s\n"
        )
        ax6.text(0.1, 0.5, stats_text, transform=ax6.transAxes,
                 fontsize=11, va='center', fontfamily=self._font_name,
                 bbox=dict(boxstyle='round', facecolor='lightyellow'))
        
        fig.suptitle(title, fontsize=16, y=0.98)
        plt.tight_layout()
        
        path = save_path or os.path.join(self.output_dir,
                                          'piv_report.png')
        plt.savefig(path, dpi=200, bbox_inches='tight')
        plt.close()
        
        logger.info(f"PIV报告图已保存: {path}")
        return path

    def save_velocity_field_vtk(self,
                                 velocity_field: np.ndarray,
                                 grid_positions: np.ndarray,
                                 filename: str = 'velocity_field.vtk'):
        """
        导出速度场为VTK格式（兼容ParaView, VisIt）
        """
        path = os.path.join(self.output_dir, filename)
        nx, ny, nz, _ = velocity_field.shape
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write("# vtk DataFile Version 3.0\n")
            f.write("3D Velocity Field\n")
            f.write("ASCII\n")
            f.write("DATASET STRUCTURED_POINTS\n")
            f.write(f"DIMENSIONS {nx} {ny} {nz}\n")
            f.write(f"ORIGIN {grid_positions[0,0,0,0]:.4f} "
                    f"{grid_positions[0,0,0,1]:.4f} "
                    f"{grid_positions[0,0,0,2]:.4f}\n")
            
            if nx > 1:
                dx = grid_positions[1,0,0,0] - grid_positions[0,0,0,0]
            else:
                dx = 1.0
            if ny > 1:
                dy = grid_positions[0,1,0,1] - grid_positions[0,0,0,1]
            else:
                dy = 1.0
            if nz > 1:
                dz = grid_positions[0,0,1,2] - grid_positions[0,0,0,2]
            else:
                dz = 1.0
            
            f.write(f"SPACING {dx:.6f} {dy:.6f} {dz:.6f}\n")
            f.write(f"POINT_DATA {nx * ny * nz}\n")
            
            # 速度向量
            f.write("VECTORS velocity float\n")
            for i in range(nx):
                for j in range(ny):
                    for k in range(nz):
                        v = velocity_field[i, j, k]
                        f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            
            # 速度幅值
            speed = np.linalg.norm(velocity_field, axis=-1)
            f.write("SCALARS speed float\n")
            f.write("LOOKUP_TABLE default\n")
            for i in range(nx):
                for j in range(ny):
                    for k in range(nz):
                        f.write(f"{speed[i,j,k]:.6f}\n")
        
        logger.info(f"VTK文件已保存: {path}")
        return path

    def create_report_figure(self,
                              volume: np.ndarray,
                              point_cloud: np.ndarray,
                              projections: Dict[str, np.ndarray],
                              stats: dict,
                              save_path: Optional[str] = None):
        """
        生成综合报告图（4宫格）
        """
        fig = plt.figure(figsize=(16, 14))
        
        # 1. 投影对比
        n_cams = len(projections)
        cols = min(n_cams, 4)
        ax1 = fig.add_subplot(2, 2, 1)
        for i, (cam_id, proj) in enumerate(projections.items()):
            ax1.plot(proj.mean(axis=0), label=f'Cam {cam_id}', alpha=0.7)
        ax1.set_title('各视角投影轮廓', fontsize=12)
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # 2. 体素切片
        ax2 = fig.add_subplot(2, 2, 2)
        mid_z = volume.shape[2] // 2
        im = ax2.imshow(volume[:, :, mid_z].T, cmap='hot', origin='lower')
        plt.colorbar(im, ax=ax2, fraction=0.046)
        ax2.set_title(f'中间切片 (Z={mid_z})', fontsize=12)
        
        # 3. 3D点云
        ax3 = fig.add_subplot(2, 2, 3, projection='3d')
        ax3.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2],
                    c=point_cloud[:, 2], cmap='coolwarm', s=0.5, alpha=0.4)
        ax3.set_title('三维点云', fontsize=12)
        ax3.set_xlabel('X (mm)')
        ax3.set_ylabel('Y (mm)')
        ax3.set_zlabel('Z (mm)')
        
        # 4. 统计信息
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.axis('off')
        stats_text = (
            f"重建统计信息\n"
            f"{'=' * 35}\n"
            f"网格尺寸: {stats.get('grid_size', 'N/A')}\n"
            f"域尺寸: {stats.get('domain_size_mm', 'N/A')} mm\n"
            f"体素尺寸: {np.array(stats.get('voxel_size_mm', [0])).round(3)} mm\n"
            f"体素值范围: [{stats.get('volume_min', 0):.3f}, "
            f"{stats.get('volume_max', 1):.3f}]\n"
            f"有效体素: {stats.get('nonzero_voxels', 0)}\n"
            f"填充率: {stats.get('fill_fraction', 0) * 100:.1f}%\n"
            f"点云点数: {len(point_cloud)}\n"
        )
        ax4.text(0.1, 0.5, stats_text, transform=ax4.transAxes,
                 fontsize=12, verticalalignment='center',
                 fontfamily=self._font_name,
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        fig.suptitle('气泡三维层析重建报告', fontsize=16, y=0.98)
        plt.tight_layout()
        
        path = save_path or os.path.join(self.output_dir, 'reconstruction_report.png')
        plt.savefig(path, dpi=200, bbox_inches='tight')
        plt.close()
        
        logger.info(f"报告图已保存: {path}")
        return path
