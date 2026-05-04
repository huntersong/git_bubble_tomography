"""
三维多相流流场测量软件

功能:
1. 多相机标定 (棋盘格/点阵, 3~N个相机)
2. 气泡图像预处理
3. MART层析重建算法
4. 三维点云输出与可视化
5. 示踪粒子3D重建 (Tomographic PIV)
6. 互相关速度场计算
7. PTV粒子跟踪测速 (四帧前后/最近邻/松弛法/STB)

使用方式:
  python main.py --gui          # 启动GUI
  python main.py --cli          # 命令行模式
  python main.py --demo         # 运行气泡重建演示
  python main.py --piv-demo     # 运行Tomographic PIV演示
  python main.py --ptv-demo     # 运行PTV粒子跟踪演示
"""
import cv2
import sys
import os
import argparse
import logging
import numpy as np

# 项目根目录添加到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from calibration.camera_calibrator import MultiCameraCalibrator, CameraParams


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%H:%M:%S'
    )


def run_gui():
    """启动GUI"""
    from gui.main_window import run_gui
    run_gui()


def run_demo():
    """运行演示（使用合成数据）"""
    import cv2
    from calibration.camera_calibrator import MultiCameraCalibrator
    from mart.mart_reconstructor import MARTReconstructor, MARTConfig
    from utils.image_processor import BubbleImageProcessor
    from visualization.visualizer import ResultVisualizer

    setup_logging(True)
    logger = logging.getLogger('demo')

    output_dir = os.path.join(os.path.dirname(__file__), 'demo_output')
    os.makedirs(output_dir, exist_ok=True)
    visualizer = ResultVisualizer(output_dir)

    print("=" * 60)
    print("气泡三维层析重建 - 演示模式")
    print("=" * 60)

    # ===== 第一步: 模拟标定 =====
    print("\n[1/4] 模拟4相机标定...")
    np.random.seed(42)

    n_cameras = 4
    image_size = (640, 480)

    # 模拟标定参数
    calibrator = MultiCameraCalibrator(
        pattern_type='checkerboard',
        pattern_size=(9, 6),
        square_size=5.0
    )

    # 为每个相机生成模拟标定参数
    from calibration.camera_calibrator import CameraParams

    for i in range(n_cameras):
        cam_id = f"cam{i+1}"

        # 模拟焦距和主点
        fx = fy = 800 + np.random.randn() * 20
        cx, cy = image_size[0] / 2 + np.random.randn() * 5, \
                 image_size[1] / 2 + np.random.randn() * 5
        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float64)

        # 模拟畸变
        D = np.array([0.1, -0.05, 0.001, -0.001, 0.0], dtype=np.float64)

        # 模拟相机位姿（围绕重建域分布）
        angle = 2 * np.pi * i / n_cameras
        radius = 100  # mm
        rvec_input = np.array([
            0.1 * np.sin(angle),
            0.1 * np.cos(angle),
            angle
        ])
        R = cv2.Rodrigues(rvec_input)[0]
        t = np.array([
            radius * np.cos(angle),
            radius * np.sin(angle),
            0
        ], dtype=np.float64)

        rvec, _ = cv2.Rodrigues(R)
        params = CameraParams(
            camera_id=cam_id,
            image_size=list(image_size),
            camera_matrix=K.tolist(),
            dist_coeffs=D.flatten().tolist(),
            rvec=rvec.flatten().tolist(),
            tvec=t.tolist(),
            rms_error=0.3 + np.random.rand() * 0.2
        )
        calibrator.camera_params[cam_id] = params

    print(calibrator.get_calibration_report())

    # ===== 第二步: 模拟气泡投影 =====
    print("\n[2/4] 生成模拟气泡投影...")
    processor = BubbleImageProcessor()

    # 创建模拟气泡（球形）
    domain_size = 20.0  # mm
    grid_size = 64
    x = np.linspace(-domain_size/2, domain_size/2, grid_size)
    y = np.linspace(-domain_size/2, domain_size/2, grid_size)
    z = np.linspace(-domain_size/2, domain_size/2, grid_size)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # 双气泡
    bubble_radius1 = 3.0
    bubble_radius2 = 2.0
    center1 = np.array([2, 1, 0])
    center2 = np.array([-3, -1, 2])

    vol = ((X - center1[0])**2 + (Y - center1[1])**2 + (Z - center1[2])**2
           < bubble_radius1**2).astype(np.float64)
    vol += ((X - center2[0])**2 + (Y - center2[1])**2 + (Z - center2[2])**2
            < bubble_radius2**2).astype(np.float64)
    vol = np.clip(vol, 0, 1)

    # 模拟各相机投影
    projections = {}
    for cam_id, params in calibrator.camera_params.items():
        K = np.array(params.camera_matrix)
        
        # 简单的正交投影模拟（沿不同角度）
        angle = 2 * np.pi * list(calibrator.camera_params.keys()).index(cam_id) / n_cameras
        
        proj = np.zeros(image_size, dtype=np.float64)
        for iz in range(grid_size):
            for iy in range(grid_size):
                if np.max(vol[:, iy, iz]) > 0:
                    # 沿X方向积分
                    line_val = np.sum(vol[:, iy, iz]) / grid_size
                    # 投影到图像坐标
                    u = int(cy + (iz - grid_size/2) * 3)
                    v = int(cx + (iy - grid_size/2) * 3)
                    if 0 <= u < image_size[0] and 0 <= v < image_size[1]:
                        proj[u, v] = line_val

        # 添加高斯模糊模拟光学效果
        proj = cv2.GaussianBlur(proj, (5, 5), 1.5)
        projections[cam_id] = proj / (proj.max() + 1e-10)

    # 显示投影
    visualizer.plot_projection_comparison(
        projections, "模拟多视角投影",
        os.path.join(output_dir, 'demo_projections.png')
    )
    print(f"  各相机投影已生成")

    # ===== 第三步: MART重建 =====
    print("\n[3/4] 执行MART重建...")

    config = MARTConfig(
        grid_size=(grid_size, grid_size, grid_size),
        domain_size=(domain_size, domain_size, domain_size),
        relaxation_factor=0.5,
        max_iterations=20,  # 演示用，实际建议50+
        voxel_threshold=0.3
    )

    reconstructor = MARTReconstructor(config)

    # 准备相机参数
    camera_params_recon = {}
    for cam_id, params in calibrator.camera_params.items():
        P = calibrator.compute_projection_matrix(cam_id)
        K = np.array(params.camera_matrix)
        K_inv = np.linalg.inv(K)
        camera_params_recon[cam_id] = {'P': P, 'K_inv': K_inv}

    # 执行重建
    volume = reconstructor.reconstruct(projections, camera_params_recon)
    points, normals = reconstructor.extract_bubble_point_cloud()
    stats = reconstructor.get_volume_stats()

    print(f"  重建完成: {stats['nonzero_voxels']} 有效体素")
    print(f"  点云: {len(points)} 个点")

    # ===== 第四步: 输出结果 =====
    print("\n[4/4] 输出结果...")

    # 点云可视化
    visualizer.plot_point_cloud(points, normals, "气泡三维点云（演示）",
                                 save_path=os.path.join(output_dir, 'demo_pointcloud.png'))

    # 体素切片
    visualizer.plot_volume_slices(volume, 'z', 5,
                                   save_path=os.path.join(output_dir, 'demo_slices_z.png'))

    # 导出点云
    visualizer.save_point_cloud_ply(points, normals, 'demo_bubble.ply')
    visualizer.save_point_cloud_pcd(points, 'demo_bubble.pcd')
    visualizer.save_volume_npy(volume, 'demo_volume.npy')

    # 综合报告
    visualizer.create_report_figure(volume, points, projections, stats,
                                     os.path.join(output_dir, 'demo_report.png'))

    # 标定结果保存
    calibrator.save_results(os.path.join(output_dir, 'calibration'))

    print("\n" + "=" * 60)
    print("演示完成！所有结果保存在:")
    print(f"  {os.path.abspath(output_dir)}")
    print("=" * 60)

    return output_dir


def run_piv_demo():
    """运行粒子追踪 + 速度场演示"""
    from particles.particle_reconstructor import (
        Particle3DReconstructor, TriangulationConfig, Particle3D
    )
    from particles.velocity_field import (
        VelocityFieldCalculator, CorrelationConfig
    )
    from visualization.visualizer import ResultVisualizer

    import cv2

    setup_logging(True)
    logger = logging.getLogger('piv_demo')

    output_dir = os.path.join(os.path.dirname(__file__), 'piv_demo_output')
    os.makedirs(output_dir, exist_ok=True)
    visualizer = ResultVisualizer(output_dir)

    print("=" * 60)
    print("Tomographic PIV 演示")
    print("=" * 60)

    # ===== Step 1: 模拟标定 =====
    print("\n[1/5] 模拟4相机标定...")

    n_cameras = 4
    image_size = (640, 480)
    calibrator = MultiCameraCalibrator(
        pattern_type='checkerboard', pattern_size=(9, 6), square_size=5.0
    )

    for i in range(n_cameras):
        cam_id = f"cam{i+1}"
        angle = 2 * np.pi * i / n_cameras
        radius = 100.0
        fx = fy = 800.0
        cx, cy = image_size[0] / 2, image_size[1] / 2
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        D = np.array([0.1, -0.05, 0.001, -0.001, 0.0])
        rvec_input = np.array([0.1 * np.sin(angle), 0.1 * np.cos(angle), angle])
        R = cv2.Rodrigues(rvec_input)[0]
        t = np.array([radius * np.cos(angle), radius * np.sin(angle), 0.0])
        rvec, _ = cv2.Rodrigues(R)
        params = CameraParams(
            camera_id=cam_id, image_size=list(image_size),
            camera_matrix=K.tolist(), dist_coeffs=D.flatten().tolist(),
            rvec=rvec.flatten().tolist(), tvec=t.tolist(),
            rms_error=0.3 + np.random.rand() * 0.2
        )
        calibrator.camera_params[cam_id] = params
    print("  标定完成")

    # ===== Step 2: 生成模拟示踪粒子 =====
    print("\n[2/5] 生成模拟示踪粒子...")
    np.random.seed(42)
    domain = 20.0  # mm
    n_particles = 500

    # 第1帧: 随机分布粒子
    pos1 = (np.random.rand(n_particles, 3) - 0.5) * domain * 0.8
    # 第2帧: 粒子有位移（模拟简单剪切流）
    shear_rate = 50.0  # 1/s
    dt = 0.002  # s
    velocity = np.zeros_like(pos1)
    velocity[:, 0] = shear_rate * pos1[:, 1] * 0.01  # u = shear * y
    velocity[:, 1] = 20.0  # v = const upward
    velocity[:, 2] = np.random.randn(n_particles) * 5  # w = random

    pos2 = pos1 + velocity * dt

    # 添加噪声
    pos1 += np.random.randn(*pos1.shape) * 0.05
    pos2 += np.random.randn(*pos2.shape) * 0.05

    # 构造Particle3D对象
    particles_f1 = [
        Particle3D(position=pos1[i], n_views=4, quality=0.8)
        for i in range(n_particles)
    ]
    particles_f2 = [
        Particle3D(position=pos2[i], n_views=4, quality=0.8)
        for i in range(n_particles)
    ]
    print(f"  生成 {n_particles} 个粒子，dt={dt}s")

    # ===== Step 3: 粒子3D重建（使用模拟数据跳过实际检测） =====
    print("\n[3/5] 粒子3D重建...")
    print(f"  第1帧: {len(particles_f1)} 个粒子")
    print(f"  第2帧: {len(particles_f2)} 个粒子")

    visualizer.plot_particle_positions_3d(
        particles_f1, velocity,
        title='Tracer Particles with Velocity',
        save_path=os.path.join(output_dir, 'particles_with_velocity.png')
    )

    # ===== Step 4: 互相关速度场计算 =====
    print("\n[4/5] 互相关速度场计算...")

    vel_config = CorrelationConfig(
        interrogation_size=(2.0, 2.0, 2.0),
        overlap_ratio=0.5,
        subpixel_refinement=True,
        peak_threshold=1.0,
        max_displacement=10.0,
        median_filter=True
    )

    vel_calculator = VelocityFieldCalculator(
        config=vel_config,
        domain_size=(domain, domain, domain),
        dt=dt
    )

    vel_result = vel_calculator.compute_velocity_field(
        particles_f1, particles_f2, grid_resolution=(24, 24, 24)
    )

    vf = vel_result['velocity_field']
    speed = np.linalg.norm(vf, axis=-1)

    print(f"  速度场网格: {vf.shape[:3]}")
    print(f"  平均速度: {speed.mean():.2f} mm/s")
    print(f"  最大速度: {speed.max():.2f} mm/s")

    # ===== Step 5: 可视化与输出 =====
    print("\n[5/5] 生成可视化...")

    # 速度矢量图
    visualizer.plot_velocity_quiver(
        vel_result['velocity_field'],
        vel_result['grid_positions'],
        slice_axis='z',
        title='Tomographic PIV - Velocity Vectors'
    )

    # 速度切面
    visualizer.plot_velocity_magnitude_slices(
        vel_result['velocity_field'],
        vel_result['grid_positions']
    )

    # 速度统计
    visualizer.plot_velocity_statistics(
        vel_result['velocity_field'],
        vel_result['snr_field']
    )

    # PIV综合报告
    visualizer.create_velocity_report(vel_result, particles_f1)

    # 导出VTK
    visualizer.save_velocity_field_vtk(
        vel_result['velocity_field'],
        vel_result['grid_positions'],
        'velocity_field.vtk'
    )

    print("\n" + "=" * 60)
    print("Tomographic PIV 演示完成！")
    print(f"所有结果保存在: {os.path.abspath(output_dir)}")
    print("=" * 60)

    return output_dir


def run_ptv_demo():
    """运行PTV粒子跟踪演示"""
    from ptv.tracker import (
        PTVTracker, PTVConfig, Track
    )
    from ptv.velocity import PTVVelocityCalculator

    setup_logging(True)
    logger = logging.getLogger('ptv_demo')

    output_dir = os.path.join(os.path.dirname(__file__), 'ptv_demo_output')
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("PTV 粒子跟踪测速演示")
    print("=" * 60)

    # ===== Step 1: 模拟多帧3D粒子数据 =====
    print("\n[1/4] 生成模拟多帧3D粒子数据...")
    np.random.seed(42)

    n_frames = 8
    domain = 20.0  # mm
    n_particles = 300
    dt = 0.001  # s

    frames_particles = {}
    true_tracks = []

    # 模拟层流 + 随机扰动
    for fi in range(n_frames):
        t = fi * dt
        positions = np.zeros((n_particles, 3))

        for i in range(n_particles):
            # 初始位置 (固定种子保证一致性)
            np.random.seed(i * 1000 + fi)
            base_x = (i % 30 - 15) * 0.5
            base_y = (i // 30 - 5) * 0.5
            base_z = np.random.uniform(-domain * 0.3, domain * 0.3)

            # 层流速度场: u = U(y), v = const, w = small oscillation
            U_max = 50.0  # mm/s
            H = domain * 0.8
            u = U_max * (1 - (2 * base_y / H) ** 2)  # 抛物线分布
            v = 10.0  # 均匀上升
            w = 5.0 * np.sin(2 * np.pi * fi / n_frames)  # 振荡

            # 添加噪声
            noise = np.random.randn(3) * 0.02

            positions[i] = [
                base_x + u * t + noise[0],
                base_y + v * t + noise[1],
                base_z + w * t + noise[2]
            ]

        frames_particles[fi] = positions

    print(f"  生成 {n_frames} 帧, 每帧 {n_particles} 个粒子")
    print(f"  dt = {dt * 1000:.2f} ms, 域 = {domain} mm")

    # ===== Step 2: 执行PTV跟踪 =====
    print("\n[2/4] 执行PTV跟踪...")

    frame_indices = list(range(n_frames))

    # 方法1: 四帧前后跟踪
    print("\n  --- 四帧前后跟踪 ---")
    config_fb = PTVConfig(
        tracking_method="forward_backward",
        max_displacement=3.0,
        min_track_length=3,
        dt=dt,
        fb_max_speed_ratio=2.0,
        fb_acceleration_limit=3.0,
    )
    tracker_fb = PTVTracker(config_fb)
    result_fb = tracker_fb.track(frames_particles, frame_indices)
    print(result_fb.summary())

    # 方法2: 最近邻跟踪
    print("\n  --- 最近邻跟踪 ---")
    config_nn = PTVConfig(
        tracking_method="nearest_neighbor",
        max_displacement=3.0,
        min_track_length=3,
        dt=dt,
    )
    tracker_nn = PTVTracker(config_nn)
    result_nn = tracker_nn.track(frames_particles, frame_indices)
    print(result_nn.summary())

    # 方法3: 松弛法跟踪
    print("\n  --- 松弛法跟踪 ---")
    config_relax = PTVConfig(
        tracking_method="relaxation",
        max_displacement=3.0,
        min_track_length=3,
        dt=dt,
        relaxation_iterations=10,
        relaxation_neighbors=6,
        relaxation_sigma=2.0,
    )
    tracker_relax = PTVTracker(config_relax)
    result_relax = tracker_relax.track(frames_particles, frame_indices)
    print(result_relax.summary())

    # ===== Step 3: 计算速度场 =====
    print("\n[3/4] 计算PTV速度场...")

    # 使用松弛法结果（通常最好）
    best_result = result_relax

    vel_calculator = PTVVelocityCalculator(dt=dt, domain_size=(domain, domain, domain))
    profile = vel_calculator.compute_from_tracks(best_result.tracks)

    print(profile.summary())

    # 插值到网格
    if len(profile.positions) >= 10:
        grid_pos, grid_vel, grid_shape = vel_calculator.interpolate_to_grid(
            profile.positions, profile.velocities,
            grid_resolution=(16, 16, 16)
        )
        print(f"  网格插值: {grid_shape}")

    # ===== Step 4: 可视化 =====
    print("\n[4/4] 生成可视化...")

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # 图1: 轨迹对比
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw={'projection': '3d'})

    for ax, result, title in zip(
        axes,
        [result_fb, result_nn, result_relax],
        ['四帧前后跟踪', '最近邻跟踪', '松弛法跟踪']
    ):
        colors = np.random.rand(min(len(result.tracks), 100), 3)
        for i, track in enumerate(result.tracks[:100]):
            pos = np.array(track.positions)
            ax.plot(pos[:, 0], pos[:, 1], pos[:, 2],
                    color=colors[i], alpha=0.4, linewidth=0.5)
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.set_title(f'{title}\n({result.n_tracks} 条轨迹)')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ptv_tracks_comparison.png'), dpi=150)
    plt.close()
    print("  轨迹对比图已保存")

    # 图2: 速度场
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    speeds = np.linalg.norm(profile.velocities, axis=1)
    axes[0].hist(speeds, bins=30, color='#2196F3', alpha=0.7, edgecolor='white')
    axes[0].axvline(profile.mean_speed, color='red', linestyle='--',
                    label=f'平均: {profile.mean_speed:.1f} mm/s')
    axes[0].set_xlabel('速度大小 (mm/s)')
    axes[0].set_ylabel('粒子数')
    axes[0].set_title('速度大小分布')
    axes[0].legend()

    # 速度分量散点图
    if len(profile.velocities) > 0:
        axes[1].scatter(profile.velocities[:, 0], profile.velocities[:, 1],
                        c=speeds, cmap='jet', s=3, alpha=0.3)
    axes[1].set_xlabel('Vx (mm/s)')
    axes[1].set_ylabel('Vy (mm/s)')
    axes[1].set_title('速度分量分布')
    axes[1].set_aspect('equal')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ptv_velocity_distribution.png'), dpi=150)
    plt.close()
    print("  速度场分布图已保存")

    # 图3: 3D速度场
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    if len(profile.positions) > 0:
        sc = ax.scatter(
            profile.positions[:, 0],
            profile.positions[:, 1],
            profile.positions[:, 2],
            c=speeds, cmap='jet', s=5, alpha=0.5
        )
        fig.colorbar(sc, ax=ax, label='速度 (mm/s)', shrink=0.6)

    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title('PTV 3D速度场')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ptv_velocity_3d.png'), dpi=150)
    plt.close()
    print("  3D速度场图已保存")

    # 导出轨迹
    csv_path = os.path.join(output_dir, 'ptv_tracks.csv')
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write("track_id,frame_idx,x,y,z,vx,vy,vz,speed\n")
        for track in best_result.tracks:
            positions, velocities, frames = track.to_arrays()
            for j in range(len(positions)):
                f.write(f"{track.particle_id},{frames[j]},")
                f.write(f"{positions[j, 0]:.6f},{positions[j, 1]:.6f},{positions[j, 2]:.6f},")
                if j < len(velocities):
                    speed = np.linalg.norm(velocities[j])
                    f.write(f"{velocities[j, 0]:.6f},{velocities[j, 1]:.6f},{velocities[j, 2]:.6f},{speed:.6f}")
                else:
                    f.write("0,0,0,0")
                f.write("\n")
    print(f"  轨迹已导出: {csv_path}")

    print("\n" + "=" * 60)
    print("PTV演示完成！")
    print(f"所有结果保存在: {os.path.abspath(output_dir)}")
    print("=" * 60)

    return output_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='三维多相流流场测量软件')
    parser.add_argument('--gui', action='store_true', help='启动GUI界面')
    parser.add_argument('--cli', action='store_true', help='命令行模式')
    parser.add_argument('--demo', action='store_true', help='运行气泡重建演示')
    parser.add_argument('--piv-demo', action='store_true', help='运行Tomographic PIV演示')
    parser.add_argument('--ptv-demo', action='store_true', help='运行PTV粒子跟踪演示')
    parser.add_argument('-v', '--verbose', action='store_true', help='详细日志')

    args = parser.parse_args()

    setup_logging(args.verbose)

    if args.gui:
        run_gui()
    elif args.ptv_demo:
        run_ptv_demo()
    elif args.piv_demo:
        run_piv_demo()
    elif args.demo:
        run_demo()
    else:
        parser.print_help()
        print("\n提示: --gui 启动GUI, --demo 气泡重建演示, --piv-demo PIV演示, --ptv-demo PTV演示")
