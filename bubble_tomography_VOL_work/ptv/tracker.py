"""
PTV 粒子跟踪算法模块

实现多种经典的PTV跟踪算法，用于从连续帧的3D粒子位置序列中
建立粒子对应关系并计算速度。

参考: PTV_report.docx 中的算法伪代码

算法列表:
1. ForwardBackwardTracker - 四帧前后跟踪法
   原理: 使用连续4帧 (i-1, i, i+1, i+2) 进行前向和后向跟踪，
         通过前后向一致性验证剔除错误匹配。

2. NearestNeighborTracker - 最近邻跟踪法
   原理: 假设粒子在短时间内位移有限，取空间距离最近的粒子作为匹配目标。

3. RelaxationTracker - 松弛法跟踪
   原理: 迭代优化方法，利用位移场的空间连续性约束改善跟踪结果。
         每次迭代中，根据邻域粒子的位移更新当前粒子的最优匹配。

4. ShakeTheBoxTracker - Shake-The-Box (STB)
   原理: 高密度3D-PTV优化方法 (Schanz et al., 2016)。
         通过迭代更新粒子位置并投影验证，适用于高浓度粒子场景。
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from scipy.spatial import KDTree
from scipy.optimize import minimize
import logging

logger = logging.getLogger(__name__)


@dataclass
class PTVConfig:
    """PTV跟踪全局配置"""

    # --- 通用参数 ---
    tracking_method: str = "forward_backward"  # forward_backward / nearest_neighbor / relaxation / stb
    max_displacement: float = 5.0        # 最大允许位移 (mm/frame)
    min_track_length: int = 4            # 最短有效轨迹长度
    max_identity_change: int = 1         # 允许的身份切换次数

    # --- 粒子检测参数 (复用Particle3DReconstructor) ---
    detection_min_area: float = 2.0
    detection_max_area: float = 200.0
    detection_circularity: float = 0.5
    detection_brightness_threshold: int = 30

    # --- 前后跟踪参数 ---
    fb_max_speed_ratio: float = 2.0      # 前后向位移比阈值
    fb_acceleration_limit: float = 3.0   # 加速度限制 (mm/frame^2)

    # --- 松弛法参数 ---
    relaxation_iterations: int = 10      # 松弛迭代次数
    relaxation_neighbors: int = 6        # 邻域粒子数
    relaxation_sigma: float = 2.0        # 邻域权重标准差 (mm)

    # --- STB参数 ---
    stb_iterations: int = 50             # STB迭代次数
    stb_convergence_threshold: float = 0.1  # 收敛阈值 (mm)
    stb_intensity_threshold: float = 0.3    # 投影强度阈值
    stb_shake_amplitude: float = 1.0     # 初始扰动幅度 (mm)

    # --- 速度计算 ---
    dt: float = 0.001                    # 帧间时间间隔 (s)


@dataclass
class Track:
    """单条粒子轨迹"""
    particle_id: int                    # 粒子唯一标识
    positions: List[np.ndarray] = field(default_factory=list)  # 位置序列
    velocities: List[np.ndarray] = field(default_factory=list)  # 速度序列
    frame_indices: List[int] = field(default_factory=list)     # 帧索引
    quality: float = 1.0                # 轨迹质量 (0~1)

    @property
    def length(self) -> int:
        return len(self.positions)

    @property
    def displacement(self) -> Optional[np.ndarray]:
        """总位移"""
        if len(self.positions) < 2:
            return None
        return self.positions[-1] - self.positions[0]

    @property
    def mean_velocity(self) -> Optional[np.ndarray]:
        """平均速度"""
        if len(self.positions) < 2 or len(self.velocities) == 0:
            return None
        return np.mean(self.velocities, axis=0)

    def to_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """转换为numpy数组"""
        return (
            np.array(self.positions),
            np.array(self.velocities) if self.velocities else np.array([]),
            np.array(self.frame_indices),
        )


@dataclass
class TrackingResult:
    """跟踪结果"""
    tracks: List[Track]                           # 所有轨迹
    unmatched_frames: Dict[int, int] = field(default_factory=dict)  # {帧索引: 未匹配粒子数}
    n_particles_total: int = 0                    # 总粒子数
    n_tracks: int = 0                             # 轨迹数
    avg_track_length: float = 0.0                 # 平均轨迹长度
    tracking_efficiency: float = 0.0              # 跟踪效率

    def get_tracks_min_length(self, min_length: int) -> List[Track]:
        """获取满足最短长度要求的轨迹"""
        return [t for t in self.tracks if t.length >= min_length]

    def summary(self) -> str:
        """生成统计摘要"""
        lines = [
            f"PTV跟踪结果统计",
            f"  总粒子数: {self.n_particles_total}",
            f"  轨迹数: {self.n_tracks}",
            f"  平均轨迹长度: {self.avg_track_length:.1f} 帧",
            f"  跟踪效率: {self.tracking_efficiency:.1%}",
        ]
        if self.unmatched_frames:
            lines.append("  未匹配帧:")
            for fi, n in self.unmatched_frames.items():
                lines.append(f"    帧 {fi}: {n} 个未匹配")
        return "\n".join(lines)


class ForwardBackwardTracker:
    """
    四帧前后跟踪法

    使用连续4帧 (i-1, i, i+1, i+2) 进行粒子匹配:
    1. 前向跟踪: 帧 i -> 帧 i+1 -> 帧 i+2
    2. 后向验证: 帧 i+2 -> 帧 i+1 -> 帧 i
    3. 前后一致性检查: 位移方向/大小偏差
    4. 剔除不一致的匹配

    伪代码 (参考PTV_report.docx):
    ```
    FOR each particle p in frame_i:
        // 前向跟踪
        q_forward = find_nearest(p, frame_{i+1})
        r_forward = find_nearest(q_forward, frame_{i+2})

        // 后向跟踪
        q_backward = find_nearest(r_forward, frame_{i+1})
        p_backward = find_nearest(q_backward, frame_i)

        // 一致性验证
        IF q_forward == q_backward AND dist(p, p_backward) < threshold:
            accept match (p, q_forward, r_forward)
        ELSE:
            reject match
    ```
    """

    def __init__(self, config: PTVConfig):
        self.config = config
        self._next_particle_id = 0

    def track(self,
              frames_particles: Dict[int, np.ndarray],
              frame_indices: List[int]) -> List[Track]:
        """
        对连续帧序列执行前后跟踪

        Parameters
        ----------
        frames_particles : Dict[int, np.ndarray]
            {帧索引: 粒子位置数组 (N, 3)} mm
        frame_indices : List[int]
            按时间排序的帧索引列表

        Returns
        -------
        tracks : List[Track]
        """
        self._next_particle_id = 0
        tracks = []
        unmatched = {}

        max_disp = self.config.max_displacement
        speed_ratio = self.config.fb_max_speed_ratio
        accel_limit = self.config.fb_acceleration_limit

        # 需要4帧才能执行四帧跟踪
        if len(frame_indices) < 4:
            logger.warning(f"前后跟踪需要至少4帧，当前只有{len(frame_indices)}帧")
            return tracks

        for k in range(len(frame_indices) - 3):
            fi0 = frame_indices[k]      # frame i-1
            fi1 = frame_indices[k + 1]  # frame i
            fi2 = frame_indices[k + 2]  # frame i+1
            fi3 = frame_indices[k + 3]  # frame i+2

            pos0 = frames_particles.get(fi0)
            pos1 = frames_particles.get(fi1)
            pos2 = frames_particles.get(fi2)
            pos3 = frames_particles.get(fi3)

            if any(p is None or len(p) == 0 for p in [pos0, pos1, pos2, pos3]):
                continue

            # 构建KDTree加速搜索
            tree1 = KDTree(pos1)
            tree2 = KDTree(pos2)
            tree3 = KDTree(pos3)

            used_in_2 = set()
            used_in_3 = set()

            for idx_0, p0 in enumerate(pos0):
                # === Step 1: 前向跟踪 0 -> 1 -> 2 -> 3 ===
                dist_01, idx_1_fwd = tree1.query(p0)
                if dist_01 > max_disp:
                    continue

                p1 = pos1[idx_1_fwd]
                dist_12, idx_2_fwd = tree2.query(p1)
                if dist_12 > max_disp:
                    continue

                p2 = pos2[idx_2_fwd]
                dist_23, idx_3_fwd = tree3.query(p2)
                if dist_23 > max_disp:
                    continue

                # === Step 2: 后向跟踪 3 -> 2 -> 1 -> 0 ===
                p3 = pos3[idx_3_fwd]
                dist_32, idx_2_bwd = tree2.query(p3)
                if dist_32 > max_disp:
                    continue

                p2_bwd = pos2[idx_2_bwd]
                dist_21, idx_1_bwd = tree1.query(p2_bwd)
                if dist_21 > max_disp:
                    continue

                p1_bwd = pos1[idx_1_bwd]
                dist_10, idx_0_bwd = KDTree(pos0).query(p1_bwd)

                # === Step 3: 前后一致性验证 ===
                # 1) 中间帧一致性: idx_2_fwd == idx_2_bwd
                if idx_2_fwd != idx_2_bwd:
                    continue

                # 2) 第一帧回溯一致性
                if dist_10 > max_disp * 0.5:
                    continue

                # 3) 速度比检查
                if dist_12 > 1e-6 and dist_23 > 1e-6:
                    ratio = dist_23 / dist_12
                    if ratio > speed_ratio or ratio < 1.0 / speed_ratio:
                        continue

                # 4) 加速度检查
                d1 = pos1[idx_1_fwd] - p0
                d2 = pos2[idx_2_fwd] - pos1[idx_1_fwd]
                d3 = pos3[idx_3_fwd] - pos2[idx_2_fwd]
                accel1 = np.linalg.norm(d2 - d1)
                accel2 = np.linalg.norm(d3 - d2)
                if accel1 > accel_limit or accel2 > accel_limit:
                    continue

                # 标记已使用
                used_in_2.add(idx_2_fwd)
                used_in_3.add(idx_3_fwd)

                # 计算速度
                positions = [p0, pos1[idx_1_fwd], pos2[idx_2_fwd], pos3[idx_3_fwd]]
                frames = [fi0, fi1, fi2, fi3]
                dt = self.config.dt

                velocities = []
                for j in range(len(positions) - 1):
                    vel = (positions[j + 1] - positions[j]) / dt
                    velocities.append(vel)

                track = Track(
                    particle_id=self._next_particle_id,
                    positions=positions,
                    velocities=velocities,
                    frame_indices=frames,
                    quality=1.0 / (1.0 + dist_10)
                )
                self._next_particle_id += 1
                tracks.append(track)

        # 记录未匹配
        for fi in frame_indices:
            pos = frames_particles.get(fi)
            if pos is not None:
                unmatched[fi] = len(pos)

        logger.info(f"四帧前后跟踪完成: {len(tracks)} 条轨迹")
        return tracks


class NearestNeighborTracker:
    """
    最近邻跟踪法

    对每对相邻帧，通过空间最近邻匹配粒子。
    使用匈牙利算法或贪心匹配处理一对多冲突。

    伪代码:
    ```
    FOR consecutive frame pairs (frame_i, frame_{i+1}):
        Build KDTree from frame_{i+1} particles
        FOR each particle p in frame_i:
            IF dist(p, nearest(p, frame_{i+1})) < max_displacement:
                match p -> nearest
            ELSE:
                mark as lost
    ```
    """

    def __init__(self, config: PTVConfig):
        self.config = config
        self._next_particle_id = 0

    def track(self,
              frames_particles: Dict[int, np.ndarray],
              frame_indices: List[int]) -> List[Track]:
        """
        最近邻跟踪

        Parameters
        ----------
        frames_particles : Dict[int, np.ndarray]
            {帧索引: 粒子位置数组 (N, 3)}
        frame_indices : List[int]
            按时间排序的帧索引列表

        Returns
        -------
        tracks : List[Track]
        """
        self._next_particle_id = 0
        max_disp = self.config.max_displacement

        if len(frame_indices) < 2:
            return []

        # 初始化: 第一帧每个粒子开启一条轨迹
        active_tracks = {}  # {particle_id: Track}
        first_pos = frames_particles.get(frame_indices[0])
        if first_pos is None:
            return []

        for i, pos in enumerate(first_pos):
            pid = self._next_particle_id
            self._next_particle_id += 1
            active_tracks[pid] = Track(
                particle_id=pid,
                positions=[pos.copy()],
                frame_indices=[frame_indices[0]]
            )

        # 逐帧跟踪
        for k in range(len(frame_indices) - 1):
            fi_curr = frame_indices[k]
            fi_next = frame_indices[k + 1]

            pos_next = frames_particles.get(fi_next)
            if pos_next is None or len(pos_next) == 0:
                continue

            tree_next = KDTree(pos_next)
            used_next = set()
            finished_ids = []

            for pid, track in active_tracks.items():
                last_pos = track.positions[-1]

                dist, idx = tree_next.query(last_pos)

                if dist < max_disp and idx not in used_next:
                    # 匹配成功
                    new_pos = pos_next[idx]
                    dt = self.config.dt
                    velocity = (new_pos - last_pos) / dt

                    track.positions.append(new_pos.copy())
                    track.velocities.append(velocity)
                    track.frame_indices.append(fi_next)

                    used_next.add(idx)
                else:
                    # 未匹配，轨迹终止
                    finished_ids.append(pid)

            for pid in finished_ids:
                del active_tracks[pid]

        # 收集所有轨迹
        tracks = list(active_tracks.values())

        logger.info(f"最近邻跟踪完成: {len(tracks)} 条活跃轨迹")
        return tracks


class RelaxationTracker:
    """
    松弛法跟踪 (Relaxation Method)

    迭代优化粒子匹配，利用位移场的空间平滑性约束。
    每次迭代:
    1. 对每个粒子，根据当前匹配计算候选位移
    2. 利用邻域粒子的加权平均位移更新候选
    3. 选择与更新后位移最一致的匹配

    伪代码:
    ```
    Initialize matches with nearest neighbor
    FOR iteration = 1 to max_iterations:
        FOR each particle p in current frame:
            neighbors = find_K_nearest(p, current_frame)
            avg_displacement = weighted_average(displacements of matched neighbors)
            // 更新匹配: 偏好与邻域位移一致的候选
            best_match = argmin_q || (q - p) - avg_displacement ||
            IF dist(p, best_match) < max_displacement:
                update match(p) = best_match
        IF convergence:
            break
    ```

    参考: Pereira et al., "A refinement procedure for 3D particle tracking velocimetry"
    """

    def __init__(self, config: PTVConfig):
        self.config = config
        self._next_particle_id = 0

    def track(self,
              frames_particles: Dict[int, np.ndarray],
              frame_indices: List[int]) -> List[Track]:
        """
        松弛法跟踪

        Parameters
        ----------
        frames_particles : Dict[int, np.ndarray]
            {帧索引: 粒子位置数组 (N, 3)}
        frame_indices : List[int]
            按时间排序的帧索引列表

        Returns
        -------
        tracks : List[Track]
        """
        self._next_particle_id = 0
        max_disp = self.config.max_displacement
        n_neighbors = self.config.relaxation_neighbors
        n_iter = self.config.relaxation_iterations
        sigma = self.config.relaxation_sigma

        if len(frame_indices) < 2:
            return []

        # 先用最近邻法获得初始匹配
        nn_tracker = NearestNeighborTracker(self.config)
        initial_tracks = nn_tracker.track(frames_particles, frame_indices)

        if not initial_tracks:
            return []

        # 对每对相邻帧执行松弛优化
        # 构建帧间匹配关系
        frame_pair_matches = {}  # (fi_curr, fi_next): {idx_curr: idx_next}

        for k in range(len(frame_indices) - 1):
            fi_curr = frame_indices[k]
            fi_next = frame_indices[k + 1]

            pos_curr = frames_particles.get(fi_curr)
            pos_next = frames_particles.get(fi_next)

            if pos_curr is None or pos_next is None:
                continue

            n_curr = len(pos_curr)
            n_next = len(pos_next)

            # 初始匹配矩阵: match[i] = j 表示 curr[i] -> next[j]
            tree_next = KDTree(pos_next)
            initial_match = np.full(n_curr, -1, dtype=int)
            initial_disp = np.zeros((n_curr, 3))

            for i in range(n_curr):
                dist, idx = tree_next.query(pos_curr[i])
                if dist < max_disp:
                    initial_match[i] = idx
                    initial_disp[i] = pos_next[idx] - pos_curr[i]

            # 迭代松弛
            match = initial_match.copy()
            disp = initial_disp.copy()

            for iteration in range(n_iter):
                disp_new = disp.copy()
                tree_curr = KDTree(pos_curr)

                for i in range(n_curr):
                    if match[i] < 0:
                        continue

                    # 找到邻域粒子
                    dists, idxs = tree_curr.query(pos_curr[i], k=min(n_neighbors + 1, n_curr))

                    neighbor_disps = []
                    neighbor_weights = []

                    for d, j in zip(dists, idxs):
                        if j == i:
                            continue
                        if j < n_curr and match[j] >= 0:
                            w = np.exp(-d**2 / (2 * sigma**2))
                            neighbor_disps.append(disp[j])
                            neighbor_weights.append(w)

                    if neighbor_disps:
                        weights = np.array(neighbor_weights)
                        weights /= weights.sum()
                        avg_disp = np.sum(
                            [w * d for w, d in zip(weights, neighbor_disps)],
                            axis=0
                        )

                        # 用平均位移重新搜索匹配
                        predicted_pos = pos_curr[i] + avg_disp
                        dist_pred, idx_pred = tree_next.query(predicted_pos)

                        if dist_pred < max_disp:
                            disp_new[i] = pos_next[idx_pred] - pos_curr[i]
                            match[i] = idx_pred

                # 检查收敛
                change = np.linalg.norm(disp_new - disp)
                disp = disp_new

                if change < 0.01:
                    logger.debug(f"松弛法在迭代 {iteration + 1} 收敛")
                    break

            frame_pair_matches[(fi_curr, fi_next)] = {
                "match": match,
                "displacement": disp
            }

        # 从优化后的匹配重建轨迹
        tracks = self._build_tracks_from_matches(
            frames_particles, frame_indices, frame_pair_matches
        )

        logger.info(f"松弛法跟踪完成: {len(tracks)} 条轨迹")
        return tracks

    def _build_tracks_from_matches(self,
                                    frames_particles: Dict[int, np.ndarray],
                                    frame_indices: List[int],
                                    matches: Dict) -> List[Track]:
        """从帧间匹配关系重建连续轨迹"""
        tracks = []
        dt = self.config.dt

        if len(frame_indices) < 2:
            return tracks

        # 简化: 从第一帧开始链式追踪
        pos_first = frames_particles.get(frame_indices[0])
        if pos_first is None:
            return tracks

        for i in range(len(pos_first)):
            track = Track(
                particle_id=self._next_particle_id,
                positions=[pos_first[i].copy()],
                frame_indices=[frame_indices[0]]
            )
            self._next_particle_id += 1

            # 逐帧延伸
            for k in range(len(frame_indices) - 1):
                fi_curr = frame_indices[k]
                fi_next = frame_indices[k + 1]

                key = (fi_curr, fi_next)
                if key not in matches:
                    break

                m = matches[key]["match"]
                d = matches[key]["displacement"]

                # 找到当前帧中与轨迹末端最近的位置索引
                pos_curr = frames_particles.get(fi_curr)
                if pos_curr is None:
                    break

                tree = KDTree(pos_curr)
                dist, idx = tree.query(track.positions[-1])

                if idx < len(m) and m[idx] >= 0:
                    pos_next = frames_particles.get(fi_next)
                    if pos_next is not None:
                        track.positions.append(pos_next[m[idx]].copy())
                        track.velocities.append(d[idx] / dt)
                        track.frame_indices.append(fi_next)

            if track.length >= 2:
                tracks.append(track)

        return tracks


class ShakeTheBoxTracker:
    """
    Shake-The-Box (STB) 高密度3D-PTV优化方法

    原理 (Schanz et al., 2016):
    1. 初始化: 使用四帧跟踪获得初始粒子位置
    2. 迭代更新:
       a. 对每个粒子施加小幅随机扰动 (shake)
       b. 将粒子投影到各相机图像
       c. 计算投影图像与实际观测的残差
       d. 保留残差减小的更新 (ICP思想)
       e. 根据投影强度调整粒子强度/存在性
    3. 收敛后输出最终粒子位置和轨迹

    适用于高浓度粒子场景 (source density > 0.05 ppp)。

    伪代码:
    ```
    Initialize particles from 4-frame tracking
    FOR iteration = 1 to max_iterations:
        FOR each particle p:
            p_new = p + random_shake(amplitude)
            residual_old = compute_projection_residual(p)
            residual_new = compute_projection_residual(p_new)
            IF residual_new < residual_old:
                p = p_new
        amplitude *= decay_factor
        IF converged:
            break
    Update particle intensities from projections
    Filter weak particles
    ```
    """

    def __init__(self, config: PTVConfig):
        self.config = config
        self._next_particle_id = 0

    def track(self,
              frames_particles: Dict[int, np.ndarray],
              frame_indices: List[int],
              calibrator=None,
              images: Dict[int, Dict[str, np.ndarray]] = None) -> List[Track]:
        """
        STB跟踪

        Parameters
        ----------
        frames_particles : Dict[int, np.ndarray]
            {帧索引: 粒子位置数组 (N, 3)}
        frame_indices : List[int]
            按时间排序的帧索引列表
        calibrator : MultiCameraCalibrator, optional
            已标定的相机参数（用于投影验证）
        images : Dict[int, Dict[str, np.ndarray]], optional
            {帧索引: {相机ID: 图像}} 用于投影残差计算

        Returns
        -------
        tracks : List[Track]
        """
        self._next_particle_id = 0
        max_iter = self.config.stb_iterations
        threshold = self.config.stb_convergence_threshold
        amplitude = self.config.stb_shake_amplitude

        if len(frame_indices) < 2:
            return []

        tracks = []

        # Step 1: 初始化跟踪（使用最近邻法）
        nn_tracker = NearestNeighborTracker(self.config)
        initial_tracks = nn_tracker.track(frames_particles, frame_indices)

        if not initial_tracks:
            return []

        # Step 2: 对每帧执行STB位置优化
        for fi in frame_indices:
            pos = frames_particles.get(fi)
            if pos is None or len(pos) == 0:
                continue

            # 如果有标定参数和图像，执行投影验证优化
            if calibrator is not None and images is not None and fi in images:
                pos_optimized = self._optimize_positions(
                    pos, images[fi], calibrator, amplitude, max_iter, threshold
                )
                # 更新位置
                frames_particles[fi] = pos_optimized

        # Step 3: 重新建立轨迹（使用优化后的位置）
        nn_tracker2 = NearestNeighborTracker(self.config)
        tracks = nn_tracker2.track(frames_particles, frame_indices)

        # Step 4: 如果有标定参数，用投影强度过滤弱粒子
        if calibrator is not None and images is not None:
            tracks = self._filter_by_projection(
                tracks, frames_particles, images, calibrator
            )

        logger.info(f"STB跟踪完成: {len(tracks)} 条轨迹")
        return tracks

    def _optimize_positions(self,
                            positions: np.ndarray,
                            frame_images: Dict[str, np.ndarray],
                            calibrator,
                            initial_amplitude: float,
                            max_iterations: int,
                            convergence_threshold: float) -> np.ndarray:
        """
        STB位置优化: 通过迭代扰动和投影验证精化粒子位置

        Parameters
        ----------
        positions : np.ndarray (N, 3)
            初始粒子位置 (mm)
        frame_images : Dict[str, np.ndarray]
            {相机ID: 粒子图像}
        calibrator : MultiCameraCalibrator
        initial_amplitude : float
            初始扰动幅度 (mm)
        max_iterations : int
        convergence_threshold : float

        Returns
        -------
        optimized : np.ndarray (N, 3)
        """
        optimized = positions.copy()
        n_particles = len(optimized)
        cam_ids = list(frame_images.keys())
        decay = 0.95  # 扰动衰减因子

        # 预计算每个粒子的初始残差
        residuals = np.array([
            self._compute_projection_residual(
                optimized[i], frame_images, calibrator, cam_ids
            )
            for i in range(n_particles)
        ])

        amp = initial_amplitude

        for iteration in range(max_iterations):
            n_accepted = 0

            for i in range(n_particles):
                # 随机扰动
                shake = np.random.randn(3) * amp
                candidate = optimized[i] + shake

                # 计算新残差
                new_residual = self._compute_projection_residual(
                    candidate, frame_images, calibrator, cam_ids
                )

                if new_residual < residuals[i]:
                    optimized[i] = candidate
                    residuals[i] = new_residual
                    n_accepted += 1

            # 衰减扰动幅度
            amp *= decay

            # 收敛检查
            if iteration > 10 and n_accepted / n_particles < 0.01:
                logger.debug(f"STB在迭代 {iteration + 1} 收敛 (接受率: {n_accepted / n_particles:.2%})")
                break

        return optimized

    def _compute_projection_residual(self,
                                      position: np.ndarray,
                                      frame_images: Dict[str, np.ndarray],
                                      calibrator,
                                      cam_ids: List[str]) -> float:
        """
        计算单个粒子位置在所有相机上的投影残差

        残差 = 各相机投影位置处的图像亮度之和（越大说明位置越好）
        取负值使得最小化残差对应最大化亮度
        """
        total_intensity = 0.0
        n_valid = 0

        for cam_id in cam_ids:
            if cam_id not in calibrator.camera_params:
                continue
            if cam_id not in frame_images:
                continue

            image = frame_images[cam_id]
            cp = calibrator.camera_params[cam_id]

            # 投影
            P = calibrator.compute_projection_matrix(cam_id)
            K = np.array(cp.camera_matrix)
            D = np.array(cp.dist_coeffs)

            pos_h = np.array([position[0], position[1], position[2], 1.0])
            proj = P @ pos_h

            if abs(proj[2]) < 1e-10:
                continue

            u = proj[0] / proj[2]
            v = proj[1] / proj[2]

            # 获取图像亮度
            h, w = image.shape[:2]
            ui, vi = int(round(u)), int(round(v))

            if 0 <= ui < w and 0 <= vi < h:
                # 取小窗口内的最大亮度
                r = 3
                y0 = max(0, vi - r)
                y1 = min(h, vi + r + 1)
                x0 = max(0, ui - r)
                x1 = min(w, ui + r + 1)

                if image.ndim == 2:
                    patch = image[y0:y1, x0:x1]
                else:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
                    patch = gray[y0:y1, x0:x1]

                total_intensity += float(patch.max()) if patch.size > 0 else 0
                n_valid += 1

        if n_valid == 0:
            return 0.0

        # 返回负亮度（用于最小化）
        return -total_intensity / n_valid

    def _filter_by_projection(self,
                               tracks: List[Track],
                               frames_particles: Dict[int, np.ndarray],
                               images: Dict[int, Dict[str, np.ndarray]],
                               calibrator) -> List[Track]:
        """根据投影强度过滤低质量轨迹"""
        import cv2

        filtered = []
        intensity_thresh = self.config.stb_intensity_threshold

        for track in tracks:
            max_intensity = 0.0
            for j, fi in enumerate(track.frame_indices):
                pos = track.positions[j]
                frame_imgs = images.get(fi, {})

                for cam_id, img in frame_imgs.items():
                    if cam_id not in calibrator.camera_params:
                        continue

                    P = calibrator.compute_projection_matrix(cam_id)
                    pos_h = np.array([pos[0], pos[1], pos[2], 1.0])
                    proj = P @ pos_h

                    if abs(proj[2]) < 1e-10:
                        continue

                    u = int(round(proj[0] / proj[2]))
                    v = int(round(proj[1] / proj[2]))
                    h, w = img.shape[:2]

                    if 0 <= u < w and 0 <= v < h:
                        if img.ndim == 2:
                            val = float(img[v, u])
                        else:
                            val = float(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[v, u])
                        max_intensity = max(max_intensity, val)

            if max_intensity >= intensity_thresh:
                filtered.append(track)

        logger.info(f"投影过滤: {len(tracks)} -> {len(filtered)} 条轨迹")
        return filtered


class PTVTracker:
    """
    PTV跟踪器统一接口

    根据配置选择不同的跟踪算法，并输出统一的 TrackingResult。
    """

    def __init__(self, config: Optional[PTVConfig] = None):
        self.config = config or PTVConfig()

    def track(self,
              frames_particles: Dict[int, np.ndarray],
              frame_indices: Optional[List[int]] = None,
              calibrator=None,
              images: Dict[int, Dict[str, np.ndarray]] = None) -> TrackingResult:
        """
        执行PTV跟踪

        Parameters
        ----------
        frames_particles : Dict[int, np.ndarray]
            {帧索引: 粒子位置数组 (N, 3)} mm
        frame_indices : List[int], optional
            按时间排序的帧索引。如果为None，自动从keys排序。
        calibrator : MultiCameraCalibrator, optional
            标定参数（STB需要）
        images : Dict[int, Dict[str, np.ndarray]], optional
            {帧索引: {相机ID: 图像}} （STB需要）

        Returns
        -------
        result : TrackingResult
        """
        if frame_indices is None:
            frame_indices = sorted(frames_particles.keys())

        # 选择跟踪算法
        method = self.config.tracking_method

        if method == "forward_backward":
            tracker = ForwardBackwardTracker(self.config)
            tracks = tracker.track(frames_particles, frame_indices)
        elif method == "nearest_neighbor":
            tracker = NearestNeighborTracker(self.config)
            tracks = tracker.track(frames_particles, frame_indices)
        elif method == "relaxation":
            tracker = RelaxationTracker(self.config)
            tracks = tracker.track(frames_particles, frame_indices)
        elif method == "stb":
            tracker = ShakeTheBoxTracker(self.config)
            tracks = tracker.track(frames_particles, frame_indices,
                                   calibrator=calibrator, images=images)
        else:
            raise ValueError(f"未知跟踪方法: {method}")

        # 按最短轨迹长度过滤
        min_len = self.config.min_track_length
        tracks = [t for t in tracks if t.length >= min_len]

        # 计算统计信息
        total_particles = sum(
            len(frames_particles.get(fi, np.array([])))
            for fi in frame_indices
        )

        avg_length = np.mean([t.length for t in tracks]) if tracks else 0

        # 跟踪效率 = 总跟踪帧数 / (总粒子数 * 总帧对数)
        n_pairs = max(1, len(frame_indices) - 1)
        tracked_frames = sum(t.length for t in tracks)
        efficiency = tracked_frames / (total_particles * n_pairs) if total_particles > 0 else 0

        result = TrackingResult(
            tracks=tracks,
            n_particles_total=total_particles,
            n_tracks=len(tracks),
            avg_track_length=float(avg_length),
            tracking_efficiency=float(min(efficiency, 1.0))
        )

        return result
