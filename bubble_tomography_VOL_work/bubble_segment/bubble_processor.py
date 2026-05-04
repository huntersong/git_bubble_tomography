"""
核心气泡处理模块：凹点识别、分类（椭圆/圆拟合）、重叠气泡分离准备。

对应 MATLAB: my_bub_processing.m (VOL1.0, 2021-06-03, 440行)
数据结构：
  bubble: list of dict, 每个字典:
    'is_spot': bool        - 是否斑点
    'ao_num': int/float    - 凹点数量 (0=椭圆拟合, 0.5=圆拟合, 1=单凹点, 2=双凹点, >2=多凹点)
    'roundness': float     - 圆度 4*pi*A/P^2
    'area': int            - 面积
    'aspect_ratio': float  - 长宽比
    'fit': list            - 拟合结果 [a, b, x0, y0, phi] 或圆 [radius, -, xc, yc, 0]
                             对于多凹点为 list of lists
    'fit_boundary': np.ndarray or list - 拟合所用的边界点 (Nx2, row/col)
"""

import numpy as np
from .ellipse_fit import ellipse_fit
from .circle_fit import circle_fit


def bubble_processing(boundaries, lab_image,
                      roundness_th=0.94, point_apart=5, bub_size_th=30,
                      length_width_ratio=(0.95, 1.1), method_flag=1,
                      single_ao_length=10):
    """
    对气泡边界进行凹点识别、分类和拟合。

    Parameters
    ----------
    boundaries : list of np.ndarray
        每个 Nx2 (row, col) 边界点数组。
    lab_image : np.ndarray
        标签图。
    roundness_th : float
        圆度判定阈值。
    point_apart : int
        凹点识别时前后点的间隔。不可更改。
    bub_size_th : int
        凹点识别的最小边界长度。
    length_width_ratio : tuple
        (min_ratio, max_ratio) 长宽比判定范围。
    method_flag : int
        0=圆度法, 1=长宽比法, 2=综合法。
    single_ao_length : int
        单凹点去除边界阈值。

    Returns
    -------
    bubble : list of dict
        识别和拟合后的气泡信息。
    bub_overlap : list of dict
        重叠气泡（凹点数 > 2）的信息。
    ao_data : list of dict
        重叠气泡的凹点详细数据。
    """
    bub_num = len(boundaries)
    if bub_num == 0:
        return [], [], []

    # ---- 提取边界坐标到二维数组 ----
    # MATLAB 用固定大小的二维数组，Python 用变长列表
    boundary_number = max(len(b) for b in boundaries) + 1 if boundaries else 1

    # iii[n] = row coords, jjj[n] = col coords
    iii = []  # list of 1D arrays (row)
    jjj = []  # list of 1D arrays (col)
    for n in range(bub_num):
        bnd = boundaries[n]
        # bwboundaries 的首尾点重合，去掉最后一个
        iii.append(bnd[:-1, 0].astype(np.float64))
        jjj.append(bnd[:-1, 1].astype(np.float64))

    F_bubble_num = np.array([len(iii[n]) for n in range(bub_num)])

    # ---- 计算包围盒 ----
    X = np.zeros((bub_num, 2))  # col min, max
    Y = np.zeros((bub_num, 2))  # row min, max
    for n in range(bub_num):
        if F_bubble_num[n] > 0:
            X[n, 0] = np.min(jjj[n])
            X[n, 1] = np.max(jjj[n])
            Y[n, 0] = np.min(iii[n])
            Y[n, 1] = np.max(iii[n])

    # ---- 凹点识别（三角函数法） ----
    theta_th = 200  # 凹点角度阈值（度）
    ao_pks = [[] for _ in range(bub_num)]
    ao_locs = [[] for _ in range(bub_num)]
    ao_point_num = np.zeros(bub_num, dtype=int)
    ao_point_de = [[] for _ in range(bub_num)]  # 凹点在 ao_locs 中的位置
    ao_point_de2 = [[] for _ in range(bub_num)]  # 凹点坐标 (row, col)

    for n in range(bub_num):
        if F_bubble_num[n] <= 2 * point_apart + 2 or F_bubble_num[n] < bub_size_th:
            continue

        aaa = np.zeros(F_bubble_num[n])  # 转动角度

        for k in range(point_apart, F_bubble_num[n] - point_apart):
            x1, y1 = jjj[n][k - point_apart], iii[n][k - point_apart]
            x2, y2 = jjj[n][k], iii[n][k]
            x3, y3 = jjj[n][k + point_apart], iii[n][k + point_apart]

            # 计算夹角
            v1 = np.array([x1 - x2, y1 - y2])
            v2 = np.array([x3 - x2, y3 - y2])
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12)
            cos_angle = np.clip(cos_angle, -1, 1)

            # 计算前序点和后续点到 x 轴的角度
            a1 = _angle_to_x_axis(x1 - x2, -(y1 - y2))  # 注意 MATLAB 行列方向
            a2 = _angle_to_x_axis(x3 - x2, -(y3 - y2))

            # 前序点逆时针到后续点的角度
            if a2 > a1:
                aaa[k] = a2 - a1
            elif a1 > a2:
                aaa[k] = 360 - (a1 - a2)

        # 角度滤波（加权平均平滑）
        aaaa = np.zeros_like(aaa)
        if F_bubble_num[n] > 11:
            aaaa[point_apart] = 0.8 * aaa[point_apart] + 0.2 * aaa[point_apart + 1]
            aaaa[F_bubble_num[n] - point_apart - 1] = (
                0.8 * aaa[F_bubble_num[n] - point_apart - 1] +
                0.2 * aaa[F_bubble_num[n] - point_apart - 2]
            )
            for i in range(point_apart + 1, F_bubble_num[n] - point_apart - 1):
                aaaa[i] = 0.2 * aaa[i - 1] + 0.6 * aaa[i] + 0.2 * aaa[i + 1]

        # 寻找极大值 (findpeaks)
        if np.max(aaaa) < 180:
            continue

        pks, locs = _find_peaks(aaaa)

        # 过滤：极大值周围10点内不应有更大值
        filtered_pks = []
        filtered_locs = []
        for pi, pk in zip(pks, locs):
            if locs > 10 and locs < F_bubble_num[n] - 10:
                neighborhood = aaaa[max(0, locs - 10):locs + 11]
                if np.sum(aaaa[locs] < neighborhood) > 0:
                    continue
            filtered_pks.append(pk)
            filtered_locs.append(locs)

        # 凹点判定
        for pi, loc in zip(filtered_pks, filtered_locs):
            if pi > theta_th:
                ao_pks[n].append(pi)
                ao_locs[n].append(loc)

        ao_point_num[n] = len(ao_pks[n])
        for idx, loc in enumerate(ao_locs[n]):
            ao_point_de[n].append(idx)
            ao_point_de2[n].append([int(iii[n][loc]), int(jjj[n][loc])])

    # ---- 构建输出 bubble 结构 ----
    bubble = []
    for n in range(bub_num):
        is_spot = (F_bubble_num[n] == 0)
        area = int(np.sum(lab_image == (n + 1)))
        perimeter = F_bubble_num[n]
        roundness = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
        aspect_ratio = abs((X[n, 1] - X[n, 0]) / (Y[n, 0] - Y[n, 1])) if (Y[n, 0] - Y[n, 1]) != 0 else 999

        b = {
            'is_spot': is_spot,
            'ao_num': ao_point_num[n],
            'roundness': roundness,
            'area': area,
            'aspect_ratio': aspect_ratio,
            'fit': [],
            'fit_boundary': [],
        }
        bubble.append(b)

    # ---- 无凹点分类：椭圆拟合 vs 圆拟合 ----
    for n in range(bub_num):
        if bubble[n]['ao_num'] != 0 or F_bubble_num[n] < 3:
            continue

        use_circle = False
        if method_flag == 1:  # 长宽比法
            if abs(length_width_ratio[0]) < abs(bubble[n]['aspect_ratio']) < abs(length_width_ratio[1]):
                use_circle = True
        elif method_flag == 0:  # 圆度法
            if abs(bubble[n]['roundness']) > roundness_th:
                use_circle = True
        else:  # 综合法
            if (abs(bubble[n]['roundness']) > roundness_th and
                    abs(length_width_ratio[0]) < abs(bubble[n]['aspect_ratio']) < abs(length_width_ratio[1])):
                use_circle = True

        if use_circle:
            radius, xc, yc = circle_fit(iii[n], jjj[n])
            bubble[n]['fit'] = [radius, radius, xc, yc, 0.0]
            bubble[n]['fit_boundary'] = np.column_stack([iii[n], jjj[n]])
            bubble[n]['ao_num'] = 0.5
        else:
            try:
                a, b_val, x0, y0, phi = ellipse_fit(jjj[n], iii[n])
                bubble[n]['fit'] = [a, b_val, x0, y0, phi]
                bubble[n]['fit_boundary'] = np.column_stack([iii[n], jjj[n]])
                bubble[n]['ao_num'] = 0
            except Exception:
                bubble[n]['fit'] = []

    # ---- 单凹点：去除凹点两侧边界后椭圆拟合 ----
    for n in range(bub_num):
        if abs(bubble[n]['ao_num']) != 1:
            continue
        if len(ao_locs[n]) == 0:
            continue

        loc = ao_locs[n][0]
        L = F_bubble_num[n]
        sal = single_ao_length

        # 去除凹点两侧 sal 个点后的边界
        if loc > sal and loc + sal < L:
            seg_i = np.concatenate([iii[n][loc + sal + 1:], iii[n][:loc - sal]])
            seg_j = np.concatenate([jjj[n][loc + sal + 1:], jjj[n][:loc - sal]])
        elif loc <= sal:
            seg_i = iii[n][loc + sal + 1: L - (sal - loc)]
            seg_j = jjj[n][loc + sal + 1: L - (sal - loc)]
        elif loc >= L - sal:
            start = sal - (L - loc) + 1
            seg_i = iii[n][start: loc - sal]
            seg_j = jjj[n][start: loc - sal]
        else:
            continue

        if len(seg_i) > 5:
            try:
                a, b_val, x0, y0, phi = ellipse_fit(seg_j, seg_i)
                bubble[n]['fit'] = [a, b_val, x0, y0, phi]
                bubble[n]['fit_boundary'] = np.column_stack([seg_i, seg_j])
                bubble[n]['ao_num'] = 1
            except Exception:
                bubble[n]['fit'] = []

    # ---- 双凹点：分区拟合 ----
    for n in range(bub_num):
        if abs(bubble[n]['ao_num']) != 2 or len(ao_locs[n]) < 2:
            continue

        loc1, loc2 = ao_locs[n][0], ao_locs[n][1]
        L = F_bubble_num[n]

        # 第一段：loc1+1 到 loc2
        seg1_i = iii[n][loc1 + 1:loc2]
        seg1_j = jjj[n][loc1 + 1:loc2]
        # 第二段：loc2+1 到末尾 + 开头到 loc1（逆时针）
        seg2_i = np.concatenate([iii[n][loc2 + 1:], iii[n][:loc1]])
        seg2_j = np.concatenate([jjj[n][loc2 + 1:], jjj[n][:loc1]])
        # 逆序（因为是逆时针方向）
        seg2_i = seg2_i[::-1]
        seg2_j = seg2_j[::-1]

        fits = []
        boundaries_used = []
        for si, sj in [(seg1_i, seg1_j), (seg2_i, seg2_j)]:
            if len(si) > 5:
                try:
                    a, b_val, x0, y0, phi = ellipse_fit(sj, si)
                    fits.append([a, b_val, x0, y0, phi])
                    boundaries_used.append(np.column_stack([si, sj]))
                except Exception:
                    fits.append([])
                    boundaries_used.append(np.array([]))
            else:
                fits.append([])
                boundaries_used.append(np.array([]))

        bubble[n]['fit'] = fits
        bubble[n]['fit_boundary'] = boundaries_used
        bubble[n]['ao_num'] = 2

    # ---- 多凹点（>2）：准备重叠气泡数据 ----
    bub_overlap = []
    ao_data = []
    for n in range(bub_num):
        if bubble[n]['ao_num'] <= 2:
            continue

        overlap_entry = dict(bubble[n])
        overlap_entry['original_index'] = n

        # 分弧段
        arcs_i = []
        arcs_j = []
        num_ao = len(ao_locs[n])
        for i in range(num_ao - 1):
            arcs_i.append(iii[n][ao_locs[n][i] + 1:ao_locs[n][i + 1]])
            arcs_j.append(jjj[n][ao_locs[n][i] + 1:ao_locs[n][i + 1]])
        # 最后一段：最后一个凹点到第一个凹点（逆时针）
        last_i = np.concatenate([iii[n][ao_locs[n][-1] + 1:], iii[n][:ao_locs[n][0]]])
        last_j = np.concatenate([jjj[n][ao_locs[n][-1] + 1:], jjj[n][:ao_locs[n][0]]])
        last_i = last_i[::-1]
        last_j = last_j[::-1]
        arcs_i.append(last_i)
        arcs_j.append(last_j)

        overlap_entry['arcs_i'] = arcs_i
        overlap_entry['arcs_j'] = arcs_j
        overlap_entry['arcs'] = [np.column_stack([ai, aj]) if len(ai) > 0 else np.array([]).reshape(0, 2)
                                 for ai, aj in zip(arcs_i, arcs_j)]

        bub_overlap.append(overlap_entry)
        ao_data.append({
            'ao_locs': ao_locs[n],
            'ao_point_de': ao_point_de[n],
            'ao_point_de2': ao_point_de2[n],
            'iii': iii[n],
            'jjj': jjj[n],
            'F_bubble_num': F_bubble_num[n],
        })

    return bubble, bub_overlap, ao_data


def _angle_to_x_axis(dx, dy):
    """计算向量 (dx, dy) 逆时针到 x 轴正半轴的角度（0-360度）。"""
    if dx == 0 and dy == 0:
        return 0
    angle = np.degrees(np.arctan2(dy, dx))  # -180 to 180
    if angle < 0:
        angle += 360
    return angle


def _find_peaks(arr):
    """简单的 findpeaks 实现，返回 (peaks, locations)。"""
    peaks = []
    locs = []
    for i in range(1, len(arr) - 1):
        if arr[i] > arr[i - 1] and arr[i] > arr[i + 1]:
            peaks.append(arr[i])
            locs.append(i)
    return peaks, locs
