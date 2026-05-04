"""
重叠气泡处理模块：弧段分类、partner 配对拟合、Qabf 清晰度加权、ifcluster 聚类验证。

对应 MATLAB: my_overlapbubbles.m (VOL1.0, 2021-06-03, 425行)
"""

import numpy as np
from .ellipse_fit import ellipse_fit


def overlap_bubbles(bub_overlap, ao_data_list, image,
                    major_arc_th=25, partner_cluster_distance=0.05,
                    partner_cluster_proportion=0.95, major_arc_angle=85):
    """
    处理多凹点重叠气泡，尝试分离出单个气泡。

    Parameters
    ----------
    bub_overlap : list of dict
        重叠气泡信息（从 bubble_processing 返回）。
    ao_data_list : list of dict
        凹点详细数据。
    image : np.ndarray
        去背景后的图像。

    Returns
    -------
    bub_overlap : list of dict
        处理后的重叠气泡信息，包含拟合结果。
    """
    if not bub_overlap:
        return bub_overlap

    # 计算 Qabf 清晰度图
    qabf_map = _compute_qabf(image)

    num2 = len(bub_overlap)

    for n in range(num2):
        entry = bub_overlap[n]
        ao_num = entry.get('ao_num', 0)
        if ao_num <= 2:
            continue

        arcs = entry.get('arcs', [])
        arcs_i = entry.get('arcs_i', [])
        arcs_j = entry.get('arcs_j', [])
        ao_data = ao_data_list[n] if n < len(ao_data_list) else {}

        num_arcs = len(arcs)
        if num_arcs == 0:
            continue

        # 计算每段弧的中点
        mid_points = []
        for i in range(num_arcs):
            ai = arcs_i[i] if i < len(arcs_i) else np.array([])
            aj = arcs_j[i] if i < len(arcs_j) else np.array([])
            if len(ai) > 0:
                mid_idx = len(ai) // 2
                mid_points.append([ai[mid_idx], aj[mid_idx]])
            else:
                mid_points.append([0, 0])

        # 极端点（弧段边界框）
        extreme_points = []
        for i in range(num_arcs):
            ai = arcs_i[i] if i < len(arcs_i) else np.array([])
            aj = arcs_j[i] if i < len(arcs_j) else np.array([])
            if len(ai) > 0:
                extreme_points.append({
                    'i_min': int(np.min(ai)), 'i_max': int(np.max(ai)),
                    'j_min': int(np.min(aj)), 'j_max': int(np.max(aj))
                })
            else:
                extreme_points.append({'i_min': 0, 'i_max': 0, 'j_min': 0, 'j_max': 0})

        # 标志数组
        mult_flag = []
        for i in range(num_arcs):
            mult_flag.append({
                'is_major': False,
                'type': 0,     # 1-4 凸向类型
                'fitted': False,
                'partner_arc': 0,
                'ellipse': None,
            })

        # 判断弧段类型和优弧
        ao_point_de2 = ao_data.get('ao_point_de2', [])
        for i in range(num_arcs - 1):
            if len(arcs_i[i]) <= major_arc_th:
                continue

            # 优弧判断
            if i < len(ao_point_de2) and (i + 1) < len(ao_point_de2):
                OA = np.array(ao_point_de2[i]) - np.array(mid_points[i])
                OB = np.array(ao_point_de2[i + 1]) - np.array(mid_points[i])
                cos_val = np.dot(OA, OB) / (np.linalg.norm(OA) * np.linalg.norm(OB) + 1e-12)
                cos_val = np.clip(cos_val, -1, 1)
                angle = np.degrees(np.arccos(cos_val))
                mult_flag[i]['is_major'] = (angle < major_arc_angle)

            # 弧的类型判断（凸向方向）
            if len(arcs_i[i]) > 0:
                line_mid_i = (arcs_i[i][0] + arcs_i[i][-1]) / 2
                line_mid_j = (arcs_j[i][0] + arcs_j[i][-1]) / 2
                mi, mj = mid_points[i]
                if mi >= line_mid_i and mj <= line_mid_j:
                    mult_flag[i]['type'] = 1
                elif mi >= line_mid_i and mj >= line_mid_j:
                    mult_flag[i]['type'] = 2
                elif mi <= line_mid_i and mj <= line_mid_j:
                    mult_flag[i]['type'] = 3
                elif mi <= line_mid_i and mj >= line_mid_j:
                    mult_flag[i]['type'] = 4

        # 最后一段弧的优弧判断
        i_last = num_arcs - 1
        if len(arcs_i[i_last]) > major_arc_th and len(ao_point_de2) > i_last:
            dis_A_mid = np.linalg.norm(np.array(ao_point_de2[i_last]) - np.array(mid_points[i_last]))
            dis_B_mid = np.linalg.norm(np.array(ao_point_de2[0]) - np.array(mid_points[i_last]))
            dis_A_B = np.linalg.norm(np.array(ao_point_de2[i_last]) - np.array(ao_point_de2[0]))
            if dis_A_mid >= dis_A_B and dis_B_mid >= dis_A_B:
                mult_flag[i_last]['is_major'] = True

        # partner 配对（射线延伸找对面弧段）
        partner = [0] * num_arcs
        for i in range(num_arcs):
            if len(arcs_i[i]) == 0:
                continue
            line_mid_i = (arcs_i[i][0] + arcs_i[i][-1]) / 2
            line_mid_j = (arcs_j[i][0] + arcs_j[i][-1]) / 2
            i1, j1 = mid_points[i]
            phi_angle = np.arctan2(i1 - line_mid_i, j1 - line_mid_j)

            arc_type = mult_flag[i]['type']
            if arc_type in (1, 3):
                direction = 1
            elif arc_type in (2, 4):
                direction = -1
            else:
                continue

            e_vals = np.arange(1, 1000, 5, dtype=np.float64)
            iii_ext = i1 + direction * e_vals * np.sin(phi_angle)
            jjjj_ext = j1 + direction * e_vals * np.cos(phi_angle)

            found = False
            for q in range(len(e_vals)):
                for w in range(num_arcs):
                    if w == i:
                        continue
                    ep = extreme_points[w]
                    if (ep['i_min'] < iii_ext[q] < ep['i_max'] and
                            ep['j_min'] < jjjj_ext[q] < ep['j_max']):
                        partner[i] = w
                        found = True
                        break
                if found:
                    break

        # 计算每段弧的平均清晰度
        mult_ao_qabf = np.zeros(num_arcs)
        for i in range(num_arcs):
            if len(arcs_i[i]) > 0:
                total_q = 0
                for q in range(len(arcs_i[i])):
                    r, c = int(arcs_i[i][q]), int(arcs_j[i][q])
                    if 0 <= r < qabf_map.shape[0] and 0 <= c < qabf_map.shape[1]:
                        total_q += qabf_map[r, c]
                mult_ao_qabf[i] = total_q / len(arcs_i[i])

        # 拟合结果存储
        fits = []
        fit_boundaries = []
        finish_count = 0

        # 先尝试 partner 拟合
        for q in range(num_arcs - 1):
            if mult_flag[q]['fitted'] or len(arcs_i[q]) <= 15:
                continue
            p = partner[q]
            if p == 0 or mult_flag[p]['fitted']:
                continue

            temp_i = np.concatenate([arcs_i[q], arcs_i[p]])
            temp_j = np.concatenate([arcs_j[q], arcs_j[p]])
            if len(temp_i) < 5:
                continue

            try:
                temp_ellipse = list(ellipse_fit(temp_j, temp_i))
                cluster_ok = _ifcluster(temp_ellipse[2], temp_ellipse[3],
                                        temp_ellipse[0], temp_ellipse[1],
                                        temp_ellipse[4], temp_i, temp_j,
                                        partner_cluster_distance, partner_cluster_proportion)
                if cluster_ok:
                    qabf_diff = abs(mult_ao_qabf[q] - mult_ao_qabf[p]) / (mult_ao_qabf[q] + 1e-12)
                    if qabf_diff < 0.2:
                        mult_flag[q]['fitted'] = True
                        mult_flag[p]['fitted'] = True
                        fits.append(temp_ellipse)
                        fit_boundaries.append(np.column_stack([temp_i, temp_j]))
                        finish_count += 1
                        mult_flag[q]['ellipse'] = temp_ellipse
            except Exception:
                pass

        # 对优弧进行单独拟合
        for i in range(num_arcs):
            if mult_flag[i]['fitted']:
                continue
            if not mult_flag[i]['is_major']:
                continue
            if len(arcs_i[i]) < 5:
                continue

            try:
                ell = list(ellipse_fit(arcs_j[i], arcs_i[i]))
                mult_flag[i]['fitted'] = True
                mult_flag[i]['ellipse'] = ell
                fits.append(ell)
                fit_boundaries.append(np.column_stack([arcs_i[i], arcs_j[i]]))
                finish_count += 1
            except Exception:
                pass

        # 剩余弧段两两聚合
        for q in range(num_arcs - 1):
            if mult_flag[q]['fitted'] or len(arcs_i[q]) <= 10:
                continue
            for w in range(q + 1, num_arcs):
                if mult_flag[w]['fitted'] or len(arcs_i[w]) <= 10:
                    continue
                if mult_flag[q]['type'] == mult_flag[w]['type']:
                    continue

                temp_i = np.concatenate([arcs_i[q], arcs_i[w]])
                temp_j = np.concatenate([arcs_j[q], arcs_j[w]])
                if len(temp_i) < 5:
                    continue

                try:
                    temp_ellipse = list(ellipse_fit(temp_j, temp_i))
                    cluster_ok = _ifcluster(temp_ellipse[2], temp_ellipse[3],
                                            temp_ellipse[0], temp_ellipse[1],
                                            temp_ellipse[4], temp_i, temp_j,
                                            0.05, 0.95)
                    if cluster_ok:
                        mult_flag[q]['fitted'] = True
                        mult_flag[w]['fitted'] = True
                        fits.append(temp_ellipse)
                        fit_boundaries.append(np.column_stack([temp_i, temp_j]))
                        finish_count += 1
                        break
                except Exception:
                    pass

        # 剩余长弧拟合
        for q in range(num_arcs):
            if mult_flag[q]['fitted'] or len(arcs_i[q]) < 50:
                continue
            try:
                ell = list(ellipse_fit(arcs_j[q], arcs_i[q]))
                mult_flag[q]['fitted'] = True
                fits.append(ell)
                fit_boundaries.append(np.column_stack([arcs_i[q], arcs_j[q]]))
                finish_count += 1
            except Exception:
                pass

        # 所有剩余弧段合并拟合
        remaining_i = []
        remaining_j = []
        for q in range(num_arcs):
            if not mult_flag[q]['fitted'] and len(arcs_i[q]) > 0:
                remaining_i.append(arcs_i[q])
                remaining_j.append(arcs_j[q])

        if remaining_i:
            re_i = np.concatenate(remaining_i)
            re_j = np.concatenate(remaining_j)
            if len(re_i) >= 5:
                try:
                    ell = list(ellipse_fit(re_j, re_i))
                    fits.append(ell)
                    fit_boundaries.append(np.column_stack([re_i, re_j]))
                except Exception:
                    pass

        entry['fit'] = fits
        entry['fit_boundary'] = fit_boundaries

    return bub_overlap


def _compute_qabf(image):
    """计算图像清晰度图（基于 Sobel 梯度）。"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    else:
        gray = image.astype(np.float64)

    h1 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float64)
    h2 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)

    SAx = _conv2d(gray, h2)
    SAy = _conv2d(gray, h1)
    return np.sqrt(SAx ** 2 + SAy ** 2)


def _conv2d(image, kernel):
    """简单的 2D 卷积（same padding）。"""
    import cv2
    return cv2.filter2D(image, cv2.CV_64F, kernel)


def _ifcluster(x0, y0, a, b, phi, i_arr, j_arr, param1=0.2, param2=0.8):
    """
    聚类验证：检查点集是否分布在以焦点为中心的椭圆上。

    x0, y0: 椭圆中心
    a, b: 半轴
    phi: 旋转角度
    i_arr, j_arr: 点集 (row, col)
    """
    a, b = abs(a), abs(b)
    if a < 10 or b < 10:
        return False

    c = np.sqrt(a ** 2 - b ** 2)
    x1 = x0 + c * np.cos(phi)
    y1 = y0 - c * np.sin(phi)
    x2 = x0 - c * np.cos(phi)
    y2 = y0 + c * np.sin(phi)

    num_cluster = 0
    n_pts = len(i_arr)
    for n in range(n_pts):
        dist = (np.sqrt((j_arr[n] - x1) ** 2 + (i_arr[n] - y1) ** 2) +
                np.sqrt((j_arr[n] - x2) ** 2 + (i_arr[n] - y2) ** 2))
        if abs(dist - 2 * a) / (2 * a) < param1:
            num_cluster += 1

    return (num_cluster / n_pts) > param2 if n_pts > 0 else False
