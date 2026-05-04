"""
气泡跨帧追踪模块：质心匹配追踪。

对应 MATLAB: my_bubble_tracking.m (VOL1.0, 2021-07-08)
"""

import numpy as np


def bubble_tracking(bubble, mode='first', tracking_data=None, maxlength_th=20):
    """
    跨帧气泡追踪。

    Parameters
    ----------
    bubble : list of dict
        当前帧气泡信息。
    mode : str
        'first' - 第一帧初始化
        'others' - 后续帧匹配
    tracking_data : list of dict or None
        上一帧的追踪数据。
    maxlength_th : float
        最大质心差阈值。

    Returns
    -------
    tracking_data : list of dict
        更新后的追踪数据。
    """
    if mode == 'first':
        return _init_tracking(bubble)
    else:
        return _match_tracking(bubble, tracking_data, maxlength_th)


def _centroid(boundary_pts):
    """计算边界点的质心。"""
    if boundary_pts is None or len(boundary_pts) == 0:
        return 0.0, 0.0
    pts = np.asarray(boundary_pts)
    return float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))


def _get_bubble_info(bubble, idx, sub_idx=0):
    """从气泡信息中提取追踪所需数据。"""
    b = bubble[idx]
    ao = b['ao_num']
    fits = b.get('fit', [])
    fit_bnd = b.get('fit_boundary', [])

    if ao > 1 and isinstance(fits, list):
        if sub_idx < len(fits) and fits[sub_idx]:
            fit = fits[sub_idx]
            bnd = fit_bnd[sub_idx] if sub_idx < len(fit_bnd) else np.array([])
            cx = fit[2] if len(fit) >= 3 else 0
            cy = fit[3] if len(fit) >= 4 else 0
            size = [fit[0], fit[1], fit[4] if len(fit) >= 5 else 0]
            return bnd, cx, cy, size
    elif ao == 0.5:
        if fits and fits[0]:
            fit = fits[0]
            bnd = fit_bnd[0] if len(fit_bnd) > 0 else np.array([])
            return bnd, fit[2] if len(fit) >= 3 else 0, fit[3] if len(fit) >= 4 else 0, [fit[0], fit[0], 0]
    elif ao in (0, 1):
        if fits and fits[0]:
            fit = fits[0]
            bnd = fit_bnd[0] if len(fit_bnd) > 0 else np.array([])
            cx = fit[2] if len(fit) >= 3 else 0
            cy = fit[3] if len(fit) >= 4 else 0
            size = [fit[0], fit[1], fit[4] if len(fit) >= 5 else 0]
            return bnd, cx, cy, size
    return None, 0, 0, []


def _init_tracking(bubble):
    """第一帧初始化追踪数据。"""
    tracking = []
    for n in range(len(bubble)):
        # 获取所有子气泡
        ao = bubble[n]['ao_num']
        if ao > 1:
            fits = bubble[n].get('fit', [])
            if isinstance(fits, list):
                for sub_i in range(len(fits)):
                    bnd, cx, cy, size = _get_bubble_info(bubble, n, sub_i)
                    if bnd is not None and len(bnd) > 0:
                        cent = _centroid(bnd)
                        tracking.append({
                            'start_frame': 1,
                            'end_frame': 1,
                            'active': True,
                            'frames': [{
                                'boundary': bnd,
                                'centroid': cent,
                                'center': [cx, cy],
                                'size': size,
                            }]
                        })
        else:
            bnd, cx, cy, size = _get_bubble_info(bubble, n)
            if bnd is not None and len(bnd) > 0:
                cent = _centroid(bnd)
                tracking.append({
                    'start_frame': 1,
                    'end_frame': 1,
                    'active': True,
                    'frames': [{
                        'boundary': bnd,
                        'centroid': cent,
                        'center': [cx, cy],
                        'size': size,
                    }]
                })
    return tracking


def _match_tracking(bubble, tracking_data, maxlength_th):
    """后续帧匹配追踪。"""
    if tracking_data is None:
        return _init_tracking(bubble)

    monitor = [False] * len(tracking_data)  # 是否已匹配

    for n in range(len(bubble)):
        ao = bubble[n]['ao_num']
        matched = False

        if ao > 1:
            fits = bubble[n].get('fit', [])
            if isinstance(fits, list):
                for sub_i in range(len(fits)):
                    if matched:
                        break
                    bnd, cx, cy, size = _get_bubble_info(bubble, n, sub_i)
                    if bnd is None or len(bnd) == 0:
                        continue
                    for t_idx in range(len(tracking_data)):
                        if monitor[t_idx] or not tracking_data[t_idx]['active']:
                            continue
                        last = tracking_data[t_idx]['frames'][-1]
                        dist = np.sqrt((cx - last['center'][0]) ** 2 + (cy - last['center'][1]) ** 2)
                        if dist <= maxlength_th:
                            monitor[t_idx] = True
                            cent = _centroid(bnd)
                            tracking_data[t_idx]['end_frame'] += 1
                            tracking_data[t_idx]['frames'].append({
                                'boundary': bnd, 'centroid': cent,
                                'center': [cx, cy], 'size': size,
                            })
                            matched = True
                            break
        else:
            bnd, cx, cy, size = _get_bubble_info(bubble, n)
            if bnd is None or len(bnd) == 0:
                continue
            for t_idx in range(len(tracking_data)):
                if monitor[t_idx] or not tracking_data[t_idx]['active']:
                    continue
                last = tracking_data[t_idx]['frames'][-1]
                dist = np.sqrt((cx - last['center'][0]) ** 2 + (cy - last['center'][1]) ** 2)
                if dist <= maxlength_th:
                    monitor[t_idx] = True
                    cent = _centroid(bnd)
                    tracking_data[t_idx]['end_frame'] += 1
                    tracking_data[t_idx]['frames'].append({
                        'boundary': bnd, 'centroid': cent,
                        'center': [cx, cy], 'size': size,
                    })
                    matched = True
                    break

    # 未匹配的旧追踪标记为消失
    for i in range(len(tracking_data)):
        if not monitor[i] and tracking_data[i]['active']:
            tracking_data[i]['active'] = False

    # 新出现的气泡
    for n in range(len(bubble)):
        ao = bubble[n]['ao_num']
        if ao > 1:
            fits = bubble[n].get('fit', [])
            if isinstance(fits, list):
                for sub_i in range(len(fits)):
                    bnd, cx, cy, size = _get_bubble_info(bubble, n, sub_i)
                    if bnd is not None and len(bnd) > 0:
                        cent = _centroid(bnd)
                        tracking_data.append({
                            'start_frame': tracking_data[0]['end_frame'] + 1 if tracking_data else 1,
                            'end_frame': tracking_data[0]['end_frame'] + 1 if tracking_data else 1,
                            'active': True,
                            'frames': [{
                                'boundary': bnd, 'centroid': cent,
                                'center': [cx, cy], 'size': size,
                            }]
                        })
        else:
            bnd, cx, cy, size = _get_bubble_info(bubble, n)
            if bnd is not None and len(bnd) > 0:
                cent = _centroid(bnd)
                tracking_data.append({
                    'start_frame': tracking_data[0]['end_frame'] + 1 if tracking_data else 1,
                    'end_frame': tracking_data[0]['end_frame'] + 1 if tracking_data else 1,
                    'active': True,
                    'frames': [{
                        'boundary': bnd, 'centroid': cent,
                        'center': [cx, cy], 'size': size,
                    }]
                })

    return tracking_data
