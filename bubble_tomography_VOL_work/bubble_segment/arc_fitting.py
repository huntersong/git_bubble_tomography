"""
弧段拟合度计算模块。

对应 MATLAB: my_arc_fitting.m (VOL2.0, 2021-06-29)
"""

import numpy as np


def arc_fitting_ellipse(arc_points, ell_data, l_th=3):
    """
    计算弧段与椭圆拟合边界的拟合度（椭圆边界离散法）。

    Parameters
    ----------
    arc_points : np.ndarray
        Nx2 数组 (row, col)，弧段点。
    ell_data : list
        [a, b, x0, y0, phi] 椭圆参数。
    l_th : float
        距离阈值（像素）。

    Returns
    -------
    fitting_rate : float
        拟合度 (0-1)。
    """
    if arc_points is None or len(arc_points) == 0:
        return 0.0
    if ell_data is None or len(ell_data) < 5:
        return 0.0

    # 检查 NaN
    if any(np.isnan(v) for v in ell_data):
        return 0.0

    a, b, x0, y0, phi = ell_data
    arc_l = len(arc_points)

    # 生成椭圆离散点
    arc_num = max(arc_l, 100)
    elli = _write_ellipse(x0, y0, a, b, phi, arc_num)

    count = 0
    for i in range(arc_l):
        pt = arc_points[i]
        dists = np.sqrt(np.sum((elli - pt) ** 2, axis=1))
        min_len = np.min(dists)
        A_loc = np.argmin(dists)
        dists[A_loc] = 1e6
        min_len2 = np.min(dists)
        B_loc = np.argmin(dists)

        # 在两个最近点之间加密
        t_2 = np.linspace(1, arc_num + 1, arc_num)
        xita_1 = 2 * np.pi / arc_num * t_2[A_loc]
        xita_2 = 2 * np.pi / arc_num * t_2[B_loc]
        xita_diff = abs(xita_1 - xita_2)
        xita_begin = min(xita_1, xita_2)

        t_new = xita_begin + xita_diff / arc_num * t_2
        x_new = (x0 + a * np.cos(t_new) * np.cos(phi) -
                 b * np.sin(t_new) * np.sin(phi))
        y_new = (y0 - a * np.cos(t_new) * np.sin(phi) -
                 b * np.sin(t_new) * np.cos(phi))
        elli_2 = np.column_stack([y_new, x_new])

        dists2 = np.sqrt(np.sum((elli_2 - pt) ** 2, axis=1))
        if np.min(dists2) <= l_th:
            count += 1

    return count / arc_l


def _write_ellipse(x0, y0, a, b, theta, num_t):
    """
    生成椭圆离散点。
    返回 Nx2 数组 (row, col)。
    """
    t = np.linspace(0, 2 * np.pi, num_t, endpoint=False)
    x = x0 + a * np.cos(t) * np.cos(theta) - b * np.sin(t) * np.sin(theta)
    y = y0 - a * np.cos(t) * np.sin(theta) - b * np.sin(t) * np.cos(theta)
    return np.column_stack([y, x])  # (row, col)
