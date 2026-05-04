"""
最小二乘圆拟合。

对应 MATLAB: my_circfit.m
"""

import numpy as np


def circle_fit(x, y):
    """
    最小二乘法圆拟合。

    Parameters
    ----------
    x : array-like
        x 坐标（列坐标 col）。
    y : array-like
        y 坐标（行坐标 row）。

    Returns
    -------
    R : float  - 半径
    xc : float - 圆心 x
    yc : float - 圆心 y
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    N = len(x)

    x1 = np.sum(x)
    x2 = np.sum(x ** 2)
    y1 = np.sum(y)
    y2 = np.sum(y ** 2)
    x1y1 = np.sum(x * y)
    x1y2 = np.sum(x * y ** 2)
    x2y1 = np.sum(x ** 2 * y)

    C = N * x2 - x1 * x1
    D = N * x1y1 - x1 * y1
    E = N * (x2 ** 2 / N) + N * x1y2 - (x2 + y2) * x1  # 简化
    # 更精确的 E
    E_val = 0.0
    for i in range(N):
        E_val += x[i] ** 3 + x[i] * y[i] ** 2
    E_val = N * E_val - (x2 + y2) * x1

    G = N * y2 - y1 * y1
    H_val = 0.0
    for i in range(N):
        H_val += x[i] ** 2 * y[i] + y[i] ** 3
    H_val = N * H_val - (x2 + y2) * y1

    denom = C * G - D * D
    if abs(denom) < 1e-12:
        # 退化情况，返回质心和平均距离
        xc, yc = np.mean(x), np.mean(y)
        R = np.mean(np.sqrt((x - xc) ** 2 + (y - yc) ** 2))
        return float(R), float(xc), float(yc)

    a_val = (H_val * D - E_val * G) / denom
    b_val = (H_val * C - E_val * D) / (D * D - G * C)
    c_val = -(a_val * x1 + b_val * y1 + x2 + y2) / N

    xc = -a_val / 2
    yc = -b_val / 2
    R = np.sqrt(a_val ** 2 + b_val ** 2 - 4 * c_val) / 2

    return float(R), float(xc), float(yc)
