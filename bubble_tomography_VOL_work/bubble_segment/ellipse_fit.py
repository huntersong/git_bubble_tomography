"""
代数椭圆拟合（伪逆矩阵求解广义特征方程）。

对应 MATLAB: my_ellipsefit.m
输出: (semimajor_axis, semiminor_axis, x0, y0, phi)
"""

import numpy as np


def ellipse_fit(x, y):
    """
    代数椭圆拟合。

    Parameters
    ----------
    x : array-like
        x 坐标数组（对应 MATLAB 的列坐标 col）。
    y : array-like
        y 坐标数组（对应 MATLAB 的行坐标 row）。

    Returns
    -------
    a : float  - 半长轴
    b : float  - 半短轴
    x0 : float - 椭圆中心 x
    y0 : float - 椭圆中心 y
    phi : float - 旋转角度 (弧度)
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()

    if len(x) < 5:
        raise ValueError("至少需要5个点进行椭圆拟合")

    # 构建 M 矩阵和 b 向量
    M = np.column_stack([2 * x * y, y ** 2, 2 * x, 2 * y, np.ones(len(x))])
    b = -(x ** 2)

    # 伪逆求解
    e, _, _, _ = np.linalg.lstsq(M, b, rcond=None)

    a_coeff = 1.0
    b_coeff = e[0]
    c_coeff = e[1]
    d_coeff = e[2]
    f_coeff = e[3]
    g_coeff = e[4]

    # Mathworld 公式
    delta = b_coeff ** 2 - a_coeff * c_coeff
    if abs(delta) < 1e-12:
        raise ValueError("拟合退化，无法确定椭圆参数")

    x0 = (c_coeff * d_coeff - b_coeff * f_coeff) / delta
    y0 = (a_coeff * f_coeff - b_coeff * d_coeff) / delta

    # phi = 0.5 * acot((c - a) / (2*b))
    if abs(2 * b_coeff) < 1e-12:
        phi = 0.0
    else:
        phi = 0.5 * np.arctan2(1, (c_coeff - a_coeff) / (2 * b_coeff))
        # acot(x) = atan(1/x)，但需要正确象限
        val = (c_coeff - a_coeff) / (2 * b_coeff)
        phi = 0.5 * np.arctan(1.0 / val) if abs(val) > 1e-12 else 0.0

    nom = 2 * (a_coeff * f_coeff ** 2 + c_coeff * d_coeff ** 2 +
                g_coeff * b_coeff ** 2 - 2 * b_coeff * d_coeff * f_coeff -
                a_coeff * c_coeff * g_coeff)

    denom1 = delta * ((c_coeff - a_coeff) * np.sqrt(1 + (4 * b_coeff ** 2) / ((a_coeff - c_coeff) ** 2 + 1e-12)) - (c_coeff + a_coeff))
    denom2 = delta * ((a_coeff - c_coeff) * np.sqrt(1 + (4 * b_coeff ** 2) / ((a_coeff - c_coeff) ** 2 + 1e-12)) - (c_coeff + a_coeff))

    if abs(denom1) < 1e-12 or abs(denom2) < 1e-12 or nom < 0:
        raise ValueError("椭圆参数计算异常")

    a_prime = np.sqrt(nom / denom1)
    b_prime = np.sqrt(nom / denom2)

    semimajor = max(a_prime, b_prime)
    semiminor = min(a_prime, b_prime)

    if a_prime < b_prime:
        phi = np.pi / 2 + phi

    return float(semimajor), float(semiminor), float(x0), float(y0), float(phi)
