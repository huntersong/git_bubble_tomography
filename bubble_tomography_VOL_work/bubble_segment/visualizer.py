"""
可视化模块：在图像上绘制椭圆和圆。

对应 MATLAB: my_plot.m + plot_ellipse.m
"""

import numpy as np
import cv2


def draw_bubbles(image, bubble, color=(0, 0, 255), thickness=1):
    """
    在图像上绘制所有已拟合的气泡椭圆/圆。

    Parameters
    ----------
    image : np.ndarray
        背景图像 (BGR 或灰度)。
    bubble : list of dict
        气泡信息列表。
    color : tuple
        BGR 颜色。
    thickness : int
        线宽。

    Returns
    -------
    result : np.ndarray
        绘制后的图像。
    """
    if image is None:
        return None

    result = image.copy()
    if len(result.shape) == 2:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    for n in range(len(bubble)):
        ao = bubble[n]['ao_num']
        fits = bubble[n].get('fit', [])

        if ao > 1 and isinstance(fits, list):
            for fit in fits:
                if fit and len(fit) >= 5 and not (isinstance(fit[0], list) and len(fit[0]) == 0):
                    _draw_ellipse(result, fit, color, thickness)
        elif ao == 0.5:
            if fits and fits[0] and len(fits[0]) >= 3:
                # 圆拟合
                radius = abs(fits[0][0])
                xc = int(round(fits[0][2]))
                yc = int(round(fits[0][3]))
                cv2.circle(result, (xc, yc), int(round(radius)), color, thickness)
        elif ao in (0, 1):
            if fits and fits[0] and len(fits[0]) >= 5:
                _draw_ellipse(result, fits[0], color, thickness)

    return result


def _draw_ellipse(image, ell, color, thickness):
    """绘制一个椭圆。"""
    try:
        import cv2
        a, b, x0, y0, phi = ell
        a, b = abs(a), abs(b)
        if a < 1 or b < 1:
            return
        # OpenCV ellipse 参数
        center = (int(round(x0)), int(round(y0)))
        axes = (int(round(a)), int(round(b)))
        angle = -np.degrees(phi)  # OpenCV 使用度，逆时针
        cv2.ellipse(image, center, axes, angle, 0, 360, color, thickness)
    except Exception:
        pass


def draw_ellipse_points(x0, y0, a, b, theta, num_t=100):
    """
    生成椭圆离散点坐标（用于 matplotlib 绘图）。

    Parameters
    ----------
    x0, y0 : float
        椭圆中心。
    a, b : float
        半长轴、半短轴。
    theta : float
        旋转角度（弧度）。
    num_t : int
        离散点数。

    Returns
    -------
    x : np.ndarray
        x 坐标数组。
    y : np.ndarray
        y 坐标数组。
    """
    t = np.linspace(0, 2 * np.pi, num_t)
    x = x0 + a * np.cos(t) * np.cos(theta) - b * np.sin(t) * np.sin(theta)
    y = y0 - a * np.cos(t) * np.sin(theta) - b * np.sin(t) * np.cos(theta)
    return x, y
