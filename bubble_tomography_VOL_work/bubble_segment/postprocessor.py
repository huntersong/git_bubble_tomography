"""
气泡后处理模块：计算等效直径、Sauter 平均直径。

对应 MATLAB: my_postprocessing.m (VOL1.0, 2021-06-04)
等效直径: d_eq = ((2a)^2 * 2b)^(1/3)，其中 a 为半长轴，b 为半短轴。
"""

import numpy as np


def postprocessing(bubble):
    """
    计算气泡等效直径和统计量。

    Parameters
    ----------
    bubble : list of dict
        气泡信息列表。

    Returns
    -------
    bubble_data : np.ndarray
        Mx4 数组: [序号, 等效直径, 气泡总数, Sauter平均直径D32]
        bubble_data[2,:] 存放过滤前的统计。
    """
    num_bubble = 0
    diameters = []

    for n in range(len(bubble)):
        ao = bubble[n]['ao_num']
        fits = bubble[n].get('fit', [])

        if ao == 0.5:
            # 圆拟合：直径 = 2*radius
            if fits and len(fits) >= 1 and fits[0]:
                d = 2 * abs(fits[0][0])
                diameters.append(d)
                num_bubble += 1
        elif ao == 0:
            if fits and len(fits) >= 1 and fits[0]:
                a, b = abs(fits[0][0]), abs(fits[0][1])
                d = ((2 * a) ** 2 * 2 * b) ** (1 / 3)
                diameters.append(d)
                num_bubble += 1
        elif ao == 1:
            if fits and len(fits) >= 1 and fits[0]:
                a, b = abs(fits[0][0]), abs(fits[0][1])
                d = ((2 * a) ** 2 * 2 * b) ** (1 / 3)
                diameters.append(d)
                num_bubble += 1
        elif ao == 2:
            if isinstance(fits, list):
                for fit in fits:
                    if fit and len(fit) >= 2:
                        try:
                            a, b = abs(fit[0]), abs(fit[1])
                            if a != 0 and b != 0:
                                d = ((2 * a) ** 2 * 2 * b) ** (1 / 3)
                                diameters.append(d)
                                num_bubble += 1
                        except (TypeError, IndexError):
                            pass
        elif ao > 2:
            if isinstance(fits, list):
                for fit in fits:
                    if fit and len(fit) >= 2:
                        try:
                            a, b = abs(fit[0]), abs(fit[1])
                            if a != 0 and b != 0:
                                d = ((2 * a) ** 2 * 2 * b) ** (1 / 3)
                                diameters.append(d)
                                num_bubble += 1
                        except (TypeError, IndexError):
                            pass

    # 计算 Sauter 平均直径 D[3,2]
    dia_3 = 0.0
    dia_2 = 0.0
    for d in diameters:
        if 2 < d < 100:
            dia_3 += d ** 3
            dia_2 += d ** 2

    d32 = dia_3 / dia_2 if dia_2 > 0 else 0

    # 构建输出数组
    if num_bubble > 0:
        bubble_data = np.zeros((max(len(diameters), 2), 4))
        for i, d in enumerate(diameters):
            bubble_data[i, 0] = i + 1
            bubble_data[i, 1] = d
        bubble_data[0, 2] = num_bubble
        bubble_data[0, 3] = d32
    else:
        bubble_data = np.zeros((2, 4))

    return bubble_data
