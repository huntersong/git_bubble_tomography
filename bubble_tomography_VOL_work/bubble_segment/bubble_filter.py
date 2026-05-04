"""
气泡异常删除模块：三级校正 — 拟合度过滤 → 异常椭圆 → 重复椭圆。

对应 MATLAB: my_bub_deleting.m (VOL1.0→2.0, 2021-06-29)
"""

import numpy as np
from .arc_fitting import arc_fitting_ellipse


def bubble_deleting(bubble, maxsize_th=1000, minsize_th=5,
                    ratio_th=0.4, repetition_length_th=10,
                    fitting_th=0.65):
    """
    三级校正删除异常气泡。

    Parameters
    ----------
    bubble : list of dict
        气泡信息列表（来自 bubble_processing）。
    maxsize_th : float
        椭圆半轴最大尺寸。
    minsize_th : float
        椭圆半轴最小尺寸。
    ratio_th : float
        长短轴比下限。
    repetition_length_th : float
        重复椭圆圆心距阈值。
    fitting_th : float
        拟合度阈值。

    Returns
    -------
    bubble : list of dict
        校正后的气泡列表。
    """
    # 第一级：拟合度过滤
    bubble = _fitting_del(bubble, fitting_th)
    # 第二级：异常椭圆删除
    maxratio_th = 1.0 / ratio_th if ratio_th > 0 else 999
    bubble = _abnormal_ellipse(bubble, maxsize_th, minsize_th, ratio_th, maxratio_th)
    # 第三级：重复椭圆删除
    bubble = _repetitive_ellipse(bubble, repetition_length_th)

    return bubble


def _fitting_del(bubble, fitting_th):
    """删除与拟合边界拟合度过小的椭圆。"""
    for n in range(len(bubble)):
        ao = bubble[n]['ao_num']

        if ao > 1:
            # 多个拟合结果
            fits = bubble[n].get('fit', [])
            fit_bnds = bubble[n].get('fit_boundary', [])
            if isinstance(fits, list) and len(fits) > 0:
                for i in range(len(fits)):
                    if fits[i] is None or (isinstance(fits[i], list) and len(fits[i]) == 0):
                        continue
                    if i < len(fit_bnds) and fit_bnds[i] is not None and len(fit_bnds[i]) > 0:
                        try:
                            fr = arc_fitting_ellipse(fit_bnds[i], fits[i])
                            if fr < fitting_th:
                                fits[i] = []
                        except Exception:
                            pass
        elif ao in (0, 1):
            fits = bubble[n].get('fit', [])
            if isinstance(fits, list) and len(fits) > 0 and fits[0]:
                fit_bnds = bubble[n].get('fit_boundary', [])
                if len(fit_bnds) > 0 and len(fit_bnds[0]) > 0:
                    try:
                        fr = arc_fitting_ellipse(fit_bnds[0], fits[0])
                        if fr < fitting_th:
                            bubble[n]['fit'] = []
                    except Exception:
                        pass
    return bubble


def _abnormal_ellipse(bubble, maxsize_th, minsize_th, ratio_th, maxratio_th):
    """剔除异常椭圆（尺寸或长宽比超限）。"""
    for n in range(len(bubble)):
        ao = bubble[n]['ao_num']

        if ao > 1:
            fits = bubble[n].get('fit', [])
            for i in range(len(fits)):
                if fits[i] is None or (isinstance(fits[i], list) and len(fits[i]) == 0):
                    continue
                ell = fits[i]
                if len(ell) >= 2:
                    a, b_val = abs(ell[0]), abs(ell[1])
                    if (a < minsize_th or b_val < minsize_th or
                            a > maxsize_th or b_val > maxsize_th or
                            a / (b_val + 1e-12) < ratio_th or
                            a / (b_val + 1e-12) > maxratio_th):
                        fits[i] = []
        elif ao in (0, 1, 0.5):
            fits = bubble[n].get('fit', [])
            if isinstance(fits, list) and len(fits) > 0 and fits[0]:
                ell = fits[0]
                if len(ell) >= 2:
                    a, b_val = abs(ell[0]), abs(ell[1])
                    if (a > maxsize_th or b_val > maxsize_th or
                            a / (b_val + 1e-12) < ratio_th or
                            a / (b_val + 1e-12) > maxratio_th):
                        bubble[n]['fit'] = []

    return bubble


def _repetitive_ellipse(bubble, repetition_length_th):
    """剔除重复椭圆（保留拟合度高的）。"""
    for n in range(len(bubble)):
        ao = bubble[n]['ao_num']
        if ao <= 1:
            continue

        fits = bubble[n].get('fit', [])
        fit_bnds = bubble[n].get('fit_boundary', [])
        if not isinstance(fits, list):
            continue

        b = len(fits)
        for i in range(b - 1):
            for j in range(i + 1, b):
                if (not fits[i] or not fits[j] or
                        (isinstance(fits[i], list) and len(fits[i]) == 0) or
                        (isinstance(fits[j], list) and len(fits[j]) == 0)):
                    continue

                # 圆心距
                cx_i = fits[i][2] if len(fits[i]) > 2 else 0
                cy_i = fits[i][3] if len(fits[i]) > 3 else 0
                cx_j = fits[j][2] if len(fits[j]) > 2 else 0
                cy_j = fits[j][3] if len(fits[j]) > 3 else 0

                dist = np.sqrt((cx_i - cx_j) ** 2 + (cy_i - cy_j) ** 2)
                if dist < repetition_length_th:
                    # 比较拟合度
                    fr_i = 0.0
                    fr_j = 0.0
                    if i < len(fit_bnds) and len(fit_bnds[i]) > 0:
                        try:
                            fr_i = arc_fitting_ellipse(fit_bnds[i], fits[i])
                        except Exception:
                            pass
                    if j < len(fit_bnds) and len(fit_bnds[j]) > 0:
                        try:
                            fr_j = arc_fitting_ellipse(fit_bnds[j], fits[j])
                        except Exception:
                            pass

                    if fr_i > fr_j:
                        fits[j] = []
                    else:
                        fits[i] = []

    return bubble
