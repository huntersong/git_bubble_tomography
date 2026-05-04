"""
统计过滤模块：基于对数正态分布的异常值过滤。

对应 MATLAB: my_statistics.m (VOL1.0, 2021-06-06)
假设气泡尺寸服从对数正态分布，去除大于上分位点(α=0.05)的气泡。
"""

import numpy as np
from scipy.stats import lognorm


def bubble_statistics(bubble_sizes, alpha=0.05, num_th=45):
    """
    对数正态分布异常值过滤。

    Parameters
    ----------
    bubble_sizes : np.ndarray
        气泡尺寸数组 (N,)。
    alpha : float
        上分位点显著性水平。
    num_th : int
        最少气泡数量阈值（低于此值不进行过滤）。

    Returns
    -------
    result : dict
        'filtered_sizes' : 过滤后的尺寸数组
        'mean_filtered'  : 过滤后均值
        'count_filtered' : 过滤后数量
        'max_filtered'   : 过滤后最大值
        'mean_original'  : 过滤前均值
        'count_original' : 过滤前数量
        'max_original'   : 过滤前最大值
    """
    sizes = np.asarray(bubble_sizes, dtype=np.float64).ravel()
    sizes = sizes[sizes > 0]  # 去除零和负值

    if len(sizes) == 0:
        return {
            'filtered_sizes': np.array([]),
            'mean_filtered': 0, 'count_filtered': 0, 'max_filtered': 0,
            'mean_original': 0, 'count_original': 0, 'max_original': 0,
        }

    mean_orig = np.mean(sizes)
    count_orig = len(sizes)
    max_orig = np.max(sizes)

    if len(sizes) > num_th:
        E_X = np.mean(sizes)
        D_X = np.var(sizes)

        mu = np.log(E_X) - 0.5 * np.log(1 + D_X / (E_X ** 2 + 1e-12))
        sigma = np.sqrt(np.log(1 + D_X / (E_X ** 2 + 1e-12)))

        p = 1 - alpha
        size_th = lognorm.ppf(p, s=sigma, scale=np.exp(mu))

        filtered = sizes[sizes <= size_th]
    else:
        filtered = sizes

    result = {
        'filtered_sizes': filtered,
        'mean_filtered': float(np.mean(filtered)) if len(filtered) > 0 else 0,
        'count_filtered': len(filtered),
        'max_filtered': float(np.max(filtered)) if len(filtered) > 0 else 0,
        'mean_original': float(mean_orig),
        'count_original': count_orig,
        'max_original': float(max_orig),
    }
    return result
