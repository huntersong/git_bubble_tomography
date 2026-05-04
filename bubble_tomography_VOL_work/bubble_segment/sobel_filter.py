"""
失焦气泡去除模块：基于 Sobel 梯度和区域灰度双阈值判定。

对应 MATLAB: my_sobel.m (VOL1.0, 2021-06-03)
"""

import numpy as np


def sobel_defocus_remove(boundaries, gray_image, lab_image,
                         grad_th=0.25, bub_grad_size_th=50, bub_gray_th=0.25):
    """
    识别失焦气泡并返回其在 boundaries 列表中的索引。

    Parameters
    ----------
    boundaries : list of np.ndarray
        每个元素为 Nx2 (row, col) 的边界点数组。
    gray_image : np.ndarray
        原始灰度图像 (float64, 0-1 or 0-255)。
    lab_image : np.ndarray
        带标签的连通域图。
    grad_th : float
        梯度阈值系数 (0-1)。
    bub_grad_size_th : int
        边界长度阈值，短于此值的更易被判定为失焦。
    bub_gray_th : float
        区域最大灰度阈值。

    Returns
    -------
    defocus_indices : list of int
        被判定为失焦的气泡在 boundaries 中的索引（0-based）。
    """
    if not boundaries:
        return []

    img_float = gray_image.astype(np.float64)
    im_S, im_L = img_float.shape
    objectnum = len(boundaries)

    if objectnum == 0:
        return []

    # ---- 计算每个边界点的 Sobel 梯度 ----
    sobelx = np.array([[-1, -2, -1],
                       [0, 0, 0],
                       [1, 2, 1]], dtype=np.float64)
    sobely = sobelx.T

    boundary_grads = []
    for i in range(objectnum):
        bnd = boundaries[i]
        grads = np.zeros(len(bnd))
        for j, (r, c) in enumerate(bnd):
            # 提取 3x3 邻域，越界用 0 填充
            r0, r1 = r - 1, r + 2
            c0, c1 = c - 1, c + 2
            patch = np.zeros((3, 3), dtype=np.float64)
            pr0, pr1 = max(0, r0), min(im_S, r1)
            pc0, pc1 = max(0, c0), min(im_L, c1)
            patch[pr0 - r0:pr0 - r0 + (pr1 - pr0),
                  pc0 - c0:pc0 - c0 + (pc1 - pc0)] = img_float[pr0:pr1, pc0:pc1]

            gx = np.sum(sobelx * patch)
            gy = np.sum(sobely * patch)
            grads[j] = np.sqrt(gx ** 2 + gy ** 2)
        boundary_grads.append(grads)

    # ---- 计算每个气泡区域的灰度统计 ----
    bub_gray = np.zeros((objectnum, 3))  # mean, area_count, min_gray
    for i in range(im_S):
        for j in range(im_L):
            lbl = lab_image[i, j]
            if lbl > 0:
                idx = lbl - 1
                if idx < objectnum:
                    bub_gray[idx, 0] += img_float[i, j]
                    bub_gray[idx, 1] += 1
                    if bub_gray[idx, 2] == 0 or img_float[i, j] < bub_gray[idx, 2]:
                        bub_gray[idx, 2] = img_float[i, j]

    # 计算平均灰度
    for i in range(objectnum):
        if bub_gray[i, 1] > 0:
            bub_gray[i, 0] /= bub_gray[i, 1]

    # ---- 计算平均梯度统计 ----
    mean_grads = np.array([np.mean(g) if len(g) > 0 else 0 for g in boundary_grads])
    max_mean_grad = np.max(mean_grads) if len(mean_grads) > 0 else 1.0
    bub_grad_th = max_mean_grad * grad_th

    # ---- 判定失焦 ----
    defocus_indices = []
    for g_i in range(objectnum):
        is_defocus = False
        if mean_grads[g_i] < bub_grad_th and len(boundaries[g_i]) < bub_grad_size_th:
            is_defocus = True
        elif bub_gray[g_i, 2] > bub_gray_th:
            is_defocus = True
        if is_defocus:
            defocus_indices.append(g_i)

    return defocus_indices
