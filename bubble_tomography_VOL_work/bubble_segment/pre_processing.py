"""
预处理模块：去背景 → 灰度调整 → 中值滤波 → 大津二值化 → 填洞 → 去小泡 → 边界提取

对应 MATLAB: my_pre_processing.m (VOL1.0, 2021-06-02)
"""

import numpy as np
import cv2


def pre_processing(image_bub, image_back=None, imagesize=None,
                   small_bub_size=20, gaussian_filter=False):
    """
    气泡图像预处理。

    Parameters
    ----------
    image_bub : np.ndarray
        气泡图像 (灰度, 0-255 uint8 或 0-1 float64)。
    image_back : np.ndarray or None
        背景图像。若为 None 则跳过去背景步骤。
    imagesize : tuple or None
        (row_start, row_end, col_start, col_end) 或 None 表示不裁剪。
    small_bub_size : int
        去除面积小于此值的连通域。
    gaussian_filter : bool
        是否使用高斯双核滤波（双边滤波）。

    Returns
    -------
    out_image : np.ndarray
        去背景后的灰度图像 (float64)。
    boundaries : list of np.ndarray
        每个元素为 Nx2 数组 (row, col)，表示一个气泡的边界点。
    gray_image : np.ndarray
        裁剪后的灰度图像 (float64)。
    lab_image : np.ndarray
        带标签的连通域图 (int32)，背景=0，气泡=1,2,...
    """
    # ---- 裁剪 ----
    if imagesize is not None:
        r0, r1, c0, c1 = imagesize
        image_bub_cut = image_bub[r0:r1, c0:c1].copy()
        if image_back is not None:
            image_back_cut = image_back[r0:r1, c0:c1].copy()
    else:
        image_bub_cut = image_bub.copy()
        image_back_cut = image_back.copy() if image_back is not None else None

    ori_image = image_bub_cut.astype(np.float64)

    # ---- 灰度归一化到 0-255 范围 ----
    if image_bub_cut.max() > 0:
        scale = 255.0 / image_bub_cut.max()
    else:
        scale = 1.0
    image_bub_fix = image_bub_cut.astype(np.float64) * scale

    if image_back_cut is not None:
        image_back_fix = image_back_cut.astype(np.float64) * scale
        image = image_back_fix - image_bub_fix
        out_image = image
    else:
        out_image = image_bub_fix
        image = image_bub_fix

    # ---- 灰度调整 (imadjust 近似: 归一化到 0-255) ----
    img_min = image.min()
    img_max = image.max()
    if img_max > img_min:
        image = (image - img_min) / (img_max - img_min) * 255.0
    else:
        image = np.zeros_like(image)

    # ---- 高斯双核滤波 (bilateralFilter 近似 imbilatfilt) ----
    if gaussian_filter:
        image_uint8 = np.clip(image, 0, 255).astype(np.uint8)
        image_uint8 = cv2.bilateralFilter(image_uint8, 9, 75, 75)
        image = image_uint8.astype(np.float64)

    # ---- 中值滤波 ----
    image_uint8 = np.clip(image, 0, 255).astype(np.uint8)
    image_uint8 = cv2.medianBlur(image_uint8, 3)
    image = image_uint8.astype(np.float64)

    # ---- 大津二值化 ----
    _, bw = cv2.threshold(image_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # ---- 填洞 ----
    # OpenCV 填洞：先取反 floodFill 再取反
    bw_filled = _fill_holes(bw)

    # ---- 去小泡 (bwareaopen) ----
    bw_clean = _bwareaopen(bw_filled, small_bub_size)

    # ---- 删除与边界粘连的气泡 ----
    lab = _remove_border_bubbles(bw_clean)

    # ---- 边界提取 (bwboundaries) ----
    boundaries = _bwboundaries(lab)

    return out_image, boundaries, ori_image, lab


def _fill_holes(bw):
    """填洞：填充二值图像中的空洞。"""
    h, w = bw.shape
    mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    # 找到所有背景种子点（边界上的0像素）
    bw_flood = bw.copy()
    # 对边界上的 0 进行 floodFill，标记为 255
    for y in range(h):
        for x in range(w):
            if y == 0 or y == h - 1 or x == 0 or x == w - 1:
                if bw_flood[y, x] == 0:
                    cv2.floodFill(bw_flood, mask, (x, y), 255)
    # 取反得到洞
    holes = cv2.bitwise_not(bw_flood)
    # 合并
    result = cv2.bitwise_or(bw, holes)
    return result


def _bwareaopen(bw, min_size):
    """去除面积小于 min_size 的连通域。"""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    result = np.zeros_like(bw)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            result[labels == i] = 255
    return result


def _remove_border_bubbles(bw):
    """去除与图像边界连通的气泡，返回重编号的标签图。"""
    num_labels, labels = cv2.connectedComponents(bw, connectivity=8)
    h, w = labels.shape
    # 找到与边界接触的标签
    border_labels = set()
    for y in range(h):
        if labels[y, 0] != 0:
            border_labels.add(labels[y, 0])
        if labels[y, w - 1] != 0:
            border_labels.add(labels[y, w - 1])
    # 也检查最右侧几列（MATLAB检查6列）
    for k in range(6):
        col = w - 1 - k
        if col >= 0:
            for y in range(h):
                if labels[y, col] != 0:
                    border_labels.add(labels[y, col])
    for x in range(w):
        if labels[0, x] != 0:
            border_labels.add(labels[0, x])
        if labels[h - 1, x] != 0:
            border_labels.add(labels[h - 1, x])

    # 移除边界气泡
    for lbl in border_labels:
        labels[labels == lbl] = 0

    # 重新编号
    mask = labels > 0
    if not mask.any():
        return labels.astype(np.int32)

    unique_vals = np.unique(labels[mask])
    new_labels = np.zeros_like(labels)
    for new_id, old_id in enumerate(unique_vals, start=1):
        new_labels[labels == old_id] = new_id
    return new_labels


def _bwboundaries(lab_image):
    """
    提取连通域边界（类似 MATLAB bwboundaries）。
    返回 list of Nx2 np.ndarray，每个数组为 (row, col) 坐标。
    """
    boundaries = []
    num_labels = lab_image.max()
    for lbl in range(1, num_labels + 1):
        mask = (lab_image == lbl).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for cnt in contours:
            # cnt shape: (N, 1, 2) -> (N, 2)，列顺序为 (x, y) = (col, row)
            pts = cnt.squeeze(axis=1)  # (N, 2) with (x, y)
            if pts.ndim == 1:
                continue
            # 转为 (row, col) 格式
            boundary = np.column_stack([pts[:, 1], pts[:, 0]])
            boundaries.append(boundary)
    return boundaries
