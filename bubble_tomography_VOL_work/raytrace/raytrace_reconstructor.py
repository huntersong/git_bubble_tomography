"""
气泡光线追踪三维重建 - 核心算法模块

基于 MATLAB raytrace_main.m 移植
流程: 二值化 → Silhouette三维重建 → 面法向初始化 → 光线追踪迭代修正 → 曲面构建
算法: Silhouette方法 + Snell定律折射修正

适用场景: 透明容器内气泡的双视角几何重建与光线追踪修正
"""

import numpy as np
from scipy import ndimage
from scipy.interpolate import interp1d
from skimage import color, filters, measure, img_as_float
import logging

logger = logging.getLogger(__name__)


# ============================================================
# 核心算法函数（移植自MATLAB）
# ============================================================

def binary_tempfunc(image):
    """二值化+空洞填充，移植自 binary_tempfunc.m"""
    if len(image.shape) > 2:
        image = color.rgb2gray(image)
    image_float = img_as_float(image)

    # 二值化，阈值0.7（与MATLAB代码一致）
    I_bw = image_float < 0.7  # 反色：气泡为1，背景为0

    # 空洞填充
    I_bw_filled = ndimage.binary_fill_holes(I_bw)

    return I_bw_filled.astype(float)


def slice_position(binary_image):
    """
    获取每一层切片的左右边界点
    移植自 slice_position.m
    返回: point1 (N,3), point2 (N,3) - 左右边界 [x, y, 0]
    """
    # 获取连通域的bounding box
    labels = measure.label(binary_image.astype(int))
    props = measure.regionprops(labels)

    if len(props) == 0:
        return np.zeros((1, 3)), np.zeros((1, 3))

    # 取最大连通域
    areas = [p.area for p in props]
    main_region = props[np.argmax(areas)]
    bbox = main_region.bbox  # (min_row, min_col, max_row, max_col)

    b = round(bbox[0])  # top row
    height = round(bbox[2] - bbox[0]) - 1

    if height <= 0:
        height = 1

    point1 = np.zeros((height, 3))
    point2 = np.zeros((height, 3))

    for m in range(height):
        mm = b + m
        if mm >= binary_image.shape[0]:
            break
        row = binary_image[mm, :]
        # 找到这一行中值为1的像素范围
        ones_idx = np.where(row > 0.5)[0]
        if len(ones_idx) > 0:
            point1[m, 0] = ones_idx[0]
            point1[m, 1] = mm
            point2[m, 0] = ones_idx[-1]
            point2[m, 1] = mm
        else:
            point1[m, 0] = 0
            point1[m, 1] = mm
            point2[m, 0] = 0
            point2[m, 1] = mm

    return point1, point2


def bezier_points(P0, P1, P2):
    """
    二次Bezier曲线插值，移植自 bezier_points.m
    P0, P1, P2: 各为 (1,3) 的控制点 [x, y, z]
    返回: xx_selected, yy_selected (去重后的x, z坐标)
    """
    t = np.linspace(0, 1, 1000)
    x = (1-t)**2 * P0[0] + 2*(1-t)*t*P1[0] + t**2 * P2[0]
    y = (1-t)**2 * P0[2] + 2*(1-t)*t*P1[2] + t**2 * P2[2]

    idx = np.where((x >= P0[0]) & (x <= P2[0]))[0]

    if len(idx) == 0:
        P2 = P1.copy()
        x = (1-t)**2 * P0[0] + 2*(1-t)*t*P1[0] + t**2 * P2[0]
        y = (1-t)**2 * P0[2] + 2*(1-t)*t*P1[2] + t**2 * P2[2]
        idx = np.where((x >= P0[0]) & (x <= P2[0]))[0]

    if len(idx) == 0:
        return np.array([P0[0]]), np.array([P0[2]])

    x_selected = np.round(x[idx])
    y_selected = np.round(y[idx])

    # 去重：连续相同x只保留第一个
    xx_out = [x_selected[0]]
    yy_out = [y_selected[0]]
    for n in range(len(x_selected) - 1):
        if x_selected[n] != x_selected[n+1]:
            xx_out.append(x_selected[n+1])
            yy_out.append(y_selected[n+1])

    return np.array(xx_out), np.array(yy_out)


def slice_camera1and2_position(point1, point2, camera1, camera2):
    """
    双相机视角交点计算，移植自 slice_camera1and2_position.m
    """
    ab1 = np.tan(camera1)
    ab2 = -np.tan(camera2)
    cb1 = ab1 * point1[0] + 0 * point1[2]
    cb2 = ab2 * point2[0] + 1 * point2[2]

    # 解线性方程组 [ab1, 0; ab2, 1] * [x; z] = [cb1; cb2]
    A = np.array([[ab1, 0], [ab2, 1]])
    b = np.array([cb1, cb2])
    Point = np.linalg.solve(A, b)

    return np.array([Point[0], point1[1], Point[1]])


def raytrace_adjust(input_bubbletemp_silhouette, input_bubbletemp_face):
    """
    光线追踪修正算法，移植自 raytrace_adjust.m
    """
    planeNormal_LED = np.array([0, 0, 1])
    planePoint_LED = np.array([0, 0, -1500])

    output_silhouette = input_bubbletemp_silhouette.copy()
    output_face = input_bubbletemp_face.copy()

    miu1 = 1.0
    miu2 = 1.3
    rho_1_2 = miu1 / miu2

    for ray_count in range(100):
        z_theta = 1.0
        x_theta = np.tan(np.pi/36) * (-1 + 2*np.random.random())
        y_theta = np.tan(np.pi/36) * (-1 + 2*np.random.random())
        ray_direction_LED = np.array([x_theta, y_theta, z_theta])

        ray_position_image = np.array([
            input_bubbletemp_silhouette[0],
            input_bubbletemp_silhouette[1], 0
        ])
        d_maxValue_imagetobubble = abs(input_bubbletemp_silhouette[2])
        ray_direction_image = np.array([0, 0, -1])

        # Snell定律计算o1
        temp = np.dot(ray_direction_image, input_bubbletemp_face)
        discriminant = temp**2 - (1 - rho_1_2**2)
        if discriminant < 0:
            continue

        o1_temp_ray = ray_direction_image / rho_1_2 - (
            temp + np.sqrt(abs(discriminant))) * (input_bubbletemp_face / rho_1_2)

        # 计算LED面上交点
        denom = np.dot(planeNormal_LED, o1_temp_ray)
        if abs(denom) < 1e-12:
            continue
        t_param = np.dot(planeNormal_LED, planePoint_LED - input_bubbletemp_silhouette) / denom
        raypoint_at_LED = input_bubbletemp_silhouette + t_param * o1_temp_ray

        # 不共面定理判别
        tempcoplane = np.dot(np.cross(ray_direction_image, ray_direction_LED), o1_temp_ray)

        if abs(tempcoplane) > 1e-10:
            d1_length = -np.dot(
                np.cross(raypoint_at_LED - ray_position_image, ray_direction_LED),
                o1_temp_ray
            ) / tempcoplane

            bubble_face_position = ray_position_image + d1_length * ray_direction_image

            # 构建正交基
            v1 = (raypoint_at_LED - bubble_face_position)
            v1_norm = np.linalg.norm(v1)
            if v1_norm < 1e-12:
                continue
            v1 = v1 / v1_norm
            v2 = ray_direction_LED / np.linalg.norm(ray_direction_LED)
            B_d1 = np.column_stack([v1, v2])

            found = False
            for degree_circle in range(360):
                phi = degree_circle * np.pi / 180
                face_direction = B_d1 @ np.array([np.cos(phi), np.sin(phi)])

                n_parallel = ray_direction_image - rho_1_2 * face_direction

                if abs(np.dot(input_bubbletemp_face, n_parallel)) <= 0.001:
                    found = True
                    o1_tempupdate_ray = face_direction
                    n1_update = n_parallel
                    break

            if found:
                output_face = np.array([
                    n1_update[0],
                    n1_update[1],
                    abs(n1_update[2])
                ])
                output_silhouette = np.array([
                    input_bubbletemp_silhouette[0],
                    input_bubbletemp_silhouette[1],
                    d1_length
                ])

    return output_silhouette, output_face


# ============================================================
# 主计算流程
# ============================================================

class RaytraceProcessor:
    """封装完整的光线追踪计算流程"""

    def __init__(self):
        self.image_camera1 = None
        self.gray_img_camera1 = None
        self.imfill_img_camera1 = None
        self.bubble_silhouette = None
        self.bubble_all_position_face_world = None
        self.bubble_all_face_direction = None
        self.bubble_front_surf = None  # (X, Y, Z)
        self.bubble_back_surf = None
        self.bubble_front_points = None
        self.bubble_back_points = None
        self.tempab = None

    def load_image(self, filepath):
        """加载气泡图像"""
        from skimage import io
        self.image_camera1 = io.imread(filepath)
        if len(self.image_camera1.shape) > 2:
            self.image_camera1 = color.rgb2gray(self.image_camera1)
        self.image_camera1 = img_as_float(self.image_camera1)
        # 裁剪边缘2像素
        h, w = self.image_camera1.shape
        if h > 4 and w > 4:
            self.image_camera1 = self.image_camera1[2:h-2, 2:w-2]
        return self.image_camera1

    def load_image_from_array(self, image_array):
        """从numpy数组加载图像"""
        img = image_array.copy()
        if len(img.shape) > 2:
            img = color.rgb2gray(img)
        img = img_as_float(img)
        h, w = img.shape
        if h > 4 and w > 4:
            img = img[2:h-2, 2:w-2]
        self.image_camera1 = img
        return self.image_camera1

    def run_step1_binary(self):
        """步骤1：二值化+空洞填充"""
        self.imfill_img_camera1 = binary_tempfunc(self.image_camera1)
        self.gray_img_camera1 = self.image_camera1 * self.imfill_img_camera1
        return self.imfill_img_camera1, self.gray_img_camera1

    def run_step2_silhouette(self, theta=90, camera1_angle=45,
                              lengthtocamera1=-1000, face=2,
                              bubble_equivalent_diameter=100):
        """步骤2：Silhouette方法三维重建"""
        gray_img_camera1 = self.gray_img_camera1
        imfill_img_camera1 = self.imfill_img_camera1

        # 获取切片边界
        point1_img_camera1, point2_img_camera1 = slice_position(imfill_img_camera1)
        point1_img_camera2, point2_img_camera2 = slice_position(imfill_img_camera1)

        # 对齐两个相机的气泡高度
        if len(point1_img_camera1) != len(point1_img_camera2):
            disparity = abs(len(point1_img_camera1) - len(point1_img_camera2))
            disparity_up = round(disparity / 2)
            disparity_down = disparity - disparity_up

            if len(point1_img_camera1) > len(point1_img_camera2):
                pad_p1 = np.tile(point1_img_camera2[-1:], (disparity_up, 1))
                pad_p2 = np.tile(point1_img_camera2[-1:], (disparity_down, 1))
                point1_img_camera2 = np.vstack([pad_p1, point1_img_camera2, pad_p2])
                pad_p1 = np.tile(point2_img_camera2[-1:], (disparity_up, 1))
                pad_p2 = np.tile(point2_img_camera2[-1:], (disparity_down, 1))
                point2_img_camera2 = np.vstack([pad_p1, point2_img_camera2, pad_p2])

        # 旋转矩阵
        theta_rad = theta / 180 * np.pi
        R_camera1toworld = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ])
        R_camera2toworld = np.array([
            [np.cos(-theta_rad), 0, -np.sin(-theta_rad)],
            [0, 1, 0],
            [np.sin(-theta_rad), 0, np.cos(-theta_rad)]
        ])

        camera1_rad = camera1_angle / 180 * np.pi
        camera2_rad = (180 - (theta - 90)) / 180 * np.pi

        # 2D→3D投影
        n_slices = len(point1_img_camera1)
        point1_3d_cam1 = np.zeros((n_slices, 3))
        point2_3d_cam1 = np.zeros((n_slices, 3))
        point1_3d_cam2 = np.zeros((n_slices, 3))
        point2_3d_cam2 = np.zeros((n_slices, 3))

        for m in range(n_slices):
            p1_cam1 = np.array([point1_img_camera1[m, 0], point1_img_camera1[m, 1], 1])
            p2_cam1 = np.array([point2_img_camera1[m, 0], point2_img_camera1[m, 1], 1])
            point1_3d_cam1[m] = (R_camera1toworld @ p1_cam1) + [0, 0, lengthtocamera1]
            point2_3d_cam1[m] = (R_camera1toworld @ p2_cam1) + [0, 0, lengthtocamera1]

            a = (point1_img_camera2[m, 0] + point2_img_camera2[m, 0]) / 2
            p1_cam2 = np.array([point1_img_camera2[m, 0] - a, point1_img_camera2[m, 1], 0])
            p1_cam2 = np.append(p1_cam2, 0)[:3]  # ensure 3 elements
            p1_cam2_3d = R_camera2toworld @ np.array([point1_img_camera2[m, 0] - a, point1_img_camera2[m, 1], 0])
            p1_cam2_3d += np.array([a, 0, 0])
            point1_3d_cam2[m] = p1_cam2_3d + [0, 0, lengthtocamera1]

            p2_cam2_3d = R_camera2toworld @ np.array([point2_img_camera2[m, 0] - a, point2_img_camera2[m, 1], 0])
            p2_cam2_3d += np.array([a, 0, 0])
            point2_3d_cam2[m] = p2_cam2_3d + [0, 0, lengthtocamera1]

        # 计算气泡中心
        bubble_center_world = 0.25 * (point1_3d_cam1 + point2_3d_cam1 +
                                       point1_3d_cam2 + point2_3d_cam2)

        d_maxValue = np.array([
            bubble_center_world[0, 0],
            bubble_center_world[0, 1],
            bubble_center_world[0, 2] + bubble_equivalent_diameter
        ])

        # 计算Pa, Pb, Pc, Pd
        P_a = np.zeros((n_slices, 3))
        P_b = np.zeros((n_slices, 3))
        P_c = np.zeros((n_slices, 3))
        P_d = np.zeros((n_slices, 3))

        for m in range(n_slices):
            P_a[m] = slice_camera1and2_position(
                point1_3d_cam1[m], point1_3d_cam2[m], camera1_rad, camera2_rad)
            P_b[m] = slice_camera1and2_position(
                point2_3d_cam1[m], point1_3d_cam2[m], camera1_rad, camera2_rad)
            P_c[m] = slice_camera1and2_position(
                point2_3d_cam1[m], point2_3d_cam2[m], camera1_rad, camera2_rad)
            P_d[m] = slice_camera1and2_position(
                point1_3d_cam1[m], point2_3d_cam2[m], camera1_rad, camera2_rad)

        # Silhouette方法：Bezier曲线构建轮廓
        # 第一遍：计算总点数
        total_points = 0
        for m in range(n_slices):
            xx1, zz1 = bezier_points(point1_3d_cam1[m], P_a[m], point1_3d_cam2[m])
            xx2, zz2 = bezier_points(point1_3d_cam2[m], P_b[m], point2_3d_cam1[m])
            xx3, zz3 = bezier_points(point2_3d_cam2[m], P_c[m], point2_3d_cam1[m])
            xx4, zz4 = bezier_points(point1_3d_cam1[m], P_d[m], point2_3d_cam2[m])
            if face == 1:
                total_points += len(xx1) + len(xx2)
            else:
                total_points += len(xx1) + len(xx2) + len(xx3) + len(xx4)

        bubble_silhouette = np.zeros((total_points, 3))
        tempab = np.zeros((n_slices, 2), dtype=int)
        tempb = 0

        nn_total = point1_img_camera1[-1, 1] + point1_img_camera1[0, 1]

        for m in range(n_slices):
            nn = point1_img_camera1[-1, 1] + point1_img_camera1[0, 1]
            n = nn_total - point1_img_camera1[m, 1]

            xx1, zz1 = bezier_points(point1_3d_cam1[m], P_a[m], point1_3d_cam2[m])
            xx2, zz2 = bezier_points(point1_3d_cam2[m], P_b[m], point2_3d_cam1[m])
            xx3, zz3 = bezier_points(point2_3d_cam2[m], P_c[m], point2_3d_cam1[m])
            xx4, zz4 = bezier_points(point1_3d_cam1[m], P_d[m], point2_3d_cam2[m])

            yy1 = n * np.ones(len(xx1))
            yy2 = n * np.ones(len(xx2))
            yy3 = n * np.ones(len(xx3))
            yy4 = n * np.ones(len(xx4))

            if face == 1:
                temp = np.column_stack([
                    np.concatenate([xx1, xx2]),
                    np.concatenate([yy1, yy2]),
                    np.concatenate([zz1, zz2])
                ])
            else:
                temp = np.column_stack([
                    np.concatenate([xx1, xx2, xx3, xx4]),
                    np.concatenate([yy1, yy2, yy3, yy4]),
                    np.concatenate([zz1, zz2, zz3, zz4])
                ])

            tempa = tempb
            tempb += len(temp)
            tempab[m, 0] = tempa
            tempab[m, 1] = (tempb - tempa) // 2 + tempa
            bubble_silhouette[tempa:tempb] = temp

        self.bubble_silhouette = bubble_silhouette
        self.tempab = tempab
        self.bubble_center_world = bubble_center_world
        self.gray_img_camera1 = gray_img_camera1
        self.imfill_img_camera1 = imfill_img_camera1
        self.point1_img_camera1 = point1_img_camera1

        return bubble_silhouette, tempab

    def run_step3_raytrace_init(self):
        """步骤3：基于灰度的面法向初始化"""
        bubble_silhouette = self.bubble_silhouette
        tempab = self.tempab
        image_01 = self.imfill_img_camera1
        gray_img = self.gray_img_camera1

        bubble_all_position_face_world = np.zeros((len(bubble_silhouette), 6))
        bubble_all_position_face_world[:, :3] = bubble_silhouette
        bubble_all_face_direction = np.zeros((len(bubble_silhouette), 3))

        for n in range(len(tempab)):
            nn = n + int(self.point1_img_camera1[0, 1])
            if nn >= image_01.shape[0]:
                continue

            row_binary = image_01[nn, :]
            ones_idx = np.where(row_binary > 0.5)[0]
            if len(ones_idx) == 0:
                continue

            slice_left = ones_idx[0]
            slice_right = ones_idx[-1]

            row_gray = gray_img[nn, :]
            if slice_right < len(row_gray):
                gray_max = np.max(row_gray)
            else:
                gray_max = np.max(row_gray)

            max_idx = np.argmax(row_gray)
            maxValue_x = max_idx

            # 中心高光点法向
            bubbletemp_silhouette = bubble_silhouette[tempab[n, 0]:tempab[n, 1], :]
            n_points = len(bubbletemp_silhouette)
            bubble_faceinfo = np.zeros((n_points, 3))
            bubble_face_direction = np.zeros((n_points, 3))

            # 右侧
            for u_px in range(max(0, int(slice_right - maxValue_x))):
                u_px += 1
                U_world = int(maxValue_x + (u_px - 1))
                idx = int(U_world - slice_left)
                if idx < 0 or idx >= n_points:
                    continue
                V_world = int(nn)
                if U_world >= 0 and U_world < gray_img.shape[1] and V_world >= 0 and V_world < gray_img.shape[0]:
                    gray_val = gray_img[V_world, U_world]
                else:
                    gray_val = 0
                temp_z_theta = np.pi/2 * (1 - gray_val / max(gray_max, 1e-10))
                bubble_faceinfo[idx] = [0, 0, temp_z_theta]

            for u_px in range(max(0, int(slice_right - maxValue_x))):
                u_px += 1
                U_world = int(maxValue_x + (u_px - 1))
                idx = int(U_world - slice_left)
                if idx < 0 or idx >= n_points:
                    continue
                V_world = int(nn)
                if U_world + 1 < gray_img.shape[1]:
                    if gray_img[V_world, U_world] >= gray_img[V_world, U_world + 1]:
                        bubble_faceinfo[idx, 2] *= -1
                x_dir = -np.sin(bubble_faceinfo[idx, 2])
                z_dir = np.cos(bubble_faceinfo[idx, 2])
                bubble_face_direction[idx] = [x_dir, 0, z_dir]

            # 左侧
            for u_px in range(max(0, int(maxValue_x - slice_left))):
                u_px += 1
                U_world = int(maxValue_x - (u_px - 1))
                idx = int(U_world - slice_left)
                if idx < 0 or idx >= n_points:
                    continue
                V_world = int(nn)
                if U_world >= 0 and U_world < gray_img.shape[1] and V_world >= 0 and V_world < gray_img.shape[0]:
                    gray_val = gray_img[V_world, U_world]
                else:
                    gray_val = 0
                temp_z_theta = np.pi/2 * (1 - gray_val / max(gray_max, 1e-10))
                bubble_faceinfo[idx] = [0, 0, temp_z_theta]

            for u_px in range(max(0, int(maxValue_x - slice_left))):
                u_px += 1
                U_world = int(maxValue_x - (u_px - 1))
                idx = int(U_world - slice_left)
                if idx < 0 or idx >= n_points:
                    continue
                V_world = int(nn)
                if U_world - 1 >= 0 and U_world - 1 < gray_img.shape[1]:
                    if gray_img[V_world, U_world] >= gray_img[V_world, U_world - 1]:
                        bubble_faceinfo[idx, 2] *= -1
                x_dir = np.sin(bubble_faceinfo[idx, 2])
                z_dir = np.cos(bubble_faceinfo[idx, 2])
                bubble_face_direction[idx] = [x_dir, 0, z_dir]

            bubble_all_face_direction[tempab[n, 0]:tempab[n, 1], :] = bubble_face_direction
            bubble_all_position_face_world[tempab[n, 0]:tempab[n, 1], 3:] = bubble_faceinfo

        self.bubble_all_position_face_world = bubble_all_position_face_world
        self.bubble_all_face_direction = bubble_all_face_direction

        return bubble_all_position_face_world, bubble_all_face_direction

    def run_step4_raytrace_adjust(self, progress_callback=None):
        """步骤4：光线追踪迭代修正"""
        bubble_all_pos = self.bubble_all_position_face_world.copy()
        bubble_all_face = self.bubble_all_face_direction.copy()
        tempab = self.tempab
        gray_img = self.gray_img_camera1
        image_01 = self.imfill_img_camera1

        for n in range(len(tempab)):
            if progress_callback:
                progress_callback(int(100 * n / len(tempab)), f"光线追踪修正: 切片 {n+1}/{len(tempab)}")

            nn = n + int(self.point1_img_camera1[0, 1])
            if nn >= image_01.shape[0]:
                continue

            row_binary = image_01[nn, :]
            ones_idx = np.where(row_binary > 0.5)[0]
            if len(ones_idx) == 0:
                continue

            slice_left = ones_idx[0]
            slice_right = ones_idx[-1]

            row_gray = gray_img[nn, :]
            gray_max = np.max(row_gray)
            maxValue_x = np.argmax(row_gray)

            # 当前切片数据
            bubbletemp_silhouette = bubble_all_pos[tempab[n, 0]:tempab[n, 1], :3].copy()
            bubbletemp_face = bubble_all_face[tempab[n, 0]:tempab[n, 1], :].copy()
            thetatemp = bubble_all_pos[tempab[n, 0]:tempab[n, 1], 5].copy()

            n_points = len(bubbletemp_silhouette)
            bubble_position_face = np.zeros((n_points, 6))
            bubble_position_face[:, :3] = bubbletemp_silhouette

            # 中心高光点
            center_idx = max(0, min(maxValue_x - slice_left, n_points-1))
            if (maxValue_x - slice_left) < n_points and (maxValue_x - slice_left) >= 0:
                bubble_position_face[center_idx, 3:6] = [0, 0, 1]

            U_maxValue = bubble_position_face[center_idx, 0]
            V_maxValue = bubble_position_face[center_idx, 1]

            threshold = 10

            # 右侧
            for u_px in range(max(0, slice_right - maxValue_x)):
                u_px += 1
                U_world = int(U_maxValue + (u_px - 1))
                V_world = int(V_maxValue)
                m = int(U_world - slice_left)
                if m < 0 or m >= n_points:
                    continue
                if U_world >= 0 and U_world < gray_img.shape[1] and V_world >= 0 and V_world < gray_img.shape[0]:
                    gray_val = gray_img[V_world, U_world]
                else:
                    gray_val = 0
                temp_z_theta = np.pi/2 * (1 - gray_val / max(gray_max, 1e-10))
                value = threshold * temp_z_theta

                if m < len(thetatemp):
                    if thetatemp[m] > 26.57 * np.pi / 180:
                        bubbletemp_silhouette[m, 2] -= value
                    elif thetatemp[m] < -26.57 * np.pi / 180:
                        bubbletemp_silhouette[m, 2] += value

                input_silhouette = bubbletemp_silhouette[m, :3]
                input_face = bubbletemp_face[m, :]

                out_silhouette, out_face = raytrace_adjust(input_silhouette, input_face)

                bubbletemp_silhouette[m, 2] = out_silhouette[2]
                bubbletemp_face[m, :] = out_face
                bubble_position_face[m, :3] = bubbletemp_silhouette[m, :3]
                bubble_position_face[m, 3:6] = bubbletemp_face[m, :]

            # 左侧
            for u_px in range(max(0, maxValue_x - slice_left)):
                u_px += 1
                U_world = int(U_maxValue - (u_px - 1))
                V_world = int(V_maxValue)
                m = int(U_world - slice_left)
                if m < 0 or m >= n_points:
                    continue
                if U_world >= 0 and U_world < gray_img.shape[1] and V_world >= 0 and V_world < gray_img.shape[0]:
                    gray_val = gray_img[V_world, U_world]
                else:
                    gray_val = 0
                temp_z_theta = np.pi/2 * (1 - gray_val / max(gray_max, 1e-10))
                value = threshold * temp_z_theta

                if m < len(thetatemp):
                    if thetatemp[m] > 26.57 * np.pi / 180:
                        bubbletemp_silhouette[m, 2] -= value
                    elif thetatemp[m] < -26.57 * np.pi / 180:
                        bubbletemp_silhouette[m, 2] += value

                input_silhouette = bubbletemp_silhouette[m, :3]
                input_face = bubbletemp_face[m, :]

                out_silhouette, out_face = raytrace_adjust(input_silhouette, input_face)

                bubbletemp_silhouette[m, 2] = out_silhouette[2]
                bubbletemp_face[m, :] = out_face
                bubble_position_face[m, :3] = bubbletemp_silhouette[m, :3]
                bubble_position_face[m, 3:6] = bubbletemp_face[m, :]

            bubble_all_pos[tempab[n, 0]:tempab[n, 1], :] = bubble_position_face

        self.bubble_all_position_face_world = bubble_all_pos

        if progress_callback:
            progress_callback(100, "光线追踪修正完成")

        return bubble_all_pos

    def run_step5_visualize(self):
        """步骤5：分离前后表面并构建曲面"""
        bubble_all_pos = self.bubble_all_position_face_world
        tempab = self.tempab

        n_total_half = len(bubble_all_pos) // 2
        bubble_front = np.zeros((n_total_half, 3))
        bubble_back = np.zeros((n_total_half, 3))

        for n in range(len(tempab)):
            a = tempab[n, 1] - tempab[n, 0]
            b = (tempab[n, 0] + 1) // 2
            if b + a - 1 < n_total_half:
                bubble_front[b:b+a-1] = bubble_all_pos[tempab[n, 0]:tempab[n, 1]-1, :3]
                end_idx = min(tempab[n, 1] + a - 1, len(bubble_all_pos))
                if tempab[n, 1] < end_idx:
                    bubble_back[b:b+a-1] = bubble_all_pos[tempab[n, 1]:end_idx, :3]

        self.bubble_front_points = bubble_front
        self.bubble_back_points = bubble_back

        # 构建正面曲面
        countmax = int(np.max(tempab[:, 1] - tempab[:, 0]))

        # 按Y值分组
        unique_y = np.unique(bubble_front[:, 1])
        n_rows = len(unique_y)

        bubble_X = np.full((n_rows, countmax), np.nan)
        bubble_Y = np.full((n_rows, countmax), np.nan)
        bubble_Z = np.full((n_rows, countmax), np.nan)

        for i, y_val in enumerate(sorted(unique_y, reverse=True)):
            mask = bubble_front[:, 1] == y_val
            pts = bubble_front[mask]
            pts = pts[np.argsort(pts[:, 0])]

            x = pts[:, 0]
            y = pts[:, 1]
            z = pts[:, 2]

            if len(x) > 3:
                x_interp = np.linspace(0, 1, countmax)
                try:
                    f_x = interp1d(np.linspace(0, 1, len(x)), x, kind='cubic')
                    f_y = interp1d(np.linspace(0, 1, len(y)), y, kind='cubic')
                    f_z = interp1d(np.linspace(0, 1, len(z)), z, kind='cubic')
                    bubble_X[i] = f_x(x_interp)
                    bubble_Y[i] = f_y(x_interp)
                    bubble_Z[i] = f_z(x_interp)
                except Exception:
                    bubble_X[i, :len(x)] = x
                    bubble_Y[i, :len(y)] = y
                    bubble_Z[i, :len(z)] = z
            else:
                bubble_X[i, :len(x)] = x
                bubble_Y[i, :len(y)] = y
                bubble_Z[i, :len(z)] = z

        self.bubble_front_surf = (bubble_X, bubble_Y, bubble_Z)

        # 构建背面曲面
        unique_y_back = np.unique(bubble_back[:, 1])
        n_rows_back = len(unique_y_back)
        countmax_back = int(np.max(tempab[:, 1] - tempab[:, 0]))

        bubble_Xb = np.full((n_rows_back, countmax_back), np.nan)
        bubble_Yb = np.full((n_rows_back, countmax_back), np.nan)
        bubble_Zb = np.full((n_rows_back, countmax_back), np.nan)

        for i, y_val in enumerate(sorted(unique_y_back, reverse=True)):
            mask = bubble_back[:, 1] == y_val
            pts = bubble_back[mask]
            pts = pts[np.argsort(pts[:, 0])]

            x = pts[:, 0]
            y = pts[:, 1]
            z = pts[:, 2]

            if len(x) > 3:
                x_interp = np.linspace(0, 1, countmax_back)
                try:
                    f_x = interp1d(np.linspace(0, 1, len(x)), x, kind='cubic')
                    f_y = interp1d(np.linspace(0, 1, len(y)), y, kind='cubic')
                    f_z = interp1d(np.linspace(0, 1, len(z)), z, kind='cubic')
                    bubble_Xb[i] = f_x(x_interp)
                    bubble_Yb[i] = f_y(x_interp)
                    bubble_Zb[i] = f_z(x_interp)
                except Exception:
                    bubble_Xb[i, :len(x)] = x
                    bubble_Yb[i, :len(y)] = y
                    bubble_Zb[i, :len(z)] = z
            else:
                bubble_Xb[i, :len(x)] = x
                bubble_Yb[i, :len(y)] = y
                bubble_Zb[i, :len(z)] = z

        self.bubble_back_surf = (bubble_Xb, bubble_Yb, bubble_Zb)

        return bubble_front, bubble_back
