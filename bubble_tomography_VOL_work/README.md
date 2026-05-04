# 气泡三维多相机层析重建系统

## 功能概述

本程序实现了基于多相机视角的气泡三维层析重建完整流程：

1. **多相机标定** — 支持棋盘格、对称点阵、非对称点阵、体标定板点阵，3~N个相机联合标定
2. **气泡图像预处理** — 背景去除、去畸变、二值化分割、投影计算
3. **MART层析重建** — 基于光线追踪的乘法代数重建算法（Multiplicative ART）
4. **三维点云输出** — PLY/PCD/OBJ格式导出，支持CloudCompare、MeshLab等软件
5. **示踪粒子3D重建** — 多视角三角测量 + 外极线约束匹配
6. **Tomographic PIV** — 3D互相关速度场计算
7. **批量时间序列处理** — 支持多时刻图像批量加载、重建和速度场计算

## 目录结构

```
bubble_tomography/
├── main.py                          # 主入口（GUI/Demo/PIV-Demo）
├── __init__.py
├── requirements.txt                 # Python依赖
├── calibration/                     # 相机标定模块
│   ├── __init__.py
│   └── camera_calibrator.py         # 多相机标定器
├── mart/                            # MART重建模块
│   ├── __init__.py
│   └── mart_reconstructor.py        # MART算法 + 光线追踪器
├── utils/                           # 工具模块
│   ├── __init__.py
│   └── image_processor.py           # 气泡图像预处理器
├── visualization/                   # 可视化模块
│   ├── __init__.py
│   └── visualizer.py                # 3D可视化与点云导出
├── particles/                       # 粒子追踪模块
│   ├── __init__.py
│   ├── particle_reconstructor.py    # 粒子3D重建
│   └── velocity_field.py            # 3D速度场计算
├── gui/                             # GUI界面
│   ├── __init__.py
│   └── main_window.py               # PyQt5主窗口（含批量处理）
└── demo_output/                     # 演示输出
```

## 快速开始

### 1. 环境依赖

```bash
pip install -r requirements.txt
```

或手动安装：
```bash
pip install numpy opencv-python scipy matplotlib scikit-image tqdm pyqt5
```

### 2. 运行演示

```bash
cd bubble_tomography

# 气泡重建演示
python main.py --demo

# Tomographic PIV演示
python main.py --piv-demo
```

### 3. 启动GUI

```bash
python main.py --gui
```

GUI界面提供四个标签页：
- **1. 相机标定**：添加相机、加载标定图像、设置标定板参数、执行标定
- **2. 气泡重建**：批量加载气泡图像序列、设置MART参数、执行重建、时间点切换查看
- **3. 结果可视化**：查看点云、体素切片、投影对比、综合报告、批量结果概览
- **4. Particle Tracking / PIV**：批量加载粒子图像、粒子3D重建、速度场计算、时间点切换

### 4. 批量时间序列处理

GUI支持从文件夹根目录批量加载多时刻图像：

```
root_dir/
├── t000/
│   ├── cam1.png
│   ├── cam2.png
│   └── cam3.png
├── t001/
│   ├── cam1.png
│   ├── cam2.png
│   └── cam3.png
└── t002/
    ├── cam1.png
    ├── cam2.png
    └── cam3.png
```

- 每个子文件夹对应一个时间点，文件夹名作为时间标识
- 子文件夹内图像名需包含对应相机ID（如 `cam1.png`）以自动匹配
- 右侧面板底部有**时间点滑块**，可快速切换查看各时刻的处理结果

## 快速开始

### 1. 环境依赖

```bash
pip install numpy opencv-python scipy matplotlib scikit-image tqdm pyqt5
```

### 2. 运行演示

```bash
cd bubble_tomography
python main.py --demo
```

演示将生成4个模拟相机的标定参数、合成气泡投影、执行MART重建，输出结果到 `demo_output/`。

### 3. 启动GUI

```bash
python main.py --gui
```

GUI界面提供三个标签页：
- **相机标定**：添加相机、加载标定图像、设置标定板参数、执行标定
- **气泡重建**：加载气泡图像和背景图、设置MART参数、执行重建
- **结果可视化**：查看点云、体素切片、投影对比、综合报告

## 使用流程

### Step 1: 相机标定

#### 命令行方式

```python
from calibration import MultiCameraCalibrator

# 创建标定器
calibrator = MultiCameraCalibrator(
    pattern_type='checkerboard',   # 棋盘格
    pattern_size=(11, 8),          # 内角点数 (宽 x 高)
    square_size=5.0                # 方格边长 (mm)
)

# 逐相机标定
camera_images = {
    'cam1': ['cam1_img1.jpg', 'cam1_img2.jpg', ...],  # 每个相机至少3张
    'cam2': ['cam2_img1.jpg', 'cam2_img2.jpg', ...],
    'cam3': ['cam3_img1.jpg', 'cam3_img2.jpg', ...],
}

results = calibrator.calibrate_multi_camera(camera_images)

# 保存结果
calibrator.save_results('./calibration_output')

# 查看报告
print(calibrator.get_calibration_report())
```

支持的标定板类型：
| 类型 | 参数值 | 说明 |
|------|--------|------|
| 棋盘格 | `'checkerboard'` | 最常用，黑白交替方格 |
| 对称圆点阵 | `'circles'` | 规则排列的圆形 |
| 非对称圆点阵 | `'acircles'` | 交错排列的圆形，精度更高 |
| 体标定板点阵 | `'volume_dots'` | 亮圆点体标定板，支持规则点阵中存在少量缺失/编码点 |

#### 标定图像拍摄建议
- 每个相机至少拍摄 **5-10张** 不同角度/距离的标定板图像
- 标定板应覆盖图像的不同区域（中心、边缘）
- 保持标定板平整，无反光
- 图像应清晰对焦

### Step 2: 气泡图像预处理

```python
from utils import BubbleImageProcessor
import cv2

processor = BubbleImageProcessor(
    background_method='reference',  # 使用无气泡参考图
    threshold_method='otsu',        # 自适应阈值
    morph_operations=True           # 形态学去噪
)

# 加载图像
bubble_images = {
    'cam1': cv2.imread('bubble_cam1.png'),
    'cam2': cv2.imread('bubble_cam2.png'),
    'cam3': cv2.imread('bubble_cam3.png'),
}

# 加载背景参考图（可选但推荐）
reference_images = {
    'cam1': cv2.imread('background_cam1.png'),
    'cam2': cv2.imread('background_cam2.png'),
    'cam3': cv2.imread('background_cam3.png'),
}

# 准备投影数据
projections = processor.prepare_projection_data(
    bubble_images,
    {cid: {'camera_matrix': p.camera_matrix, 'dist_coeffs': p.dist_coeffs}
     for cid, p in calibrator.camera_params.items()},
    reference_images=reference_images,
    projection_type='soft_edge'  # 柔化边缘投影
)
```

### Step 3: MART层析重建

```python
from mart import MARTReconstructor, MARTConfig

# 配置重建参数
config = MARTConfig(
    grid_size=(64, 64, 64),       # 重建网格分辨率
    domain_size=(20, 20, 20),      # 重建域尺寸 (mm)
    relaxation_factor=0.5,          # 松弛因子 (0-1)
    max_iterations=50,              # 最大迭代次数
    voxel_threshold=0.1,            # 体素提取阈值
    ray_sample_step=0.2             # 光线采样步长 (mm)
)

reconstructor = MARTReconstructor(config)

# 准备相机投影矩阵
camera_params_recon = {}
for cam_id, params in calibrator.camera_params.items():
    P = calibrator.compute_projection_matrix(cam_id)
    K = np.array(params.camera_matrix)
    camera_params_recon[cam_id] = {
        'P': P,
        'K_inv': np.linalg.inv(K)
    }

# 执行重建
volume = reconstructor.reconstruct(projections, camera_params_recon)

# 提取气泡点云（基于Marching Cubes）
points, normals = reconstructor.extract_bubble_point_cloud()

print(f"重建体素场: {volume.shape}")
print(f"点云点数: {len(points)}")
```

### Step 4: 结果输出

```python
from visualization import ResultVisualizer

viz = ResultVisualizer(output_dir='./results')

# 3D点云可视化
viz.plot_point_cloud(points, normals)

# 体素切片
viz.plot_volume_slices(volume, axis='z', num_slices=5)

# 投影对比
viz.plot_projection_comparison(projections)

# 综合报告
viz.create_report_figure(volume, points, projections,
                          reconstructor.get_volume_stats())

# 导出点云文件
viz.save_point_cloud_ply(points, normals, 'bubble.ply')   # PLY格式
viz.save_point_cloud_pcd(points, 'bubble.pcd')             # PCD格式
viz.save_point_cloud_obj(points, normals, 'bubble.obj')    # OBJ格式
viz.save_volume_npy(volume, 'volume.npy')                  # 体素数据
```

## MART算法参数调优指南

| 参数 | 默认值 | 说明 | 调优建议 |
|------|--------|------|---------|
| `grid_size` | (64,64,64) | 三维网格分辨率 | 增大提高精度但计算量立方增长 |
| `domain_size` | (20,20,20) | 重建域物理尺寸(mm) | 应略大于气泡群的实际范围 |
| `relaxation_factor` | 0.5 | 松弛因子μ | 0.1~0.3更稳定但收敛慢；0.5~0.8更快但可能发散 |
| `max_iterations` | 50 | 最大迭代次数 | 通常20~50次足够收敛 |
| `voxel_threshold` | 0.1 | 表面提取阈值 | 根据重建值分布调整 |
| `ray_sample_step` | 0.2 | 光线采样步长(mm) | 越小越精确但越慢 |

## 输出文件格式

| 格式 | 扩展名 | 兼容软件 |
|------|--------|---------|
| PLY | `.ply` | MeshLab, CloudCompare, ParaView |
| PCD | `.pcd` | CloudCompare, PCL Viewer |
| OBJ | `.obj` | Blender, MeshLab |
| NPY | `.npy` | Python (numpy.load) |

## 常见问题

**Q: 标定重投影误差太大怎么办？**
A: 确保标定图像清晰、标定板平整、角点被完整检测。尝试增加标定图像数量（10+张），覆盖图像不同区域。

**Q: 重建结果出现伪影？**
A: 降低松弛因子（0.1~0.3），增加迭代次数，减小光线采样步长，确保相机角度覆盖足够（建议相邻相机间隔≤45°）。

**Q: 重建速度太慢？**
A: 减小网格分辨率（如32³），增大光线采样步长，减少迭代次数。MART计算复杂度为O(N_rays × N_voxels × N_iterations)。

## 引用

如果本程序对您的研究有帮助，请引用：

> MART算法: Gordon, R., Bender, R., & Herman, G. T. (1970). Algebraic reconstruction techniques (ART) for three-dimensional electron microscopy and X-ray photography. Journal of Theoretical Biology, 29(3), 471-481.

## 许可

MIT License
