

# 2d_pointcloud_optimization

#### 可微分渲染：

1. **渲染逻辑**：

   - 将一组点云数据渲染成一张图像。

   - 渲染过程包括计算每个点对图像的贡献，并将其累积到画布上，最终生成一个RGB图像。

2. **主要参数**：
   - 点云参数：`points`, `radii`, `colors`, `alpha`
   - 高斯权重：透明度权重随距离增加而减小
   - 颜色贡献：计算每个像素的 RGB 贡献 `rgb_contrib` 和 `Alpha` 贡献
   - 距离场掩码：二值掩码 `mask`表示像素在当前点的影响范围

#### 点云随机初始化：

1. **点云生成逻辑**：

   - 基于边缘密度进行概率采样（边缘区域采样概率提高3倍）
   - 自动适配不同尺寸输入图像（缩放至512x512，推荐使用512*512输入）
   
2. **参数初始化策略**：

   - **位置**：优先分布在图像边缘区域，保留结构特征
   - **半径**：可调的随机像素，兼顾覆盖范围和细节保留
   - **颜色**：直接采样对应像素位置颜色值或统一初始化为固定值
   - **透明度**：统一初始化为固定值

3. **点数计算**：

   ```
   实际点数 = 基数点数 * (1 + 边缘权重)
   ```

   例如当num_points=300，edge_weight=3时，实际生成约1200个点

#### 训练：

1. **优化器**：
   - Adam优化器。

2. **损失函数**：
   - SSIM和L1损失联合，lamuda为0.8/0.2；（SSIM loss=1-SSIM）

## 环境搭建

```shell
 git clone https://github.com/Aur1anna/2d_pointcloud_optimization.git
 
 cd 2d_pointcloud_optimization
 
 pip install -r requirements.txt
 # or
 conda env create -f environments.yml
```

## 如何使用

```shell
# 首先将target_image.png放在项目地址，只接受512*512尺寸

# 生成初始化点云（示例）
python random_init.py target_image.png --output_dir ./init_cloud
## 在目标文件夹下生成一个2d_point_cloud.npz

# 查看生成结果
python point_cloud_2d.py check ./init_cloud
#或直接渲染点云到图片：
python rendering.py /input /output
## rendering.py只在 /input 路径下寻找2d_point_cloud.npz或optimized.npz，否则报错

# 进行优化训练
python train_optimizer.py --init_path ./init_cloud  
##--init_path默认为train_optimizer.py文件夹下的2d_point_cloud.npz
```
