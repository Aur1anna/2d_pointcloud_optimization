import os
import sys
import torch
from PIL import Image
import numpy
from tqdm import tqdm
import time
from point_cloud_2d import point_cloud_load

def gaussian_kernel(dx, dy, sigma):
    """计算高斯核权重"""
    distance_squared = dx**2 + dy**2
    return torch.exp(-distance_squared / (0.5 * sigma**2))


def render_point_cloud_2d(points, radii, colors, alpha, image):
    """渲染二维点云到圆形范围"""
    height, width, _ = image.shape
    grid_x, grid_y = torch.meshgrid(
        torch.arange(0, width, device='cuda'),
        torch.arange(0, height, device='cuda'),
        indexing='xy'
    )
    grid_x = grid_x.T
    grid_y = grid_y.T

    # 初始化更新张量，避免直接修改 `image`
    image_update = torch.zeros_like(image, device='cuda')

    # 遍历点云
    for i in range(points.shape[0]):
        # 获取点的信息
        cx, cy = points[i]
        r = radii[i]
        color = colors[i]
        a = alpha[i]

        # 计算该点的影响范围
        dx = grid_x - cx
        dy = grid_y - cy
        distance_squared = dx**2 + dy**2

        # 筛选圆形范围内的像素
        mask = distance_squared <= r**2
        weight = gaussian_kernel(dx[mask], dy[mask], r)

        # 获取 mask 内的点索引
        mask_indices = mask.nonzero(as_tuple=True)

         # 创建更新张量
        temp_update = torch.zeros_like(image, device='cuda')
        
        # 更新 RGB 通道
        temp_update[mask_indices[0], mask_indices[1], :3] = (
            weight.unsqueeze(-1) * color * a
        )

        # 更新 Alpha 通道
        temp_update[mask_indices[0], mask_indices[1], 3] = weight * a

        # 累加到 `image_update`
        image_update += temp_update
    
    # 将更新应用到 `image`
    image += image_update    
    return image

def normalize_image(image):
    """归一化图像，避免透明度溢出"""
    image[..., :3] = image[..., :3] / (image[..., 3:4] + 1e-8)
    image[..., 3] = torch.clamp(image[..., 3], 0, 1)
    return image



def render_and_save(input_dir,output_dir):
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA version:", torch.version.cuda)
    
    # 初始化点云和画布
    height, width = 512, 512
    
    points, radii, colors, alpha = point_cloud_load(input_dir)
    
    if points is None:
        print("Failed to load point cloud data.")
        return
    
    # 初始化白底 RGBA 图像
    image = torch.ones((height, width, 4), dtype=torch.float32, device='cuda') 
    
    start_time = time.time()  # 开始时间
    
    print("Rendering...")
    num_points = points.shape[0]
    for i in tqdm(range(num_points), desc='Rendering Progress', unit='point'):
        rendered_image = render_point_cloud_2d(points[i:i+1], radii[i:i+1], colors[i:i+1], alpha[i:i+1], image)
        
        # 更新已用时间和预计剩余时间
        elapsed_time = time.time() - start_time
        avg_time_per_point = elapsed_time / (i + 1)
        remaining_time = avg_time_per_point * (num_points - (i + 1))
        # tqdm.write(f'Elapsed: {elapsed_time:.2f}s, Estimated Remaining: {remaining_time:.2f}s')

    # 归一化图像
    rendered_image = normalize_image(rendered_image)
    print("Rendered")

    # 可视化结果
    rendered_image_np = (rendered_image[..., :3] * 255).byte().cpu().numpy()
    temp_file = os.path.join(output_dir, "rendered_image.png")
    Image.fromarray(rendered_image_np).save(temp_file)
    print(f"Rendered image saved to {temp_file}")
    
    


if __name__ == "__main__":

    print("CUDA available:", torch.cuda.is_available())
    print("CUDA version:", torch.version.cuda)
    print("init")
    """
    # 初始化点云和画布
    height, width = 512, 512
    points = torch.tensor([[100, 200], [300, 400]], dtype=torch.float32, device='cuda')  # 点的位置
    radii = torch.tensor([20, 30], dtype=torch.float32, device='cuda')  # 点的半径
    colors = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32, device='cuda')  # 点的颜色
    alpha = torch.tensor([0.8, 0.6], dtype=torch.float32, device='cuda')  # 点的透明度
    image = torch.ones((height, width, 4), dtype=torch.float32, device='cuda')  # 初始化 RGBA 图像
    """
    # 初始化点云和画布
    height, width = 512, 512
    # 扩展后的点的位置
    points = torch.tensor([
        [358, 200], [300, 400], 
        [150, 250], [100, 450],
        [120, 220], [320, 420],
        [289, 270], [370, 270],
        [140, 240], [340, 440],
        [160, 360], [360, 100],
        [130, 230], [150, 430],
        [180, 350], [54, 480],
        [190, 290], [390, 490],
        [110, 210], [130, 290],
        [30, 215], [315, 415]
    ], dtype=torch.float32, device='cuda')
    # 扩展后的点的半径
    radii = torch.tensor([
        20, 30, # 原始的两个半径
        25, 35,
        22, 32,
        27, 37,
        200, 34,
        26, 36,
        23, 33,
        28, 38,
        29, 39,
        21, 31,
        21.5, 31.5
    ], dtype=torch.float32, device='cuda')
    # 扩展后的点的颜色
    colors = torch.tensor([
        [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], # 原始的两种颜色
        [0.0, 0.0, 1.0], [1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0], [0.0, 1.0, 1.0],
        [0.5, 0.0, 0.0], [0.0, 0.5, 0.0],
        [0.0, 0.0, 0.5], [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5], [0.0, 0.5, 0.5],
        [0.2, 0.2, 0.2], [0.8, 0.8, 0.8],
        [0.7, 0.3, 0.0], [0.0, 0.7, 0.3],
        [0.3, 0.0, 0.7], [0.7, 0.0, 0.3],
        [0.3, 0.7, 0.0], [0.0, 0.3, 0.7],
        [0.05, 0.723, 0.54], [0.25, 0.01, 0.425]
    ], dtype=torch.float32, device='cuda')
    # 扩展后的点的透明度
    alpha = torch.tensor([
        0.8, 0.6, # 原始的两个透明度
        0.7, 0.5,
        0.85, 0.65,
        0.75, 0.55,
        1, 0.62,
        0.72, 0.52,
        0.87, 0.67,
        0.77, 0.57,
        0.88, 0.68,
        0.81, 0.61,
        0.79, 0.69
    ], dtype=torch.float32, device='cuda')
    
    # 初始化白底 RGBA 图像
    image = torch.ones((height, width, 4), dtype=torch.float32, device='cuda') 
    """
    image[:, :, :3] = 1.0  # 设置RGB通道为1，表示白色
    image[:, :, 3] = 0.01   # 设置Alpha通道为0.01，表示完全透明，背景完全透明时渲染会完全失真
    """
    



    # 渲染点云
    """
    print("rendering")
    rendered_image = render_point_cloud_2d(points, radii, colors, alpha, image)
    """
    start_time = time.time()  # 开始时间
    # 渲染点云并显示进度条
    print("rendering")
    num_points = points.shape[0]
    for i in tqdm(range(num_points), desc='Rendering Progress', unit='point'):
        rendered_image = render_point_cloud_2d(points[i:i+1], radii[i:i+1], colors[i:i+1], alpha[i:i+1], image)

        # 更新已用时间和预计剩余时间
        elapsed_time = time.time() - start_time
        avg_time_per_point = elapsed_time / (i + 1)
        remaining_time = avg_time_per_point * (num_points - (i + 1))
        #tqdm.write(f'Elapsed: {elapsed_time:.2f}s, Estimated Remaining: {remaining_time:.2f}s')

    
    # 归一化图像
    rendered_image = normalize_image(rendered_image)
    print("rendered")

    # 可视化结果
    rendered_image = (image[..., :3] * 255).byte().cpu().numpy()
    temp_file = "temp_image.png"
    Image.fromarray(rendered_image).save(temp_file)
    print("手动打开 temp_file 查看渲染图像")
