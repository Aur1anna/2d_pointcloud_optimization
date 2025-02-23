import os
import cv2
import torch
import numpy as np
from point_cloud_2d import point_cloud_save

def generate_initial_points(target_image_path, num_points=300, edge_weight=3.0):
    """
    基于目标图像生成初始点云
    参数：
    - target_image_path: 目标图像路径
    - num_points: 总点数基数（实际点数会根据边缘密度调整）
    - edge_weight: 边缘区域采样权重
    返回：位置、半径、颜色、透明度张量
    """
    # 读取并处理图像
    img = cv2.imread(target_image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    
    # 边缘检测
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    
    # 生成采样概率图
    edge_mask = (edges > 0).astype(float) * edge_weight + 1.0
    prob_map = cv2.GaussianBlur(edge_mask, (21, 21), 0)
    prob_map /= prob_map.sum()
    
    # 基于概率图采样坐标
    flat_indices = np.random.choice(
        h*w, size=int(num_points * (1 + edge_weight)), 
        p=prob_map.flatten()
    )
    y_coords, x_coords = np.unravel_index(flat_indices, (h, w))
    
    # 转换为标准化坐标
    points = np.column_stack((
        x_coords * (512.0 / w),  # 缩放至512x512坐标系
        y_coords * (512.0 / h)
    ))
    
    # 初始化参数
    radii = np.random.uniform(1, 100, size=len(points))
    colors = img[y_coords, x_coords] / 255.0  # 采样对应位置颜色
    # colors = np.full((len(points), 3), 0.5, dtype=np.float32)  #颜色统一0.5
    alpha = np.full(len(points), 0.4)  # 透明度统一0.9
    
    # 转换为CUDA Tensor
    return (
        torch.tensor(points, dtype=torch.float32, device='cuda'),
        torch.tensor(radii, dtype=torch.float32, device='cuda'),
        torch.tensor(colors, dtype=torch.float32, device='cuda'),
        torch.tensor(alpha, dtype=torch.float32, device='cuda')
    )

def random_init(target_image_path, output_dir):
    # 生成初始化数据
    points, radii, colors, alpha = generate_initial_points(target_image_path)
    
    # 保存点云
    point_cloud_save(points, radii, colors, alpha, output_dir)
    print(f"Random initialized point cloud saved to {output_dir}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate random initialized point cloud')
    parser.add_argument('target_image', type=str, help='Path to target image')
    parser.add_argument('--output_dir', type=str, default='./', help='Output directory')
    
    args = parser.parse_args()
    
    # 检查目标图像存在
    if not os.path.exists(args.target_image):
        raise FileNotFoundError(f"Target image {args.target_image} not found")
    
    # 执行初始化
    random_init(args.target_image, args.output_dir)