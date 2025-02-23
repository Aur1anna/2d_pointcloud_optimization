import os
import torch
import numpy as np
from torch.optim import Adam
from PIL import Image
from pytorch_msssim import SSIM
from point_cloud_2d import point_cloud_load

# 点云文件加载函数（来自point_cloud_2d.py）
'''
def point_cloud_load(data_dir):
    file_path = os.path.join(data_dir, '2d_point_cloud.npz')
    data = np.load(file_path)
    points = torch.from_numpy(data['points']).to(dtype=torch.float32, device='cuda')
    radii = torch.from_numpy(data['radii']).to(dtype=torch.float32, device='cuda')
    colors = torch.from_numpy(data['colors']).to(dtype=torch.float32, device='cuda')
    alpha = torch.from_numpy(data['alpha']).to(dtype=torch.float32, device='cuda')
    print(f"\n{len(points)} points found.")
    return points, radii, colors, alpha
'''

# 修正后的可微渲染函数（输出RGB三通道）
def differentiable_render(points, radii, colors, alpha, height=512, width=512):
    canvas = torch.zeros((height, width, 4), dtype=torch.float32, device='cuda')
    
    # 创建坐标网格（优化内存布局）
    y_coords, x_coords = torch.meshgrid(
        torch.arange(height, device='cuda'),
        torch.arange(width, device='cuda'),
        indexing='ij'
    )
    
    # 矢量化的贡献计算
    for i in range(points.shape[0]):
        cx, cy = points[i]
        r = radii[i]
        color = colors[i]
        a = alpha[i]
        
        # 距离场计算
        dx = x_coords - cx
        dy = y_coords - cy
        dist_sq = dx**2 + dy**2
        
        # 高斯权重（保持计算图）
        weight = torch.exp(-dist_sq / (0.5 * r**2 + 1e-8))
        mask = (dist_sq <= r**2).float()
        
        # 计算各通道贡献
        rgb_contrib = weight.unsqueeze(-1) * color * a * mask.unsqueeze(-1)
        alpha_contrib = weight * a * mask
        
        # 累积到画布（避免原地操作）
        canvas = canvas + torch.cat([rgb_contrib, alpha_contrib.unsqueeze(-1)], dim=-1)
    
    # 混合计算（仅输出RGB）
    rgb = canvas[..., :3] / (canvas[..., 3:] + 1e-8)
    return rgb.permute(2, 0, 1).unsqueeze(0)  # 输出形状 [1, 3, 512, 512]

# 加载目标图像（确保RGB三通道）
def load_target_image(image_path, device='cuda'):
    img = Image.open(image_path).convert('RGB').resize((512, 512))
    img_tensor = torch.tensor(np.array(img)/255.0, dtype=torch.float32).to(device)
    return img_tensor.permute(2, 0, 1).unsqueeze(0)  # [1, 3, 512, 512]

# 训练参数配置
num_epochs = 50
learning_rate = 0.1

# 初始化点云参数（确保已生成初始数据）
points, init_radii, init_colors, init_alpha = point_cloud_load(os.path.dirname(__file__))

# 创建可训练参数（显式声明）
radii = torch.nn.Parameter(init_radii.clone())
colors = torch.nn.Parameter(init_colors.clone())
alpha = torch.nn.Parameter(init_alpha.clone())

# 加载目标图像（替换为你的图片路径）
target_image = load_target_image("target_image.png")

# 初始化优化器和损失函数
optimizer = Adam([radii, colors, alpha], lr=learning_rate)
ssim_loss_fn = SSIM(data_range=1.0, size_average=True, channel=3)
l1_loss_fn = torch.nn.L1Loss()

# 训练循环
for epoch in range(num_epochs):
    optimizer.zero_grad()
    
    # 前向传播
    rendered = differentiable_render(points, radii, colors, alpha)
    
    # 损失计算（确保输入维度匹配）
    ssim_loss = 1 - ssim_loss_fn(rendered, target_image)
    l1_loss = l1_loss_fn(rendered, target_image)
    total_loss = 0.8*ssim_loss + 0.2*l1_loss
    
    # 反向传播
    total_loss.backward()
    
    # 梯度裁剪（防止NaN）
    torch.nn.utils.clip_grad_norm_([radii, colors, alpha], 1.0)
    
    # 参数更新
    optimizer.step()
    
    # 物理约束（非原地操作）
    with torch.no_grad():
        radii.data = radii.data.clamp(1.0, 100.0)
        colors.data = colors.data.clamp(0.0, 1.0)
        alpha.data = alpha.data.clamp(0.01, 1.0)
    
    # 训练监控
    if (epoch+1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {total_loss.item():.4f} "
              f"(SSIM: {ssim_loss.item():.4f}, L1: {l1_loss.item():.4f})")

# 保存优化结果
def save_optimized_params(points, radii, colors, alpha, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    np.savez(os.path.join(save_dir, 'optimized.npz'),
             points=points.cpu().numpy(),
             radii=radii.detach().cpu().numpy(),
             colors=colors.detach().cpu().numpy(),
             alpha=alpha.detach().cpu().numpy())
    print(f"Optimized parameters saved to {save_dir}")

save_optimized_params(points, radii, colors, alpha, "optimized_results")