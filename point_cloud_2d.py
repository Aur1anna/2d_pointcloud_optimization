import os
import sys
import torch
import numpy as np

def point_cloud_save(points,radii,colors,alpha,output_dir): 
    # 保存点云数据，传入四个参数矩阵和保存路径
    points_np = points.cpu().numpy()
    radii_np = radii.cpu().numpy()
    colors_np = colors.cpu().numpy()
    alpha_np = alpha.cpu().numpy()
    # 构建完整的文件路径
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    file_path = os.path.join(output_dir, '2d_point_cloud.npz')
    # 保存：
    np.savez(file_path, points=points_np, radii=radii_np, colors=colors_np, alpha=alpha_np)
    print(f"2d_point_cloud.npz saved at {file_path}")


def point_cloud_load(data_dir):
    # 尝试加载 '2d_point_cloud.npz'
    file_path = os.path.join(data_dir, '2d_point_cloud.npz')
    
    if os.path.exists(file_path):
        try:
            data = np.load(file_path)
            if 'points' in data and 'radii' in data and 'colors' in data and 'alpha' in data:
                print(f"Loaded from {file_path}")
            else:
                print(f"File {file_path} does not contain the required keys.")
                return None
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    else:
        # 如果 '2d_point_cloud.npz' 不存在，则尝试加载 'optimized.npz'
        file_path = os.path.join(data_dir, 'optimized.npz')
        if os.path.exists(file_path):
            try:
                data = np.load(file_path)
                if 'points' in data and 'radii' in data and 'colors' in data and 'alpha' in data:
                    print(f"Loaded from {file_path}")
                else:
                    print(f"File {file_path} does not contain the required keys.")
                    return None
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                return None
        else:
            print(f"NO '2d_point_cloud.npz' or 'optimized.npz' FOUND AT {data_dir}")
            return None

    # 转换回 Tensor
    points = torch.from_numpy(data['points']).to(dtype=torch.float32, device='cuda')
    radii = torch.from_numpy(data['radii']).to(dtype=torch.float32, device='cuda')
    colors = torch.from_numpy(data['colors']).to(dtype=torch.float32, device='cuda')
    alpha = torch.from_numpy(data['alpha']).to(dtype=torch.float32, device='cuda')

    print(f"\n{len(points)} points found.")
    return points, radii, colors, alpha


def check_point_cloud(data_dir):
    # 点云文件可视化：
    points,radii,colors,alpha = point_cloud_load(data_dir)
    # 将每个张量转换为NumPy数组
    points_np = points.cpu().numpy()
    radii_np = radii.cpu().numpy()
    colors_np = colors.cpu().numpy()
    alpha_np = alpha.cpu().numpy()
    
    print(f"{len(points)} points found.")
    # 合并所有数据到一个数组中
    combined_data = np.column_stack((points_np, radii_np[:, np.newaxis], colors_np, alpha_np[:, np.newaxis]))
    
    # 构建保存路径
    combined_txt_path = os.path.join(data_dir, '2d_point_cloud.txt')
    
    # 保存为txt文件
    header = "x,y,radius,r,g,b,alpha"
    np.savetxt(combined_txt_path, combined_data, fmt='%f', delimiter=', ', header=header, comments='')
    
    print(f"2d_point_cloud data saved to {combined_txt_path}")
    

def default_init():
    # 用固定参数初始化一个default_point_cloud.npz以便测试
    # 点的位置
    global_points = torch.tensor([
        [358, 200], [300, 400], [150, 250], [100, 450],
        [120, 220], [320, 420], [289, 270], [370, 270],
        [140, 240], [340, 440], [160, 360], [360, 100],
        [130, 230], [150, 430], [180, 350], [54, 480],
        [190, 290], [390, 490], [110, 210], [130, 290],
        [30, 215], [315, 415]
    ], dtype=torch.float32, device='cuda')
    # 点的半径
    global_radii = torch.tensor([
        20, 30, 25, 35,
        22, 32, 27, 37,
        200, 34, 26, 36,
        23, 33, 28, 38,
        29, 39, 21, 31,
        21.5, 31.5
    ], dtype=torch.float32, device='cuda')
    # 点的颜色
    global_colors = torch.tensor([
        [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0],
        [0.0, 0.0, 0.5], [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5],
        [0.2, 0.2, 0.2], [0.8, 0.8, 0.8], [0.7, 0.3, 0.0], [0.0, 0.7, 0.3],
        [0.3, 0.0, 0.7], [0.7, 0.0, 0.3], [0.3, 0.7, 0.0], [0.0, 0.3, 0.7],
        [0.05, 0.723, 0.54], [0.25, 0.01, 0.425]
    ], dtype=torch.float32, device='cuda')
    # 点的透明度
    global_alpha = torch.tensor([
        0.8, 0.6, 0.7, 0.5,
        0.85, 0.65, 0.75, 0.55,
        1, 0.62, 0.72, 0.52,
        0.87, 0.67, 0.77, 0.57,
        0.88, 0.68, 0.81, 0.61,
        0.79, 0.69
    ], dtype=torch.float32, device='cuda')
    
    return global_points, global_radii, global_colors, global_alpha

def main():

    global_points = None
    global_radii = None
    global_colors = None
    global_alpha = None
    
    global_points, global_radii, global_colors, global_alpha = default_init()
    
    current_path = os.path.abspath(os.path.dirname(__file__))
    output_dir = current_path
    
    # 保存点云数据
    point_cloud_save(global_points, global_radii, global_colors, global_alpha, output_dir)
    
    check_point_cloud(output_dir)

if __name__ == "__main__":
    
    main()
    


"""
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Point Cloud Operations")
    subparsers = parser.add_subparsers(dest='command', help='Sub-command help')

    # Load command
    load_parser = subparsers.add_parser('load', help='Load point cloud data')
    load_parser.add_argument('data_dir', type=str, help='Directory containing 2d_point_cloud.npz')

    # Check command
    check_parser = subparsers.add_parser('check', help='Check and save point cloud data as txt')
    check_parser.add_argument('data_dir', type=str, help='Directory containing 2d_point_cloud.npz')

    # Default init command
    default_init_parser = subparsers.add_parser('default_init', help='Initialize default point cloud data')
    default_init_parser.add_argument('--output_dir', type=str, help='Output directory to save default point cloud data', default=os.path.dirname(__file__))

    args = parser.parse_args()

    if args.command == 'load':
        points, radii, colors, alpha = point_cloud_load(args.data_dir)
        print("Point cloud data loaded successfully.")
        # You can add additional processing here if needed

    elif args.command == 'check':
        check_point_cloud(args.data_dir)

    elif args.command == 'default_init':
        output_dir = args.output_dir
        global_points, global_radii, global_colors, global_alpha = default_init()
        point_cloud_save(global_points, global_radii, global_colors, global_alpha, output_dir)

    else:
        parser.print_help()


"""