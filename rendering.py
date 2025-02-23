import os
import sys
import torch
from PIL import Image
import numpy as np
from tqdm import tqdm
import time
import argparse
from render_test_tqdm import render_point_cloud_2d, normalize_image, render_and_save
from point_cloud_2d import point_cloud_load

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Optimize 2D Point Cloud")
    parser.add_argument("input_dir", type=str, help="Directory containing initial point cloud and target image")
    parser.add_argument("output_dir", type=str, help="Directory to save optimized point cloud and intermediate results")
    
    args = parser.parse_args()

    print("CUDA available:", torch.cuda.is_available())
    print("CUDA version:", torch.version.cuda)
    
    render_and_save(args.input_dir,args.output_dir)
    
    print(f"done.")



if __name__ == "__main__":
    main()