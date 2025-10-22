
import torch
import numpy as np
from huggingface_hub import hf_hub_download
import viser
import time
import os
import random
from PIL import Image

from visualization.utils import lift_pixels_to_world

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mvtracker = torch.hub.load("ethz-vlg/mvtracker", "mvtracker", pretrained=True, device=device)
    server = viser.ViserServer(host="0.0.0.0")

    # Make empty tensor
    base_dir = "../../datasets/0926"
    B, T , W, H= 3, 132, 1280, 1280 # Isn't H = 720?? Why the height of second view image is 1006? 

    rgbs = torch.zeros((B, T, 3, H, W), dtype=torch.float32)
    depths = torch.zeros((B, T, 1, H, W), dtype=torch.float32)
    intrs = torch.zeros((B, T, 3, 3), dtype = torch.float32)
    extrs = torch.zeros((B, T, 4, 4), dtype = torch.float32)

    C = np.diag([1, -1, -1, 1])

    for b in range(B):
        rgbd_dir = os.path.join(base_dir,f"{b+1}")
        first_img_path = os.path.join(rgbd_dir, "rgb_00000.png")
        W_crop, H_crop = Image.open(first_img_path).size

        for t in range(T):
            rgb_file_path = os.path.join(rgbd_dir, f"rgb_{t:05d}.png")
            img = Image.open(rgb_file_path).convert('RGB')
            rgb_array = np.array(img).astype(np.float32) 
            rgbs[b, t, :, 0:H_crop, 0:W_crop] = torch.from_numpy(rgb_array.transpose(2,0,1)) 
            
            depth_file_path = os.path.join(rgbd_dir, f"depth_{t:05d}.npy")
            depth_data = np.load(depth_file_path).astype(np.float32) 
            depths[b, t, 0, 0:H_crop, 0:W_crop] = torch.from_numpy(depth_data)

        intr_file_path = os.path.join(rgbd_dir, f"intrinsic.npy")
        intr_mat = np.load(intr_file_path).astype(np.float32) # (3,3)
        intr_data = np.repeat(intr_mat[None,:,:], T, axis = 0)
        intrs[b] = torch.from_numpy(intr_data)

        extr_file_path = os.path.join(rgbd_dir, f"extrinsic.npy")
        extr_mat = np.load(extr_file_path).astype(np.float32) 
        extr_mat = extr_mat @ C
        extr_mat = np.linalg.inv(extr_mat)
        extr_data = np.repeat(extr_mat[None,:,:], T, axis = 0) # (4,4)
        extrs[b] = torch.from_numpy(extr_data)
    
    query_points_path = os.path.join(base_dir,"query_point.npy")
    query_points = np.load(query_points_path).astype(np.float32) # (20, 3)

    rgbs, depths, intrs, extrs, = rgbs.cuda(), depths.cuda(), intrs.cuda(), extrs.cuda()


    def add_point_cloud(t):
        pcd_world = []
        colors = []

        for cam_idx in range(B):
            pcd = lift_pixels_to_world(depths[cam_idx, t, 0], intrs[cam_idx, t], extrs[cam_idx, t])
            pcd_world.append(pcd.detach().cpu().numpy())
            color = rgbs[cam_idx, t] # (3, H, W)
            colors.append(color.permute(1,2,0).detach().cpu().numpy() / 255.0)

        pcd_world = np.concatenate(pcd_world) # (H*B, W, 3)
        colors = np.concatenate(colors) # (H*B, W, 3)       

        server.scene.add_point_cloud(
            name="/point_cloud",
            points=pcd_world.reshape(-1,3),
            colors=colors.reshape(-1, 3),
            point_size=0.01,
        )

        color = np.array([1.0, 1.0, 0.0], dtype=np.float32)
        server.scene.add_point_cloud(
                name="/query_point_cloud",
                points=query_points,
                colors=np.tile(color, (query_points.shape[0], 1)),
                point_size=0.01,
        )


    for t in range(T):
        add_point_cloud(t)

    while True:
        time.sleep(0.05)

if __name__ == "__main__":
    main()