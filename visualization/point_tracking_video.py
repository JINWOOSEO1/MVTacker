import torch
from torch import nn
import numpy as np
from huggingface_hub import hf_hub_download
import viser
import time
import os
from PIL import Image
import random
from mvtracker.models.core.mvtracker.mvtracker import MVTracker
from visualization.utils import lift_pixels_to_world
import rospy

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mvtracker = MVTracker(hidden_size=256, fmaps_dim=128) 
    mvtracker.load_state_dict(torch.load("checkpoints/mvtracker_200000_june2025.pth"))
    mvtracker.to(device).eval()
    server = viser.ViserServer(host="0.0.0.0")

    # Make New samples
    base_dir = "../../datasets/0926"
    B, T , W, H= 3, 132, 1020, 1020 

    rgbs = np.zeros((B, T, 3, H, W), dtype=np.float32)
    depths = np.zeros((B, T, 1, H, W), dtype=np.float32)
    intrs = np.zeros((B, T, 3, 3), dtype = np.float32)
    extrs = np.zeros((B, T, 4, 4), dtype = np.float32)

    for b in range(B):
        rgbd_folder = os.path.join(base_dir, f"{b+1}")
        first_img_path = os.path.join(rgbd_folder, "rgb_00000.png")
        W_crop, H_crop = Image.open(first_img_path).size
        for t in range(T):
            rgb_file_path = os.path.join(rgbd_folder, f"rgb_{t:05d}.png")
            img = Image.open(rgb_file_path).convert('RGB')
            rgb_array = np.array(img).astype(np.float32) 
            rgbs[b,t,:,0:H_crop,0:W_crop] = rgb_array.transpose(2,0,1)
            
            depth_file_path = os.path.join(rgbd_folder, f"depth_{t:05d}.npy")
            depth_data = np.load(depth_file_path).astype(np.float32) 
            depths[b,t,0,0:H_crop,0:W_crop] = depth_data

        intr_file_path = os.path.join(rgbd_folder, f"intrinsic.npy")
        intr_mat = np.load(intr_file_path).astype(np.float32) # (3,3)
        intrs[b] = np.repeat(intr_mat[None,:,:], T, axis = 0)

        extr_file_path = os.path.join(rgbd_folder, f"extrinsic.npy")
        extr_mat = np.load(extr_file_path).astype(np.float32) 
        extr_mat = extr_mat @ np.diag([1, -1, -1, 1])
        extr_mat = np.linalg.inv(extr_mat)
        extrs[b] = np.repeat(extr_mat[None,:,:], T, axis = 0) # (4,4)

        print(f"{b+1} view is finished")

    query_points = np.load("../../datasets/0926/video_input_query_point.npy").astype(np.float32)
    zeros = np.zeros((query_points.shape[0],1), dtype=np.float32)
    query_points = np.hstack((zeros, query_points)) # (20,4)

    N = query_points.shape[0]

    # set tensor
    rgb_tensor = torch.from_numpy(rgbs).to(device)
    depth_tensor = torch.from_numpy(depths).to(device)
    intr_tensor = torch.from_numpy(intrs).to(device)
    extr_tensor = torch.from_numpy(extrs).to(device)
    query_points_tensor = torch.from_numpy(query_points).to(device)
    
    rgb_tensor_interp = nn.functional.interpolate(
        input=rgb_tensor.reshape(B*T, 3, H, W),
        scale_factor=0.5,
        mode='bilinear',
    ).reshape(B, T, 3, H//2, W//2)

    depth_tensor_interp = nn.functional.interpolate(
        input=depth_tensor.reshape(B*T, 1, H, W),
        scale_factor=0.5,
        mode='nearest',
    ).reshape(B, T, 1, H//2, W//2)

    intr_tensor_interp = intr_tensor // 2.0
    intr_tensor_interp[:, :, 2, 2] = 1.00

    with torch.no_grad():
        start_time = time.time()        
        results = mvtracker(
            rgbs=rgb_tensor_interp[None] / 255.0,   
            depths=depth_tensor_interp[None],       
            intrs=intr_tensor_interp[None],         
            extrs=extr_tensor[None,:,:,0:3,:],        
            query_points=query_points_tensor[None],
        )    
        print("Iteration takes: ", time.time()-start_time)

    pred_tracks = results["traj_e"].to(device)  # [1,K,N,3] tensor
    pred_vis = results["vis_e"].to(device)      # [1,K,N]

    def add_point_cloud(t):
        pcd_world = []
        colors = []
        for cam_idx in range(B):
            pcd = lift_pixels_to_world(depth_tensor_interp[cam_idx, t, 0], intr_tensor_interp[cam_idx, t], extr_tensor[cam_idx, t])
            pcd_world.append(pcd.detach().cpu().numpy())
            color = rgb_tensor_interp[cam_idx, t] # (3, H, W)
            colors.append(color.permute(1,2,0).detach().cpu().numpy() / 255.0)

        pcd_world = np.concatenate(pcd_world) # (H*B, W, 3)
        colors = np.concatenate(colors) # (H*B, W, 3)    

        # crop workspace
        workspace_bounds = np.array([[0, 1.0], [-0.5, 0.5], [-0.5, 0.5]])
        workspace_mask = (pcd_world[..., 0] > workspace_bounds[0, 0]) & (pcd_world[..., 0] < workspace_bounds[0, 1]) &\
            (pcd_world[..., 1] > workspace_bounds[1, 0]) & (pcd_world[..., 1] < workspace_bounds[1, 1]) &\
            (pcd_world[..., 2] > workspace_bounds[2, 0]) & (pcd_world[..., 2] < workspace_bounds[2, 1])
        pcd_world = pcd_world[workspace_mask]
        colors = colors[workspace_mask]

        server.scene.add_point_cloud(
            name="/point_cloud",
            points=pcd_world.reshape(-1,3),
            colors=colors.reshape(-1, 3),
            point_size=0.005,
        )

        track_data = pred_tracks[0,t].detach().cpu().numpy()
        color = np.array([0.76, 0.96, 0.15], dtype=np.float32)
        server.scene.add_point_cloud(
                name="/query_point_cloud",
                points=track_data,
                colors=np.tile(color, (track_data.shape[0], 1)),
                point_size=0.01,
        )
    for t in range(T):
        add_point_cloud(t)
        time.sleep(0.5)
        print(f"Finish rendering {t}th image")
    
    

if __name__ == "__main__":
    main()