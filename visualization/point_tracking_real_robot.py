import torch
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
    B, T , W, H= 3, 132, 1020, 1020 # Isn't H = 720?? Why the height of second view image is 1006? 

    rgbs = torch.zeros((B, T, 3, H, W), dtype=torch.float32)
    depths = torch.zeros((B, T, 1, H, W), dtype=torch.float32)
    intrs = torch.zeros((B, T, 3, 3), dtype = torch.float32)
    extrs = torch.zeros((B, T, 4, 4), dtype = torch.float32)

    for b in range(B):
        rgbd_folder = os.path.join(base_dir, f"{b+1}")
        first_img_path = os.path.join(rgbd_folder, "rgb_00000.png")
        W_crop, H_crop = Image.open(first_img_path).size
        for t in range(T):
            rgb_file_path = os.path.join(rgbd_folder, f"rgb_{t:05d}.png")
            img = Image.open(rgb_file_path).convert('RGB')
            rgb_array = np.array(img).astype(np.float32) 
            rgbs[b,t,:,0:H_crop,0:W_crop] = torch.from_numpy(rgb_array.transpose(2,0,1)) 
            
            depth_file_path = os.path.join(rgbd_folder, f"depth_{t:05d}.npy")
            depth_data = np.load(depth_file_path).astype(np.float32) 
            depths[b,t,0,0:H_crop,0:W_crop] = torch.from_numpy(depth_data)

        intr_file_path = os.path.join(rgbd_folder, f"intrinsic.npy")
        intr_mat = np.load(intr_file_path).astype(np.float32) # (3,3)
        intr_data = np.repeat(intr_mat[None,:,:], T, axis = 0)
        intrs[b] = torch.from_numpy(intr_data)

        extr_file_path = os.path.join(rgbd_folder, f"extrinsic.npy")
        extr_mat = np.load(extr_file_path).astype(np.float32) 
        extr_mat = extr_mat @ np.diag([1, -1, -1, 1])
        extr_mat = np.linalg.inv(extr_mat)
        extr_data = np.repeat(extr_mat[None,:,:], T, axis = 0) # (4,4)
        extrs[b] = torch.from_numpy(extr_data)

        print(f"{b+1} view is finished")

    query_points_path = os.path.join(base_dir,"query_point.npy")
    query_points = np.load(query_points_path).astype(np.float32)
    zeros = np.zeros((query_points.shape[0],1))
    query_points = np.hstack((zeros, query_points)) # (20,4)
    query_points = torch.from_numpy(query_points).float()

    rgbs, depths, intrs, extrs, query_points = rgbs.cuda(), depths.cuda(), intrs.cuda(), extrs.cuda(), query_points.cuda()



    rgb_ls= []
    depth_ls = []
    intr_ls = []
    extr_ls = []
    query_points_ls = []
    K = 12 # the number of frames that needs to operate mvtracker
    cnt_time = -1

    # rgb_window, depth, ... initialize with zeros
    rgb_window = torch.zeros((B, K, 3, H, W)).to(device)
    # ...

    current_frame_idx = 0 # 0 ~ inf
    
    # inside loop
    # current_insert_location = current_frame_idx % K
    # rgb_window[cam_idx, current_insert_location, ...] = torch.from_numpy(rgb).to(device)
    # rgb_input_tensor = torch.cat((rgb_window[current_insert_location:], rgb_window[:current_insert_location]), dim=1)

    def render_image(rgb, depth, intr, extr):
        rgb_ls.append(rgb)
        depth_ls.append(depth)
        intr_ls.append(intr)
        extr_ls.append(extr)

        nonlocal cnt_time
        cnt_time = cnt_time + 1
        if(cnt_time < (K-1)): # if cnt_time is 0~(K-2), there are lack of infromation to apply mvtracker
            return
        
        rgb_window = torch.stack(rgb_ls[-K:], dim=1)        # (B, K, 3, H, W)
        depth_window = torch.stack(depth_ls[-K:], dim=1)    # (B, K, 1, H, W)
        intr_window = torch.stack(intr_ls[-K:], dim=1)      # (B, K, 3, 3)
        extr_window = torch.stack(extr_ls[-K:], dim=1)      # (B, K, 4, 4)
        
        def add_point_cloud(t):
            pcd_world = []
            colors = []
            for cam_idx in range(B):
                pcd = lift_pixels_to_world(depth_window[cam_idx, t, 0], intr_window[cam_idx, t], extr_window[cam_idx, t])
                pcd_world.append(pcd.detach().cpu().numpy())
                color = rgb_window[cam_idx, t] # (3, H, W)
                colors.append(color.permute(1,2,0).detach().cpu().numpy() / 255.0)

            pcd_world = np.concatenate(pcd_world) # (H*B, W, 3)
            colors = np.concatenate(colors) # (H*B, W, 3)       
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
          

        
        query_points_window = query_points if cnt_time == K-1 else query_points_ls[-(K-1)]

        with torch.no_grad():
            start_time = time.time()        

            results = mvtracker(
                rgbs=rgb_window[None].to(device) / 255.0,   
                depths=depth_window[None].to(device),       
                intrs=intr_window[None].to(device),         
                extrs=extr_window[None,:,:,0:3,:].to(device),        
                query_points=query_points_window[None].to(device),
            )    
            print("Iteration takes: ", time.time()-start_time)


        pred_tracks = results["traj_e"].to(device)  # [1,K,N,3] tensor
        pred_vis = results["vis_e"].to(device)      # [1,K,N]

        new_column = torch.full((pred_tracks.shape[2], 1), 0, device=device, dtype = torch.float32)   
        
        if cnt_time == K-1:  
            for idx in range(K):
                query_points_ls.append(torch.concatenate((new_column, pred_tracks[0,idx]), dim=1))
                add_point_cloud(idx)
        else:
            query_points_ls.append(torch.concatenate((new_column, pred_tracks[0,K-1]), dim=1))
            add_point_cloud(K-1)
        print(f"Finish rendering {cnt_time+1}th image")
        
        rgb_ls.pop(0)
        depth_ls.pop(0)
        extr_ls.pop(0)
        intr_ls.pop(0)

    for t in range(0,T):
        render_image(rgbs[:,t,:,:,:], depths[:,t,:,:,:], intrs[:,t,:,:], extrs[:,t,:,:])
    

if __name__ == "__main__":
    main()