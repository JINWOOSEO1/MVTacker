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
    zeros = np.zeros((query_points.shape[0],1))
    query_points = np.hstack((zeros, query_points)) # (20,4)

    N = query_points.shape[0]
    K = 12 # the number of frames that needs to operate mvtracker
    current_frame_idx = -1

    # set queue
    rgb_window = torch.zeros((B, K, 3, H, W)).to(device)
    depth_window = torch.zeros((B, K, 1, H, W)).to(device)
    intr_window = torch.zeros((B, K, 3, 3)).to(device)
    extr_window = torch.zeros((B, K, 4, 4)).to(device)
    query_points_window = torch.zeros((K, N, 4)).to(device)

    def render_image(rgb, depth, intr, extr):
        nonlocal current_frame_idx
        nonlocal query_points_window

        current_frame_idx = current_frame_idx + 1
        current_insert_location = current_frame_idx % K

        rgb_window[:,current_insert_location] = torch.from_numpy(rgb).to(device)
        depth_window[:, current_insert_location] = torch.from_numpy(depth).to(device)
        intr_window[:, current_insert_location] = torch.from_numpy(intr).to(device)
        extr_window[:, current_insert_location] = torch.from_numpy(extr).to(device)

        if(current_frame_idx < (K-1)): # if current_frame_idx is 0~(K-2), there are lack of infromation to apply mvtracker
            return

        rgb_input_tensor = torch.cat((rgb_window[:, current_insert_location+1:], rgb_window[:, :current_insert_location+1]), dim=1)
        depth_input_tensor = torch.cat((depth_window[:, current_insert_location+1:], depth_window[:, :current_insert_location+1]), dim=1)
        intr_input_tensor = torch.cat((intr_window[:, current_insert_location+1:], intr_window[:, :current_insert_location+1]), dim=1)
        extr_input_tensor = torch.cat((extr_window[:, current_insert_location+1:], extr_window[:, :current_insert_location+1]), dim=1)

        if current_frame_idx == K-1:
            query_points_window[1] = torch.from_numpy(query_points).to(device)

        with torch.no_grad():
            start_time = time.time()        

            results = mvtracker(
                rgbs=rgb_input_tensor[None] / 255.0,   
                depths=depth_input_tensor[None],       
                intrs=intr_input_tensor[None],         
                extrs=extr_input_tensor[None,:,:,0:3,:],        
                query_points=query_points_window[1:2],
            )    
            print("Iteration takes: ", time.time()-start_time)


        pred_tracks = results["traj_e"].to(device)  # [1,K,N,3] tensor
        pred_vis = results["vis_e"].to(device)      # [1,K,N]

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

            track_data = pred_tracks[0,-1].detach().cpu().numpy()
            color = np.array([0.76, 0.96, 0.15], dtype=np.float32)
            server.scene.add_point_cloud(
                    name="/query_point_cloud",
                    points=track_data,
                    colors=np.tile(color, (track_data.shape[0], 1)),
                    point_size=0.01,
            )
          
        new_column = torch.zeros((K, N, 1), device=device, dtype = torch.float32)   

        # Neglect rendering 0~K-2 images
        query_points_window = torch.concatenate((new_column, pred_tracks[0]), dim=2)
        add_point_cloud(current_insert_location)
        print(f"Finish rendering {current_frame_idx+1}th image")
        
    for t in range(0,T):
        render_image(rgbs[:,t,:,:,:], depths[:,t,:,:,:], intrs[:,t,:,:], extrs[:,t,:,:])
    

if __name__ == "__main__":
    main()