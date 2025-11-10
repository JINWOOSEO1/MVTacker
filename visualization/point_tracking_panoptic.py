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

    ############# Make New samples #####################
    base_dir = "../../datasets/panoptic-multiview/basketball"
    img_dir = os.path.join(base_dir, "ims")
    depth_dir = os.path.join(base_dir, "dynamic3dgs_depth")

    # Make empty tensor
    B, T = 4, 150
    first_img_path = os.path.join(img_dir, "0", "000000.jpg")
    img = Image.open(first_img_path)
    W, H = img.size

    rgbs = torch.empty((B, T, 3, H, W), dtype=torch.float32)
    depths = torch.empty((B, T, 1, H, W), dtype=torch.float32)

    # Assign to the tensor(rgbs, depths)
    for b in range(B):
        rgb_folder = os.path.join(img_dir, f"{b}")
        for t in range(T):
            rgb_file_path = os.path.join(rgb_folder, f"{t:06d}.jpg")

            img = Image.open(rgb_file_path).convert('RGB')
            rgb_array = np.array(img).astype(np.float32)
            rgbs[b, t] = torch.from_numpy(rgb_array.transpose(2,0,1)) 
        depth_file_path = os.path.join(depth_dir, f"depths_{b:02d}.npy")
        
        depth_data = np.load(depth_file_path).astype(np.float32)
        depths[b] = torch.from_numpy(depth_data[0:T,None,:,:])
        print(f"{b} view is finished")

    # Assign to the tensor(intrs, extrs)
    samples = np.load(os.path.join(base_dir,"tapvid3d_annotations.npz"))
    intrs = torch.from_numpy(samples["intrinsics"])[0:B,None,:,:].float() # (31, 1, 3, 3)
    intrs = intrs.repeat(1, T, 1, 1)
    extrs = torch.from_numpy(samples["extrinsics"])[0:B,None,0:3,:].float() # (31, 1, 3, 4)
    extrs = extrs.repeat(1, T, 1, 1)  
    
    query_points = torch.from_numpy(samples["query_points_3d"]).float() # (2275, 4)
    query_points = query_points[query_points[:,0]<11] # (140, 4)
    # query_points[:,0] = 0.0
    rgbs, depths, intrs, extrs, query_points = rgbs.cuda(), depths.cuda(), intrs.cuda(), extrs.cuda(), query_points.cuda()
    ####################################################

    def add_point_cloud(t):
        pcd_world = []
        colors = []
        for cam_idx in range(B):
            pcd = lift_pixels_to_world(depths[cam_idx, t, 0], intrs[cam_idx, t], extrs[cam_idx, t])
            pcd_world.append(pcd.detach().cpu().numpy())
            color = rgbs[cam_idx, t] # (3, H, W)
            colors.append(color.permute(1,2,0).detach().cpu().numpy() / 255.0)

        rot_mat3 = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype = np.float32)

        pcd_world = np.concatenate(pcd_world) # (H*B, W, 3)
        colors = np.concatenate(colors) # (H*B, W, 3)       
        server.scene.add_point_cloud(
            name="/point_cloud",
            points=pcd_world.reshape(-1,3)@rot_mat3,
            colors=colors.reshape(-1, 3),
            point_size=0.01,
        )

        track_data = pred_tracks[0,t].detach().cpu().numpy()
        color = np.array([1.0, 1.0, 0.0], dtype=np.float32)
        server.scene.add_point_cloud(
                name="/query_point_cloud",
                points=track_data@rot_mat3,
                colors=np.tile(color, (track_data.shape[0], 1)),
                point_size=0.03,
        )

    
    with torch.no_grad():
        start_time = time.time()        

        results = mvtracker(
            rgbs=rgbs[None].to(device) / 255.0,   
            depths=depths[None].to(device),       
            intrs=intrs[None].to(device),         
            extrs=extrs[None].to(device),        
            query_points=query_points[None].to(device),
        )    
        print("Iteration takes: ", time.time()-start_time)

    pred_tracks = results["traj_e"].to(device)  # [1,T,N,3] tensor
    pred_vis = results["vis_e"].to(device)      # [1,T,N]

    for t in range(T):
        add_point_cloud(t)
        print(f"add point cloud: {t}")
        time.sleep(0.1)



if __name__ == "__main__":
    main()