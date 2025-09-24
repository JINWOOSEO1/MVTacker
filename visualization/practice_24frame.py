import torch
import numpy as np
from huggingface_hub import hf_hub_download
import viser
import time

from utils import lift_pixels_to_world

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mvtracker = torch.hub.load("ethz-vlg/mvtracker", "mvtracker", pretrained=True, device=device)
    server = viser.ViserServer(host="0.0.0.0")

    # Example input from demo sample (downloaded automatically)
    sample = np.load(hf_hub_download("ethz-vlg/mvtracker", "data_sample.npz"))
    rgbs = torch.from_numpy(sample["rgbs"]).float() # (B, T, 3, H, W)
    depths = torch.from_numpy(sample["depths"]).float() # (B, T, 1, H, W)
    intrs = torch.from_numpy(sample["intrs"]).float() # (B, T, 3, 3)
    extrs = torch.from_numpy(sample["extrs"]).float() # (B, T, 3, 4)
    query_points = torch.from_numpy(sample["query_points"]).float() # (N, 4)
    query_points = query_points[query_points[:,0] == 0]
    
    # Data Augmentation
    rgbs = torch.tile(rgbs, (1,5,1,1,1))
    depths = torch.tile(depths, (1,5,1,1,1))
    intrs = torch.tile(intrs, (1,5,1,1))
    extrs = torch.tile(extrs, (1,5,1,1))

    rgb_ls= []
    depth_ls = []
    intr_ls = []
    extr_ls = []
    query_points_ls = []
    cnt_time = -1
    B, T, _, H, W = depths.shape # (4, 24, 1, 480, 640)
    K = 24 # the number of frames that needs to operate mvtracker(12 is available)
    
    def render_image(rgb, depth, intr, extr):
        rgb_ls.append(rgb)
        depth_ls.append(depth)
        intr_ls.append(intr)
        extr_ls.append(extr)

        nonlocal cnt_time
        cnt_time = cnt_time+1
        if(((cnt_time % K) != 0) or cnt_time==0):
            return

        rgb_window = torch.stack(rgb_ls[-K:], dim=1) # (B, K, 3, H, W)
        depth_window = torch.stack(depth_ls[-K:], dim=1) # (B, K, 3, H, W)
        intr_window = torch.stack(intr_ls[-K:], dim=1) # (B, K, 3, 3)
        extr_window = torch.stack(extr_ls[-K:], dim=1) # (B, K, 3, 4)
        query_points_window = query_points
        # It occurs out of bound because the input video is not smooth
        # query_points_window = query_points if not query_points_ls else query_points_ls[-1]

        with torch.no_grad():
            results = mvtracker(
                rgbs=rgb_window[None].to(device) / 255.0,
                depths=depth_window[None].to(device),
                intrs=intr_window[None].to(device),
                extrs=extr_window[None].to(device),
                query_points_3d=query_points_window[None].to(device),
            )    
        pred_tracks = results["traj_e"].cpu()  # [1,K,N,3]
        pred_vis = results["vis_e"].cpu()      # [1,K,N]
        
        # Add time column in query_points_ls
        for idx in range(K):
            new_column_value = cnt_time - K + idx
            new_column = torch.full((pred_tracks.shape[2], 1), new_column_value)
            query_points_ls.append(torch.cat((new_column, pred_tracks[0,idx]), dim=1))
        
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
                point_size=0.03,
            )

            track_data = pred_tracks[0,t].detach().cpu().numpy()
            color = np.array([1.0, 1.0, 0.0], dtype=np.float32)

            server.scene.add_point_cloud(
                    name="/query_point_cloud",
                    points=track_data,
                    colors=np.tile(color, (track_data.shape[0], 1)),
                    point_size=0.05,
            )
        
        for i in range(0,K):
            add_point_cloud(i)
            time.sleep(0.1)
        print("Finish rendering")
        
    for t in range(0,T*5):
        render_image(rgbs.select(1,t), depths.select(1,t), intrs.select(1,t), extrs.select(1,t))
        time.sleep(0.1)
if __name__ == "__main__":
    main()