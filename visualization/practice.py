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

    T = 6
    # Example input from demo sample (downloaded automatically)
    sample = np.load(hf_hub_download("ethz-vlg/mvtracker", "data_sample.npz"))
    rgbs = torch.from_numpy(sample["rgbs"]).float()[:,0:T,:,:,:] # (B, T, 3, H, W)
    depths = torch.from_numpy(sample["depths"]).float()[:,0:T,:,:,:] # (B, T, 1, H, W)
    intrs = torch.from_numpy(sample["intrs"]).float()[:,0:T,:,:] # (B, T, 3, 3)
    extrs = torch.from_numpy(sample["extrs"]).float()[:,0:T,:,:] # (B, T, 3, 4)
    query_points = torch.from_numpy(sample["query_points"]).float() # (N, 4)
    query_points = query_points[query_points[:,0]==0]
    
    rgbs, depths, intrs, extrs, query_points = rgbs.cuda(), depths.cuda(), intrs.cuda(), extrs.cuda(), query_points.cuda()
    
    B, _, _, H, W = depths.shape # (4, T, 1, 480, 640)

    with torch.no_grad():
        results = mvtracker(
            rgbs=rgbs[None].to(device) / 255.0,
            depths=depths[None].to(device),
            intrs=intrs[None].to(device),
            extrs=extrs[None].to(device),
            query_points_3d=query_points[None].to(device),
        )    
    pred_tracks = results["traj_e"].cpu()  # [1,T,N,3]
    pred_vis = results["vis_e"].cpu()      # [1,T,N]

    def slider_changed(event: viser.GuiEvent):
        timestep = event.target.value
        add_point_cloud(timestep)
        add_query_point_cloud(timestep)

    with server.gui.add_folder("Time Slider"):
        gui_slider = server.gui.add_slider(
            "Timestep_slider",
            min=0,
            max=T-1,
            step=1,
            initial_value=0,
            disabled=False,
        )

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

        # # crop workspace
        # workspace_bounds = np.array([[-2.0, 2.0], [-2.0, 2.0], [-2.0, 2.0]])
        # workspace_mask = (pcd_world[..., 0] > workspace_bounds[0, 0]) & (pcd_world[..., 0] < workspace_bounds[0, 1]) &\
        #     (pcd_world[..., 1] > workspace_bounds[1, 0]) & (pcd_world[..., 1] < workspace_bounds[1, 1]) &\
        #     (pcd_world[..., 2] > workspace_bounds[2, 0]) & (pcd_world[..., 2] < workspace_bounds[2, 1])

        # pcd_world = pcd_world[workspace_mask]
        # colors = colors[workspace_mask]

        server.scene.add_point_cloud(
            name="/point_cloud",
            points=pcd_world.reshape(-1,3),
            colors=colors.reshape(-1, 3),
            point_size=0.03,
        )

    def add_query_point_cloud(t):
        track_data = pred_tracks[0,t].detach().cpu().numpy()
        color = np.array([1.0, 1.0, 0.0], dtype=np.float32)

        server.scene.add_point_cloud(
                name="/query_point_cloud",
                points=track_data,
                colors=np.tile(color, (track_data.shape[0], 1)),
                point_size=0.05,
            )
    
    add_point_cloud(0)
    add_query_point_cloud(0)
    gui_slider.on_update(slider_changed)    
    
    # For video
    # for t in range(1,24):
    #     add_point_cloud(t)
    #     add_query_point_cloud(t)
    
    
    while True:
        time.sleep(0.05)


if __name__ == "__main__":
    main()