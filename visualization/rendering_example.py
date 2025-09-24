
import torch
import numpy as np
from huggingface_hub import hf_hub_download
import viser
import time
import os
import random
from PIL import Image

from utils import lift_pixels_to_world

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mvtracker = torch.hub.load("ethz-vlg/mvtracker", "mvtracker", pretrained=True, device=device)
    server = viser.ViserServer(host="0.0.0.0")

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
    intrs = torch.empty((B, T, 3, 3), dtype = torch.float32)
    extrs = torch.empty((B, T, 4, 4), dtype = torch.float32) 

    # Assign to the tensor(rgbs, depths)
    for b in range(4):
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
    samples_init = np.load(os.path.join(base_dir,"init_pt_cld.npz"))
    
    import pdb
    pdb.set_trace()
    intrs = torch.from_numpy(samples["intrinsics"])[0:4,None,:,:].float() # (31, 1, 3, 3)
    intrs = intrs.repeat(1, T, 1, 1)
    extrs = torch.from_numpy(samples["extrinsics"])[0:4,None,0:3,:].float() # (31, 1, 3, 4)
    extrs = extrs.repeat(1, T, 1, 1)  
    randn_ls = random.sample(range(2275), 1000)
    query_points = torch.from_numpy(samples["query_points_3d"])[randn_ls,:].float() # (2275, 3)

    with torch.no_grad():
        results = mvtracker(
            rgbs=rgbs[None].to(device) / 255.0,
            depths=depths[None].to(device),
            intrs=intrs[None].to(device),
            extrs=extrs[None].to(device),
            query_points_3d=query_points[None].to(device),
        )    
    print("Finish Tracking")
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
        rot_mat3 = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype = np.float32)
        server.scene.add_point_cloud(
            name="/point_cloud",
            points=pcd_world.reshape(-1,3)@rot_mat3,
            colors=colors.reshape(-1, 3),
            point_size=0.01,
        )

    def add_query_point_cloud(t):
        track_data = pred_tracks[0,t].detach().cpu().numpy()
        color = np.array([1.0, 1.0, 0.0], dtype=np.float32)
        rot_mat3 = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype = np.float32)
        server.scene.add_point_cloud(
                name="/query_point_cloud",
                points=track_data@rot_mat3,
                colors=np.tile(color, (track_data.shape[0], 1)),
                point_size=0.03,
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