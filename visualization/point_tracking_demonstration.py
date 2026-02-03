import torch
import numpy as np
import viser
import time
import os
import cv2
import argparse

from mvtracker.models.core.mvtracker.mvtracker import MVTracker

def lift_pixels_to_world(depths, intrs, extrs):
    device = depths.device
    H, W = depths.shape

    u, v = torch.meshgrid(
        torch.arange(W, device=device),
        torch.arange(H, device=device),
        indexing="xy"   
    )
    ones = torch.ones_like(u)

    # Calculate camera frame coordinate
    pix = torch.stack((u, v, ones), dim=-1).float()  # (H, W, 3)
    K_inv = torch.inverse(intrs)  # (3,3)
    cam_coords = (K_inv @ pix.view(-1, 3).T) * depths.view(-1).reshape(1,-1)  # (3, N)

    #Calculate world frame coordinate
    R = extrs[0:3, :3]  # (3,3)
    t = extrs[0:3, 3]   # (3,)
    world_coords = R.T @ (cam_coords - t.view(3,1))

    return world_coords.T.view(H, W, 3)  # (H, W, 3)


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mvtracker = MVTracker(hidden_size=256, fmaps_dim=128) 
    mvtracker.load_state_dict(torch.load("checkpoints/mvtracker_200000_june2025.pth"))
    mvtracker.to(device).eval()
    server = viser.ViserServer(host="0.0.0.0")

    # Preparing data
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    if not args.is_robot:
        if args.size == "big":
            input_dir = os.path.join(curr_dir, "../datasets/outputs_big")
        else:
            input_dir = os.path.join(curr_dir, "../datasets/outputs_small")
    else:
        if args.size == "big":
            input_dir = os.path.join(curr_dir, "../datasets/outputs_big_robot")
        else:
            input_dir = os.path.join(curr_dir, "../datasets/outputs_small_robot")
    
    B, T, H, W = 3, 180, 480, 640
    early_stop_T = T
    SCALE_FACTOR = 6.0
    rgbs = torch.empty((B, T, 3, H, W), dtype = torch.float32)
    depths = torch.empty((B, T, 1, H, W), dtype = torch.float32)

    for b in range(B):
        for t in range(T):
            img_path = os.path.join(input_dir, f"cam_{b+1}", f"{t:05d}.png")
            img_bgr= cv2.imread(img_path)
            if img_bgr is None:
                early_stop_T = t-1
                break
            rgb_array = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.uint8)
            rgbs[b, t] = torch.from_numpy(rgb_array.transpose(2,0,1))
            
            depth_path = os.path.join(input_dir, f"cam_{b+1}", f"{t:05d}.npy")
            depth_array = np.load(depth_path) / 1000.0 * SCALE_FACTOR
            depth_array = depth_array.astype(np.float32)
            depths[b, t, 0] = torch.from_numpy(depth_array)
        print(f"{b} view is finished")

    if args.is_robot:
        extr_path = os.path.join(curr_dir, "../datasets/extrinsics_mujoco_robot")
    else:
        extr_path = os.path.join(curr_dir, "../datasets/extrinsics_mujoco")
    extr_ls = []
    for b in range(B):
        extr_mat = np.load(os.path.join(extr_path, f"camera_{b+1}_extrinsic.npy")).astype(np.float32) 
        extr_mat[:,3] = extr_mat[:,3] * SCALE_FACTOR
        extr_ls.append(extr_mat)
    extrs = torch.from_numpy(np.stack(extr_ls, axis=0)).unsqueeze(1)
    extrs = extrs.repeat(1, T, 1, 1)  # (B, T, 4, 4)
    intrs = torch.tensor([579.41, 0.0, 320.0, 0.0, 579.41, 240.0, 0.0, 0.0, 1.0], dtype=torch.float32).reshape(3,3)
    intrs = intrs.repeat(B, T, 1, 1)

    rgbs, depths, intrs, extrs= rgbs.cuda(), depths.cuda(), intrs.cuda(), extrs.cuda()

    # Select query points
    main_cam_idx = 0 # 0~2

    display_img_path = os.path.join(input_dir, f"cam_{main_cam_idx+1}", f"00000.png")
    display_img = cv2.imread(display_img_path).copy()

    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    query_points_xy = []

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            query_points_xy.append((x, y))
           
            print(f"Selected point: ({x}, {y})")
    cv2.setMouseCallback("image", mouse_callback)

    while True:
        cv2.imshow("image", display_img)
        for (x, y) in query_points_xy:
            cv2.circle(display_img, (x, y), 5, (0, 0, 255), -1)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

    query_points = []
    for (x, y) in query_points_xy:
        depth = depths[main_cam_idx,0,0,y,x]
        K_inv = torch.inverse(intrs[main_cam_idx,0])
        pix = torch.tensor([x, y, 1.0], dtype=torch.float32, device=device)
        cam_coord = K_inv @ pix * depth
        R = extrs[main_cam_idx, 0, 0:3, :3]
        t = extrs[main_cam_idx, 0, 0:3, 3]
        point_3d = R.T @ (cam_coord - t)
        query_points.append(point_3d)

    query_points = torch.stack(query_points, dim=0).to(device) # (N, 3)
    zeros = torch.zeros((query_points.shape[0], 1), dtype=torch.float32, device=device)
    query_points = torch.cat([zeros, query_points], dim=1) # (N, 4)

    def add_point_cloud(t):
        pcd_world = []
        colors = []
        for cam_idx in range(B):
            pcd = lift_pixels_to_world(depths[cam_idx, t, 0], intrs[cam_idx, t], extrs[cam_idx, t])
            pcd_world.append(pcd.detach().cpu().numpy())
            color = rgbs[cam_idx, t] # (3, H, W)
            colors.append(color.permute(1,2,0).detach().cpu().numpy() / 255.0)

            # # crop workspace
            # workspace_bounds = np.array([[-0.2, 0.9], [-0.5, 0.5], [-0.03, 0.5]])
            # workspace_bounds = workspace_bounds * SCALE_FACTOR
            # workspace_mask = (pcd_world[..., 0] > workspace_bounds[0, 0]) & (pcd_world[..., 0] < workspace_bounds[0, 1]) &\
            #     (pcd_world[..., 1] > workspace_bounds[1, 0]) & (pcd_world[..., 1] < workspace_bounds[1, 1]) &\
            #     (pcd_world[..., 2] > workspace_bounds[2, 0]) & (pcd_world[..., 2] < workspace_bounds[2, 1])

            # pcd_world = pcd_world[workspace_mask]
            # colors = colors[workspace_mask]

        pcd_world = np.concatenate(pcd_world) # (H*B, W, 3)
        colors = np.concatenate(colors) # (H*B, W, 3)       
        server.scene.add_point_cloud(
            name="/point_cloud",
            points=pcd_world.reshape(-1,3),
            colors=colors.reshape(-1, 3),
            point_size=0.01,
        )

        track_data = pred_tracks[0,t].detach().cpu().numpy()
        color = np.array([1.0, 1.0, 0.0], dtype=np.float32)
        server.scene.add_point_cloud(
                name="/query_point_cloud",
                points=track_data,
                colors=np.tile(color, (track_data.shape[0], 1)),
                point_size=0.03,
        )

    with torch.no_grad():
        start_time = time.time()        
        results = mvtracker(
            rgbs=rgbs[None,:,:early_stop_T,:,:,:],   
            depths=depths[None,:,:early_stop_T,:,:,:],       
            intrs=intrs[None,:,:early_stop_T,:,:],         
            extrs=extrs[None,:,:early_stop_T,0:3,:],        
            query_points=query_points[None],
        )    
        print("Iteration takes: ", time.time()-start_time)

    pred_tracks = results["traj_e"].to(device)  # [1,T,N,3] tensor
    pred_vis = results["vis_e"].to(device)      # [1,T,N]

    for t in range(T):
        add_point_cloud(t)
        print(f"add point cloud: {t}")
        time.sleep(0.1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=str, default="small", help="big or small dataset")
    parser.add_argument("--is_robot", action="store_true", default=False, help="whether the dataset is from robot")
    args = parser.parse_args()
    main(args)