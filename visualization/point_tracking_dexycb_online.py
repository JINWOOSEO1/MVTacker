import torch
from torch.utils.data import DataLoader
import numpy as np
import viser
import time
import os 
from PIL import Image

from mvtracker.models.core.mvtracker.mvtracker_online import MVTrackerOnline
from mvtracker.datasets.utils import collate_fn
from mvtracker.datasets.dexycb_multiview_dataset import DexYCBMultiViewDataset
from visualization.utils import lift_pixels_to_world
from mvtracker.evaluation.evaluator_3dpt import evaluate_3dpt

def main(num_sample):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mvtracker = MVTrackerOnline(hidden_size=256, fmaps_dim=128)
    mvtracker.load_state_dict(torch.load("checkpoints/mvtracker_200000_june2025.pth"))
    mvtracker.to(device).eval()
    server = viser.ViserServer(host="0.0.0.0")

    dataset = DexYCBMultiViewDataset.from_name("dex-ycb-multiview-removehand", "../../datasets/DexYCB/")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    AJ_ls = []
    OA_ls = []
    MTE_ls = []
    Accuracy_ls = []

    for batch, _ in dataloader:
        

        track_upscaling_factor = batch.track_upscaling_factor
        
        rgbs = batch.video[0].to(device)
        depths = batch.videodepth[0].to(device)
        extrs = batch.extrs[0].to(device)
        intrs = batch.intrs[0].to(device)
        query_points = batch.query_points_3d[0].to(device)
        trajectory = batch.trajectory_3d[0].numpy() # (T, N, 3)
        visibility = batch.visibility[0]
        visibility_any = visibility.any(dim=0).numpy() # (T, N)
        B, T, _, H, W = rgbs.shape



        # sampling query_points
        # n_points = _query_points.shape[0]
        # movement = np.zeros(n_points)
        # for point_idx in range(n_points):
        #     point_track = _trajectory[_visibility_any[:, point_idx], point_idx, :]
        #     movement[point_idx] = np.linalg.norm(point_track[1:] - point_track[:-1], axis=-1).sum()

        # is_dyn = movement > 1.0
        # dyn_indices = np.nonzero(is_dyn)[0]
        # num_sample = min(num_sample, dyn_indices.shape[0])
        # rand_perm= torch.randperm(dyn_indices.shape[0])[:num_sample]
        # dyn_rand_indices = dyn_indices[rand_perm]

        # query_points = _query_points[dyn_indices]
        # trajectory = _trajectory[:, dyn_indices]
        # visibility_any = _visibility_any[:, dyn_indices] 

        # print(f"total points: {dyn_indices.shape[0]}")

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

            # crop workspace
            workspace_bounds = np.array([[-3.0, 3.0], [-3.0, 3.0], [0.0, 3.0]])
            workspace_mask = (pcd_world[..., 0] > workspace_bounds[0, 0]) & (pcd_world[..., 0] < workspace_bounds[0, 1]) &\
                (pcd_world[..., 1] > workspace_bounds[1, 0]) & (pcd_world[..., 1] < workspace_bounds[1, 1]) &\
                (pcd_world[..., 2] > workspace_bounds[2, 0]) & (pcd_world[..., 2] < workspace_bounds[2, 1])
            pcd_world = pcd_world[workspace_mask]
            colors = colors[workspace_mask]

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
        
        pred_tracks = torch.zeros((1, T, query_points.shape[0], 3), device=device, dtype=torch.float32)
        pred_vis = torch.zeros((1, T, query_points.shape[0]), device=device, dtype=torch.float32)

        mvtracker.init_video_online_processing()
        while mvtracker.online_ind + mvtracker.step < T:
            idx = mvtracker.online_ind
            with torch.no_grad():
                results = mvtracker(
                    rgbs=rgbs[:,idx:idx+mvtracker.S][None] / 255.0,
                    depths=depths[:,idx:idx+mvtracker.S][None],
                    intrs=intrs[:,idx:idx+mvtracker.S][None],
                    extrs=extrs[:,idx:idx+mvtracker.S][None],
                    query_points=query_points[None],
                )

            pred_tracks[:, idx:idx+mvtracker.S] = results["traj_e"]
            pred_vis[:, idx:idx+mvtracker.S] = results["vis_e"]   

        # sampling query_points
        # n_points = query_points.shape[0]
        # movement = np.zeros(n_points)
        # for point_idx in range(n_points):
        #     point_track = trajectory[visibility_any[:, point_idx], point_idx, :]
        #     movement[point_idx] = np.linalg.norm(point_track[1:] - point_track[:-1], axis=-1).sum()

        # is_dyn = movement > 1.0
        # dyn_indices = np.nonzero(is_dyn)[0]

        # query_points = query_points[dyn_indices]
        # trajectory = trajectory[:, dyn_indices]
        # visibility_any = visibility_any[:, dyn_indices] 
        # pred_tracks = pred_tracks[:,:,dyn_indices,:]
        # pred_vis = pred_vis[:,:,dyn_indices]

        eval_result = evaluate_3dpt(
            gt_tracks = trajectory,
            gt_visibilities = visibility_any,
            pred_tracks = pred_tracks[0].detach().cpu().numpy(),
            pred_visibilities = (pred_vis[0]>0.5).detach().cpu().numpy(),
            evaluation_setting = "dexycb-multiview",
            track_upscaling_factor = track_upscaling_factor,   
            query_points = query_points.detach().cpu().numpy(),
        )


        # print("================Results================")
        # print(f"AJ: {eval_result['3dpt/model__average_jaccard__dynamic']}")
        AJ_ls.append(eval_result['3dpt/model__average_jaccard__dynamic'])
        # print(f"Accuracy: {eval_result['3dpt/model__average_pts_within_thresh__dynamic']}")
        Accuracy_ls.append(eval_result['3dpt/model__average_pts_within_thresh__dynamic'])
        # print(f"MTE: {eval_result['3dpt/model__mte_visible__dynamic']}")
        MTE_ls.append(eval_result['3dpt/model__mte_visible__dynamic'])
        # print(f"OA: {eval_result['3dpt/model__occlusion_accuracy__dynamic']}")
        OA_ls.append(eval_result['3dpt/model__occlusion_accuracy__dynamic'])

        # for t in range(T):
        #     add_point_cloud(t)
        #     print(f"add point cloud: {t}")
        #     time.sleep(1.0)

    print("==============Final Results===============")
    print(f"AJ: {np.mean(AJ_ls):.2f}")
    print(f"Accuracy: {np.mean(Accuracy_ls):.2f}")
    print(f"MTE: {np.mean(MTE_ls):.2f}")
    print(f"OA: {np.mean(OA_ls):.2f}")

if __name__ == "__main__":
    for num_sample in [60]:
        main(num_sample)