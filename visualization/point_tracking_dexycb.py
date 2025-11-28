import torch
from torch.utils.data import DataLoader
import numpy as np
import viser
import time
import os 
from PIL import Image
import logging
logging.basicConfig(level=logging.INFO, filename='app.log')

from mvtracker.models.core.mvtracker.mvtracker import MVTracker
from mvtracker.datasets.utils import collate_fn
from mvtracker.datasets.dexycb_multiview_dataset import DexYCBMultiViewDataset
from visualization.utils import lift_pixels_to_world
from mvtracker.evaluation.evaluator_3dpt import evaluate_3dpt

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mvtracker = MVTracker(hidden_size=256, fmaps_dim=128)
    mvtracker.load_state_dict(torch.load("checkpoints/mvtracker_200000_june2025.pth"))
    mvtracker.to(device).eval()
    server = viser.ViserServer(host="0.0.0.0")

    dataset = DexYCBMultiViewDataset.from_name("dex-ycb-multiview-duster0123-removehand", "../../datasets/DexYCB/")
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

        with torch.no_grad():
            results = mvtracker(
                rgbs=rgbs[None] / 255.0,
                depths=depths[None],
                intrs=intrs[None],
                extrs=extrs[None],
                query_points=query_points[None],
            )
        
        pred_tracks = results["traj_e"].to(device)  # [1,T,N,3] tensor
        pred_vis = results["vis_e"].to(device)   

        eval_result = evaluate_3dpt(
            gt_tracks = trajectory,
            gt_visibilities = visibility_any,
            pred_tracks = pred_tracks[0].detach().cpu().numpy(),
            pred_visibilities = (pred_vis[0]>0.5).detach().cpu().numpy(),
            evaluation_setting = "dexycb-multiview",
            track_upscaling_factor = track_upscaling_factor,   
            query_points = query_points.detach().cpu().numpy(),
        )

        print("================Results================")
        print(f"AJ: {eval_result['3dpt/model__average_jaccard__dynamic-static-mean']}")
        AJ_ls.append(eval_result['3dpt/model__average_jaccard__dynamic-static-mean'])
        print(f"Accuracy: {eval_result['3dpt/model__average_pts_within_thresh__dynamic-static-mean']}")
        Accuracy_ls.append(eval_result['3dpt/model__average_pts_within_thresh__dynamic-static-mean'])
        print(f"MTE: {eval_result['3dpt/model__mte_visible__dynamic-static-mean']}")
        MTE_ls.append(eval_result['3dpt/model__mte_visible__dynamic-static-mean'])
        print(f"OA: {eval_result['3dpt/model__occlusion_accuracy__dynamic-static-mean']}")
        OA_ls.append(eval_result['3dpt/model__occlusion_accuracy__dynamic-static-mean'])
        
        # for t in range(T):
        #     add_point_cloud(t)
        #     time.sleep(1.0)

    print("==============Final Results===============")
    print(f"AJ: {np.mean(AJ_ls):.2f}")
    print(f"Accuracy: {np.mean(Accuracy_ls):.2f}")
    print(f"MTE: {np.mean(MTE_ls):.2f}")
    print(f"OA: {np.mean(OA_ls):.2f}")


if __name__ == "__main__":
    main()