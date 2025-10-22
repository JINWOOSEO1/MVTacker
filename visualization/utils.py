import torch
import numpy as np
import time

def lift_pixels_to_world(depths, intrs, extrs):
    """
    depths: (1, H, W) depth map
    intrs: (3, 3) intrinsic matrix
    extrs: (4, 4) extrinsic matrix (world → cam)

    world_coords: (3, H, W)
    """
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