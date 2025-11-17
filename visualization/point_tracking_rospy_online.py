import torch
import numpy as np
from huggingface_hub import hf_hub_download
import viser
import time
import os
import sys
from PIL import Image
import random
from mvtracker.models.core.mvtracker.mvtracker_online import MVTrackerOnline
from visualization.utils import lift_pixels_to_world
import rospy
from argparse import ArgumentParser
import cv2
from sensor_msgs.msg import CompressedImage, CameraInfo
import pdb

main_camera_index = 2

def ros_compressed_image_to_cv_image(
    msg: CompressedImage, encoding=cv2.IMREAD_UNCHANGED
):
    # Decompress the image data, jpeg byte array to cv2 image
    np_arr = np.fromstring(msg.data, np.uint8)
    if np_arr.size == 0:
        rospy.logerr("Received empty image data")
        return None
    cv_image = cv2.imdecode(np_arr, encoding)
    return cv_image

class DataCaptureNode:
    def __init__(self, num_cameras=1, crop_regions_path=None):
        self.window_len = 12
        self.num_new_img = 0

        self.num_cameras = num_cameras

        self.rgb_subscribers = []
        self.depth_subscribers = []
        self.camera_info_subscribers = []

        self.rgb_images = [[] for _ in range(num_cameras)]
        self.depth_images = [[] for _ in range(num_cameras)]
        self.camera_infos = [[] for _ in range(num_cameras)]

        self.crop_regions = None
        if crop_regions_path is not None:
            self.crop_regions = np.load(crop_regions_path, allow_pickle=True)

        for i in range(num_cameras):
            rgb_topic = f"/cam_{i+1}/rgb/compressed"
            depth_topic = f"/cam_{i+1}/depth/compressed"
            camera_info_topic = f"/cam_{i+1}/rgb/camera_info"

            self.rgb_subscribers.append(
                rospy.Subscriber(
                    rgb_topic, CompressedImage, self.rgb_callback, callback_args=i
                )
            )
            self.depth_subscribers.append(
                rospy.Subscriber(
                    depth_topic, CompressedImage, self.depth_callback, callback_args=i
                )
            )
            self.camera_info_subscribers.append(
                rospy.Subscriber(
                    camera_info_topic,
                    CameraInfo,
                    self.camera_info_callback,
                    callback_args=i,
                )
            )

    def rgb_callback(self, msg, camera_index):
        cv_image = ros_compressed_image_to_cv_image(msg, cv2.IMREAD_COLOR)

        if self.crop_regions is not None:
            x1, y1, x2, y2 = self.crop_regions[camera_index]
            cv_image = cv_image[y1:y2, x1:x2]

        TARGET_SIZE = (640, 360) 
        if cv_image.shape[:2] != (TARGET_SIZE[1], TARGET_SIZE[0]):
            cv_image = cv2.resize(cv_image, TARGET_SIZE, 
                                interpolation=cv2.INTER_LINEAR)
            
        self.rgb_images[camera_index].append(cv_image.astype(np.uint8))
        if(len(self.rgb_images[camera_index]) > self.window_len):
            self.rgb_images[camera_index].pop(0)
        if camera_index == main_camera_index:
            self.num_new_img = self.num_new_img + 1

    def depth_callback(self, msg, camera_index):
        cv_image = ros_compressed_image_to_cv_image(msg, cv2.IMREAD_UNCHANGED)

        if self.crop_regions is not None:
            x1, y1, x2, y2 = self.crop_regions[camera_index]
            cv_image = cv_image[y1:y2, x1:x2]

        TARGET_SIZE = (640, 360) 
        if cv_image.shape[:2] != (TARGET_SIZE[1], TARGET_SIZE[0]):
            cv_image = cv2.resize(cv_image, TARGET_SIZE, 
                                interpolation=cv2.INTER_NEAREST)

        self.depth_images[camera_index].append(cv_image)
        if(len(self.depth_images[camera_index]) > self.window_len):
            self.depth_images[camera_index].pop(0)

    def camera_info_callback(self, msg: CameraInfo, camera_index):
        if len(self.camera_infos[camera_index]) is not self.window_len:
            K = list(msg.K)
            cx = K[2]
            cy = K[5]
            if self.crop_regions is not None:
                x1, y1, x2, y2 = self.crop_regions[camera_index]
                cx -= x1
                cy -= y1
                K[2] = cx
                K[5] = cy
            # Scaling for resizing to (640, 360 )
            K[0] = K[0]/2
            K[4] = K[4]/2
            K[2] = K[2]/2
            K[5] = K[5]/2
            
            self.camera_infos[camera_index].append(np.array(K).reshape(3,3))

    def get_camera_intrinsic(self, index):
        if self.camera_infos[index] is None:
            return None
        return np.array(self.camera_infos[index].K).reshape(3, 3)

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    capture_node = DataCaptureNode(num_cameras=args.num_cameras, crop_regions_path=args.crop_regions_path)
    
    mvtracker = MVTrackerOnline(hidden_size=256, fmaps_dim=128, capture_node=capture_node) 
    mvtracker.load_state_dict(torch.load("checkpoints/mvtracker_200000_june2025.pth"))
    mvtracker.to(device).eval()
    mvtracker.init_video_online_processing()

    server = viser.ViserServer(host="0.0.0.0")
    rospy.init_node("data_capture_node", anonymous=True)

    # Extrinsic matrix & Query points
    base_dir = "../../datasets/rospy_input_1021"
    extrs = np.zeros((args.num_cameras, 4, 4), dtype = np.float32)
    for b in range(args.num_cameras):
        extr_file_path = os.path.join(base_dir, f"cam_{b+1}_extrinsics.npy")
        extr_mat = np.load(extr_file_path).astype(np.float32) 
        extr_mat = extr_mat @ np.diag([1, -1, -1, 1])
        extrs[b] = np.linalg.inv(extr_mat)

    query_points_path = os.path.join(base_dir,"query_point.npy")
    query_points = np.load(query_points_path).astype(np.float32)
    zeros = np.zeros((query_points.shape[0],1), dtype = np.float32)
    query_points = np.hstack((zeros, query_points)) # (20,4)
    query_points = torch.from_numpy(query_points).to(device)

    B = args.num_cameras
    N = query_points.shape[0]
    K = 12 # the number of frames that needs to operate mvtracker
    H, W = 360, 640 # need to change compatible with new input

    def render_image(rgb, depth, intr, extr, channel_last=False):
        if channel_last:
            rgb = rgb.transpose(0, 1, 4, 2, 3)   # (B, K, 3, H, W)
            depth = depth.transpose(0, 1, 4, 2, 3) # (B, K, 1, H, W)

        rgb_window = torch.from_numpy(rgb).to(device)
        depth_window = torch.from_numpy(depth).to(device)
        intr_window = torch.from_numpy(intr).to(device)
        extr_window = torch.from_numpy(extr).to(device)

        with torch.no_grad():
            start_time = time.time()        
            results = mvtracker(
                rgbs=rgb_window[None] / 255.0,   
                depths=depth_window[None],       
                intrs=intr_window[None],         
                extrs=extr_window[None,:,:,0:3,:],        
                query_points=query_points[None],
            )    
            print("Tracking time takes: ", time.time()-start_time)

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

            track_data = pred_tracks[0,t].detach().cpu().numpy()
            color = np.array([0.76, 0.96, 0.15], dtype=np.float32)
            server.scene.add_point_cloud(
                    name="/query_point_cloud",
                    points=track_data,
                    colors=np.tile(color, (track_data.shape[0], 1)),
                    point_size=0.01,
            )

        start_time = time.time()
        for t in range(mvtracker.step):
            add_point_cloud(t)
        print("Rendering time takes: ", time.time()-start_time)
        
    while not rospy.is_shutdown():
        if  len(capture_node.rgb_images[main_camera_index]) < capture_node.window_len or \
            len(capture_node.depth_images[main_camera_index]) < capture_node.window_len or \
            len(capture_node.camera_infos[main_camera_index]) < capture_node.window_len:
            
            rospy.loginfo_throttle(1.0, "Waiting for data from all cameras...")
            rospy.sleep(0.1) 
            continue
    
        print(capture_node.num_new_img, " new images received.")
        capture_node.num_new_img = 0

        rgbs = np.array(capture_node.rgb_images, dtype=np.float32).copy() # (B, K, H, W, 3)
        depths = np.array(capture_node.depth_images, dtype=np.float32).copy()[:,:,:,:,None] / 1000.0 # (B, K, H, W, 1)
        intrs = np.array(capture_node.camera_infos, dtype=np.float32) # (B, K, 3, 3)
        _extrs = np.tile(extrs[:, None, : ,:], (1, capture_node.window_len, 1, 1)) # (B, K, 4, 4)

        s_time = time.time()
        render_image(rgbs, depths, intrs, _extrs, channel_last=True) 
        print("Total time takes: ", time.time()-s_time, "\n")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--num_cameras", type=int, default=1, help="Number of cameras to subscribe to"
    )
    parser.add_argument(
        "--crop_regions_path",
        type=str,
        default=None,
        help="Path to load the cropping regions",
    )
    parser.add_argument(
        "--save_dir", type=str, default=None, help="Path to save the rgbd stream"
    )
    args = parser.parse_args()

    if args.save_dir is not None:
        os.makedirs(os.path.dirname(args.save_dir), exist_ok=True)

        for i in range(args.num_cameras):
            os.makedirs(os.path.join(args.save_dir, f"{i+1}"), exist_ok=True)
    
    main(args)