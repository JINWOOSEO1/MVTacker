# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import glob
import io
import logging
import os
import pickle
import re
import sys
from pathlib import Path
from typing import *

import matplotlib
import mediapy as media
import numpy as np
#import rerun as rr
import torch
from PIL import Image
from scipy.spatial.transform import Rotation as R

from mvtracker.datasets.utils import Datapoint, transform_scene

DatasetElement = Mapping[str, Mapping[str, Union[np.ndarray, str]]]


def resize_video(video: np.ndarray, output_size: Tuple[int, int]) -> np.ndarray:
    """Resize a video to output_size."""
    # If you have a GPU, consider replacing this with a GPU-enabled resize op,
    # such as a jitted jax.image.resize.  It will make things faster.
    return media.resize_video(video, output_size)


def sample_queries_first(
        target_occluded: np.ndarray,
        target_points: np.ndarray,
        frames: np.ndarray,
) -> Mapping[str, np.ndarray]:
    """Package a set of frames and tracks for use in TAPNet evaluations.
    Given a set of frames and tracks with no query points, use the first
    visible point in each track as the query.
    Args:
      target_occluded: Boolean occlusion flag, of shape [n_tracks, n_frames],
        where True indicates occluded.
      target_points: Position, of shape [n_tracks, n_frames, 2], where each point
        is [x,y] scaled between 0 and 1.
      frames: Video tensor, of shape [n_frames, height, width, 3].  Scaled between
        -1 and 1.
    Returns:
      A dict with the keys:
        video: Video tensor of shape [1, n_frames, height, width, 3]
        query_points: Query points of shape [1, n_queries, 3] where
          each point is [t, y, x] scaled to the range [-1, 1]
        target_points: Target points of shape [1, n_queries, n_frames, 2] where
          each point is [x, y] scaled to the range [-1, 1]
    """
    valid = np.sum(~target_occluded, axis=1) > 0
    target_points = target_points[valid, :]
    target_occluded = target_occluded[valid, :]

    query_points = []
    for i in range(target_points.shape[0]):
        index = np.where(target_occluded[i] == 0)[0][0]
        x, y = target_points[i, index, 0], target_points[i, index, 1]
        query_points.append(np.array([index, x, y]))  # [t, x, y]
    query_points = np.stack(query_points, axis=0)

    return {
        "video": frames[np.newaxis, ...],
        "query_points": query_points[np.newaxis, ...],
        "target_points": target_points[np.newaxis, ...],
        "occluded": target_occluded[np.newaxis, ...],
    }


def sample_queries_strided(
        target_occluded: np.ndarray,
        target_points: np.ndarray,
        frames: np.ndarray,
        query_stride: int = 5,
) -> Mapping[str, np.ndarray]:
    """Package a set of frames and tracks for use in TAPNet evaluations.

    Given a set of frames and tracks with no query points, sample queries
    strided every query_stride frames, ignoring points that are not visible
    at the selected frames.

    Args:
      target_occluded: Boolean occlusion flag, of shape [n_tracks, n_frames],
        where True indicates occluded.
      target_points: Position, of shape [n_tracks, n_frames, 2], where each point
        is [x,y] scaled between 0 and 1.
      frames: Video tensor, of shape [n_frames, height, width, 3].  Scaled between
        -1 and 1.
      query_stride: When sampling query points, search for un-occluded points
        every query_stride frames and convert each one into a query.

    Returns:
      A dict with the keys:
        video: Video tensor of shape [1, n_frames, height, width, 3].  The video
          has floats scaled to the range [-1, 1].
        query_points: Query points of shape [1, n_queries, 3] where
          each point is [t, y, x] scaled to the range [-1, 1].
        target_points: Target points of shape [1, n_queries, n_frames, 2] where
          each point is [x, y] scaled to the range [-1, 1].
        trackgroup: Index of the original track that each query point was
          sampled from.  This is useful for visualization.
    """
    tracks = []
    occs = []
    queries = []
    trackgroups = []
    total = 0
    trackgroup = np.arange(target_occluded.shape[0])
    for i in range(0, target_occluded.shape[1], query_stride):
        mask = target_occluded[:, i] == 0
        query = np.stack(
            [
                i * np.ones(target_occluded.shape[0:1]),
                target_points[:, i, 1],
                target_points[:, i, 0],
            ],
            axis=-1,
        )
        queries.append(query[mask])
        tracks.append(target_points[mask])
        occs.append(target_occluded[mask])
        trackgroups.append(trackgroup[mask])
        total += np.array(np.sum(target_occluded[:, i] == 0))

    return {
        "video": frames[np.newaxis, ...],
        "query_points": np.concatenate(queries, axis=0)[np.newaxis, ...],
        "target_points": np.concatenate(tracks, axis=0)[np.newaxis, ...],
        "occluded": np.concatenate(occs, axis=0)[np.newaxis, ...],
        "trackgroup": np.concatenate(trackgroups, axis=0)[np.newaxis, ...],
    }


class TapVidDataset(torch.utils.data.Dataset):

    @staticmethod
    def from_name(dataset_name: str, dataset_root: str):
        """
        Examples of datasets supported by this factory method:
        - tapvid2d-davis-nodepth
        - tapvid2d-davis-moge
        - tapvid2d-davis-zoedepth
        - tapvid2d-davis-videodepthanything
        - tapvid2d-davis-megasam
        - tapvid2d-davis-mogewithextrinsics
        - tapvid2d-davis-mogewithextrinsics-256x256
        - tapvid2d-davis-mogewithextrinsics-256x256-single
        """
        if dataset_name.startswith("tapvid2d-davis-"):
            # Parse the dataset name, chunk by chunk
            non_parsed = dataset_name.replace("tapvid2d-davis-", "", 1)

            # Extract depth estimator (until first possible resolution or single flag)
            match = re.match(r"([^-]+)", non_parsed)
            assert match is not None
            depth_estimator_name = match.group(1)
            non_parsed = non_parsed.replace(depth_estimator_name, "", 1)

            # Extract resolution
            resize_to = None
            match = re.search(r"-([0-9]+x[0-9]+)", non_parsed)
            if match:
                width, height = map(int, match.group(1).split("x"))
                resize_to = (width, height)
                non_parsed = non_parsed.replace(match.group(0), "", 1)

            # Check for single point flag
            single_point = "-single" in non_parsed
            non_parsed = non_parsed.replace("-single", "", 1) if single_point else non_parsed

            # Ensure no unparsed parts left
            assert non_parsed == "", f"Unparsed part of the dataset name: {non_parsed}"

            data_root = os.path.join(dataset_root, "tapvid_davis/tapvid_davis.pkl")
            return TapVidDataset(
                dataset_type="davis",
                data_root=data_root,
                resize_to=resize_to,
                queried_first=True,
                depth_estimator_name=depth_estimator_name,
                depth_estimator_batch_size=2,
                depth_estimator_device="cuda",
                stream_rerun_depth_viz=False,
                save_rerun_depth_viz=False,
            )

    def __init__(
            self,
            data_root,
            dataset_type="davis",
            resize_to=(256, 256),
            queried_first=True,
            depth_estimator_name="moge-with-extrinsics",
            depth_estimator_batch_size=2,
            depth_estimator_device="cuda",
            stream_rerun_depth_viz=False,
            save_rerun_depth_viz=False,
    ):
        self.dataset_type = dataset_type
        self.resize_to = resize_to
        self.queried_first = queried_first
        if self.dataset_type == "kinetics":
            self.depth_cache_root = os.path.join(data_root, "depth_cache")
        else:
            self.depth_cache_root = os.path.join(os.path.dirname(data_root), "depth_cache")
        os.makedirs(self.depth_cache_root, exist_ok=True)
        if self.dataset_type == "kinetics":
            all_paths = glob.glob(os.path.join(data_root, "*_of_0010.pkl"))
            points_dataset = []
            for pickle_path in all_paths:
                with open(pickle_path, "rb") as f:
                    data = pickle.load(f)
                    points_dataset = points_dataset + data
            self.points_dataset = points_dataset
        else:
            with open(data_root, "rb") as f:
                self.points_dataset = pickle.load(f)
            if self.dataset_type == "davis":
                self.video_names = list(self.points_dataset.keys())
        logging.info("found %d unique videos in %s" % (len(self.points_dataset), data_root))

        self.depth_estimator_name = depth_estimator_name
        self.depth_estimator_batch_size = depth_estimator_batch_size
        self.depth_estimator_device = depth_estimator_device

        self.stream_rerun_depth_viz = stream_rerun_depth_viz
        self.save_rerun_depth_viz = save_rerun_depth_viz

        # # Dummy call all items to generate rerun visualizations
        # self.stream_rerun_depth_viz = False
        # self.save_rerun_depth_viz = True
        # for i in tqdm(range(len(self.points_dataset))):
        #     try:
        #         self[i]
        #     except Exception as e:
        #         logging.error(f"Error processing video {i}: {e}")
        #         logging.info(f"But we continue anyway")
        #         continue
        # exit()

    def __getitem__(self, index):
        if self.dataset_type == "davis":
            video_name = self.video_names[index]
        else:
            video_name = index
        frames = self.points_dataset[video_name]["video"].copy()

        if isinstance(frames[0], bytes):
            # TAP-Vid is stored and JPEG bytes rather than `np.ndarray`s.
            def decode(frame):
                byteio = io.BytesIO(frame)
                img = Image.open(byteio)
                return np.array(img)

            frames = np.array([decode(frame) for frame in frames])

        target_points = self.points_dataset[video_name]["points"].copy()
        if self.resize_to is not None:
            frames = resize_video(frames, self.resize_to)
            target_points *= np.array([self.resize_to[1] - 1, self.resize_to[0] - 1])
        else:
            target_points *= np.array([frames.shape[2] - 1, frames.shape[1] - 1])
        assert target_points[:, :, 0].min() >= 0
        assert target_points[:, :, 0].max() <= frames.shape[2] - 1
        assert target_points[:, :, 1].min() >= 0
        assert target_points[:, :, 1].max() <= frames.shape[1] - 1

        T, H, W, C = frames.shape
        N, T, D = target_points.shape

        target_occ = self.points_dataset[video_name]["occluded"].copy()
        if self.queried_first:
            converted = sample_queries_first(target_occ, target_points, frames)
        else:
            converted = sample_queries_strided(target_occ, target_points, frames)
        assert converted["target_points"].shape[1] == converted["query_points"].shape[1]

        trajs = (torch.from_numpy(converted["target_points"])[0].permute(1, 0, 2).float())  # T, N, D

        rgbs = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
        visibles = torch.logical_not(torch.from_numpy(converted["occluded"]))[0].permute(1, 0)  # T, N
        query_points_2d = torch.from_numpy(converted["query_points"])[0]  # T, N

        # Let's estimate depths RIGHT HERE
        res = f"{H}x{W}"
        cached_file_zoedepth_nk = os.path.join(self.depth_cache_root, f"zoedepth_nk__{video_name}__{res}.npz")
        cached_file_moge = os.path.join(self.depth_cache_root, f"moge__{video_name}__{res}.npz")
        cached_file_megasam = os.path.join(self.depth_cache_root, f"megasam__{video_name}__{res}-v1.npz")
        if self.depth_estimator_name == "nodepth":
            depth = np.ones((T, H, W))
            intrs = np.eye(3) * max(H, W)
            extrs = np.eye(4)[None].repeat(T, axis=0)
        elif self.depth_estimator_name == "zoedepth":
            depth = zoedepth_nk(rgbs, self.depth_estimator_batch_size, self.depth_estimator_device,
                                cached_file_zoedepth_nk)
            _, intrs, _, _, _ = moge(rgbs, self.depth_estimator_batch_size, self.depth_estimator_device,
                                     cached_file_moge)
            extrs = np.eye(4)[None].repeat(T, axis=0)
        elif self.depth_estimator_name == "moge":
            depth, intrs, _, _, mask = moge(rgbs, self.depth_estimator_batch_size, self.depth_estimator_device,
                                            cached_file_moge)
            depth[~mask] = 0
            extrs = np.eye(4)[None].repeat(T, axis=0)
        elif self.depth_estimator_name == "mogewithextrinsics":
            depth, intrs, extrs, _, mask = moge(rgbs, self.depth_estimator_batch_size, self.depth_estimator_device,
                                                cached_file_moge)
            depth[~mask] = 0
        elif self.depth_estimator_name == "videodepthanything":
            raise NotImplementedError("videodepthanything is not implemented yet")
        elif self.depth_estimator_name == "megasam":
            try:
                depth, intrs, extrs = megasam(
                    rgbs=rgbs,
                    batch_size=self.depth_estimator_batch_size,
                    device=self.depth_estimator_device,
                    cached_file=cached_file_megasam,
                )
            except Exception as e:
                logging.error(f"MegaSAM error for {video_name} ({rgbs.shape=}) (we will use moge depth instead): {e}")
                depth, intrs, extrs, _, mask = moge(rgbs, self.depth_estimator_batch_size, self.depth_estimator_device,
                                                    cached_file_moge)
                depth[~mask] = 0
        else:
            raise NotImplementedError

        depth = torch.from_numpy(depth).float()
        if intrs.ndim == 2:
            intrs = intrs[None].repeat(T, axis=0)
        intrs = torch.from_numpy(intrs).float()
        extrs_square = torch.from_numpy(extrs).float()
        extrs = extrs_square[:, :3, :]

        intrs_inv = torch.inverse(intrs)
        extrs_inv = torch.inverse(extrs_square)

        # Project trajectories to 3D
        trajs_depth = trajs.new_ones((T, N, 1)) * np.inf
        for t in range(T):
            # # V1: Not good enough, depths are jumping to the background near edges because of interpolation
            # trajs_depth[t] = bilinear_sample2d(
            #     im=depth[t][None, None],
            #     x=trajs[t, :, 0][None],
            #     y=trajs[t, :, 1][None],
            # )[0].permute(1, 0).type(trajs_depth.dtype)

            # V2: Still not good, taking the closest pixel only (without interpolating) still has jumps at edges
            x_nearest = trajs[t, :, 0].round().long()
            y_nearest = trajs[t, :, 1].round().long()
            depth_nearest = depth[t].view(-1)[(y_nearest * W + x_nearest).view(-1)]
            depth_nearest = depth_nearest.view(1, -1).type(trajs_depth.dtype).permute(1, 0)
            trajs_depth[t] = depth_nearest

            # # V3: Taking the minimum depth value of the neighbors also fails when there are other things in front.
            # depth_pad = F.pad(depth[t][None, None], (1, 1, 1, 1), mode="replicate")  # Pad to handle edges
            # depth_min = -F.max_pool2d(-depth_pad, kernel_size=9, stride=1)  # Min pooling using negation
            # depth_min_sampled = depth_min[0, 0, trajs[t, :, 1].long(), trajs[t, :, 0].long()].type(trajs_depth.dtype)
            # trajs_depth[t] = depth_min_sampled[:, None]
        assert torch.all(torch.isfinite(trajs_depth)).item()
        trajs_camera = torch.einsum("Tij,TNj->TNi", intrs_inv, to_homogenous_torch(trajs)) * trajs_depth
        trajs_world = torch.einsum("Tij,TNj->TNi", extrs_inv, to_homogenous_torch(trajs_camera))[..., :3]
        trajs_3d = trajs_world

        trajs_w_z = torch.cat([trajs, trajs_depth], dim=2)

        # Project query points to 3D
        qp_t = query_points_2d[:, 0].float()
        qp_xyz_pixel = query_points_2d[:, 1:].float()
        qp_depth = qp_xyz_pixel.new_ones((N, 1)) * np.inf
        qp_xyz_world = qp_xyz_pixel.new_ones((N, 3)) * np.inf
        for t in range(T):
            qp_mask = qp_t == t
            if qp_mask.sum() == 0:
                continue

            # V2 depth interpolation
            x_nearest = qp_xyz_pixel[qp_mask, 0].round().long()
            y_nearest = qp_xyz_pixel[qp_mask, 1].round().long()
            depth_nearest = depth[t].view(-1)[(y_nearest * W + x_nearest).view(-1)]
            depth_nearest = depth_nearest.view(1, -1).type(trajs_depth.dtype).permute(1, 0)
            qp_depth[qp_mask] = depth_nearest

            qp_xyz_pixel_t = to_homogenous_torch(qp_xyz_pixel[qp_mask])
            qp_xyz_camera_t = torch.einsum("ij,Nj->Ni", intrs_inv[t], qp_xyz_pixel_t) * qp_depth[qp_mask]
            qp_xyz_world_t = torch.einsum("ij,Nj->Ni", extrs_inv[t], to_homogenous_torch(qp_xyz_camera_t))[..., :3]
            qp_xyz_world[qp_mask] = qp_xyz_world_t
        assert torch.all(torch.isfinite(qp_depth))
        assert torch.all(torch.isfinite(qp_xyz_world))
        query_points_3d = torch.cat([qp_t[:, None], qp_xyz_world], dim=1)

        # Visualize the depth estimation in Rerun
        radii_scale = 0.1
        streams = []
        if self.stream_rerun_depth_viz: streams += [True]
        if self.save_rerun_depth_viz: streams += [False]
        for stream in streams:
            # depth_zoedepth = zoedepth_nk(rgbs, self.depth_estimator_batch_size, self.depth_estimator_device,
            #                              cached_file_zoedepth_nk)
            depth_moge, intrinsics_moge, w2c_moge, _, mask_moge = moge(
                rgbs=rgbs,
                batch_size=self.depth_estimator_batch_size,
                device=self.depth_estimator_device,
                cached_file=cached_file_moge,
            )

            # TODO: But what intrinsics did Zoe really assume or use, if any?
            K = intrinsics_moge
            K_inv = np.linalg.inv(K)

            rr.init("TAPVid-2D Estimated Depths", recording_id="v0.1")
            if stream:
                rr.connect_tcp()
            rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

            rr.set_time_seconds("frame", 0)
            rr.log(
                "world/xyz",
                rr.Arrows3D(
                    vectors=[[1, 0, 0], [0, 2, 0], [0, 0, 3]],
                    colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
                ),
            )
            for t in range(T):
                rr.set_time_seconds("frame", t / 12)
                rgb = rgbs[t].permute(1, 2, 0).numpy()

                # Log the depth used for 3D tracking
                rr.log(f"{video_name}/image/depth_for_tracking", rr.Pinhole(
                    image_from_camera=intrs[t].numpy(),
                    width=W,
                    height=H,
                ))
                rr.log(f"{video_name}/image/depth_for_tracking", rr.Transform3D(
                    translation=np.linalg.inv(extrs_square[t].numpy())[:3, 3],
                    mat3x3=np.linalg.inv(extrs_square[t].numpy())[:3, :3],
                ))
                rr.log(f"{video_name}/image/depth_for_tracking/depth", rr.DepthImage(
                    image=depth[t].numpy(),
                    point_fill_ratio=0.2,
                ))
                rr.log(f"{video_name}/image/depth_for_tracking/rgb", rr.Image(rgb))

                # Log all other depth maps
                # d_zoe = depth_zoedepth[t, 0]
                d_moge = depth_moge[t]
                c2w_moge = np.linalg.inv(w2c_moge[t])
                # for name, archetype in [
                #     ("depth-zoe", rr.DepthImage(d_zoe, point_fill_ratio=0.2)),
                #     ("depth-moge", rr.DepthImage(d_moge, point_fill_ratio=0.2)),
                #     ("depth-moge-with-extrinsics", rr.DepthImage(d_moge, point_fill_ratio=0.2)),
                # ]:
                #     rr.log(f"{video_name}/image/{name}", rr.Pinhole(image_from_camera=K, width=W, height=H))
                #     rr.log(f"{video_name}/image/{name}/{name}", archetype)
                #     if name == "depth-moge-with-extrinsics":
                #         transform = rr.Transform3D(translation=c2w_moge[:3, 3], mat3x3=c2w_moge[:3, :3])
                #         rr.log(f"{video_name}/image/{name}", transform)

                # Convert depth map to 3D point cloud
                y, x = np.indices((H, W))
                homo_pixel_coords = np.stack([x.ravel(), y.ravel(), np.ones_like(x).ravel()], axis=1).T

                for _name, _depth, _w2c in [
                    ("used_for_tracking", depth[t].numpy(), extrs_square[t]),
                    # ("zoe", d_zoe, None),
                    ("moge", d_moge, w2c_moge[t]),
                    ("moge-with-extrinsics", d_moge, w2c_moge[t]),
                ]:
                    depth_values = _depth.ravel()
                    cam_coords = (K_inv @ homo_pixel_coords) * depth_values
                    if _w2c is None:
                        world_coords = cam_coords.T
                    else:
                        world_coords = from_homogeneous(
                            np.einsum("ij,Nj->Ni", np.linalg.inv(_w2c), to_homogeneous(cam_coords.T)))
                    valid_mask = depth_values > 0
                    world_coords = world_coords[valid_mask]
                    rgb_colors = rgb.reshape(-1, 3)[valid_mask].astype(np.uint8)
                    rr.log(f"{video_name}/pointcloud/{_name}",
                           rr.Points3D(world_coords, colors=rgb_colors, radii=0.001))

            def log_tracks(
                    tracks: np.ndarray,
                    visibles: np.ndarray,
                    query_timestep: np.ndarray,
                    colors: np.ndarray,
                    track_names=None,

                    entity_format_str="{}",

                    log_points=True,
                    points_radii=0.03 * radii_scale,
                    invisible_color=[0., 0., 0.],

                    log_line_strips=True,
                    max_strip_length_past=6,
                    max_strip_length_future=1,
                    hide_invisible_strips=True,
                    strips_radii=0.0027 * radii_scale,

                    log_error_lines=False,
                    error_lines_radii=0.0042 * radii_scale,
                    error_lines_color=[1., 0., 0.],
                    gt_for_error_lines=None,
            ) -> None:
                """
                Log tracks to Rerun.

                Parameters:
                    tracks: Shape (T, N, 3), the 3D trajectories of points.
                    visibles: Shape (T, N), boolean visibility mask for each point at each timestep.
                    query_timestep: Shape (T, N), the frame index after which the tracks start.
                    colors: Shape (N, 4), RGBA colors for each point.
                    entity_prefix: String prefix for entity hierarchy in Rerun.
                    entity_suffix: String suffix for entity hierarchy in Rerun.
                """

                T, N, _ = tracks.shape
                assert tracks.shape == (T, N, 3)
                assert visibles.shape == (T, N)
                assert query_timestep.shape == (N,)
                assert query_timestep.min() >= 0
                assert query_timestep.max() < T
                assert colors.shape == (N, 4)

                for n in range(N):
                    track_name = track_names[n] if track_names is not None else f"track-{n}"
                    rr.log(entity_format_str.format(track_name, rr.Clear(recursive=True)))
                    for t in range(query_timestep[n], T):
                        rr.set_time_seconds("frame", t / 12)

                        # Log the point (special handling for invisible points)
                        if log_points:
                            rr.log(
                                entity_format_str.format(f"{track_name}/point"),
                                rr.Points3D(
                                    positions=[tracks[t, n]],
                                    colors=[colors[n, :3]] if visibles[t, n] else [invisible_color],
                                    radii=points_radii,
                                ),
                            )

                        # Log line segments for visible tracks
                        if log_line_strips and t > query_timestep[n]:
                            strip_t_start = max(t - max_strip_length_past, query_timestep[n].item())
                            strip_t_end = min(t + max_strip_length_future, T - 1)

                            if not hide_invisible_strips:
                                strips = np.stack([
                                    tracks[strip_t_start:strip_t_end, n],
                                    tracks[strip_t_start + 1:strip_t_end + 1, n],
                                ], axis=-2)
                                strips_visibility = visibles[strip_t_start + 1:strip_t_end + 1, n]
                                strips_colors = np.where(
                                    strips_visibility[:, None],
                                    colors[None, n, :3],
                                    [invisible_color],
                                )
                            else:
                                point_sequence = tracks[strip_t_start:strip_t_end + 1, n]
                                point_sequence_visible = point_sequence[visibles[strip_t_start:strip_t_end + 1, n]]
                                strips = np.stack([point_sequence_visible[:-1], point_sequence_visible[1:]], axis=-2)
                                strips_colors = colors[None, n, :3]

                            rr.log(
                                entity_format_str.format(f"{track_name}/line"),
                                rr.LineStrips3D(strips=strips, colors=strips_colors, radii=strips_radii),
                            )

                        if log_error_lines:
                            assert gt_for_error_lines is not None
                            strips = np.stack([
                                tracks[t, n],
                                gt_for_error_lines[t, n],
                            ], axis=-2)
                            rr.log(
                                entity_format_str.format(f"{track_name}/error"),
                                rr.LineStrips3D(strips=strips, colors=error_lines_color, radii=error_lines_radii),
                            )

            # Log the tracks
            trajs_3d_np = trajs_3d.cpu().numpy()
            visibles_np = visibles.cpu().numpy()
            query_timestep_np = query_points_3d[:, 0].cpu().numpy().round().astype(int)
            cmap = matplotlib.colormaps["gist_rainbow"]
            norm = matplotlib.colors.Normalize(vmin=trajs_3d_np[..., 0].min(), vmax=trajs_3d_np[..., 0].max())
            track_color = cmap(norm(trajs_3d_np[-1, :, 0]))
            # track_color = track_color * 0 + 1  # Just make all tracks white

            log_tracks(
                tracks=trajs_3d_np,
                visibles=visibles_np,
                query_timestep=query_timestep_np,
                colors=track_color,
                entity_format_str=f"{video_name}/tracks/{{}}",
                max_strip_length_future=0,
            )

            if not stream:
                rr_rrd_path = os.path.join(self.depth_cache_root, f"rerun_viz__{video_name}.rrd")
                rr.save(rr_rrd_path)
                logging.info(f"Saved Rerun recording to: {os.path.abspath(rr_rrd_path)}")

        V = 1
        rgbs = rgbs[None]
        trajs = trajs[None]
        trajs_w_z = trajs_w_z[None]
        trajs_3d = trajs_3d
        query_points_3d = query_points_3d
        visibles = visibles[None]
        depth = depth[None, :, None]
        feats = None
        intrs = intrs[None]
        extrs = extrs[None]

        assert rgbs.shape == (V, T, 3, H, W)
        assert depth.shape == (V, T, 1, H, W)
        assert feats is None
        assert intrs.shape == (V, T, 3, 3)
        assert extrs.shape == (V, T, 3, 4)
        assert trajs.shape == (V, T, N, 2)
        assert trajs_w_z.shape == (V, T, N, 3)
        assert visibles.shape == (V, T, N)
        assert trajs_3d.shape == (T, N, 3)
        assert query_points_3d.shape == (N, 4)

        # Normalize the scene to be similar to training scenes
        rot_x = R.from_euler('x', -90, degrees=True).as_matrix()
        rot_y = R.from_euler('y', 0, degrees=True).as_matrix()
        rot_z = R.from_euler('z', 0, degrees=True).as_matrix()
        rot = rot_z @ rot_y @ rot_x
        T_rot = torch.eye(4)
        T_rot[:3, :3] = torch.from_numpy(rot)

        ## V1: GT track-agnostic transformation
        # scale = 10
        # translate_x = 0
        # translate_y = -15
        # translate_z = 2
        #
        # T_scale_and_translate = torch.tensor([
        #     [scale, 0.0, 0.0, translate_x],
        #     [0.0, scale, 0.0, translate_y],
        #     [0.0, 0.0, scale, translate_z],
        #     [0.0, 0.0, 0.0, 1.0],
        # ], dtype=torch.float32)

        ## V2: GT track-aware transformation
        # Rotate the 3D GT tracks first
        trajs_3d_homo = torch.cat([trajs_3d, torch.ones_like(trajs_3d[..., :1])], dim=-1)
        trajs_3d_rotated = torch.einsum('ij,TNj->TNi', T_rot, trajs_3d_homo)[..., :3]

        # Mask out non-visible points
        visible_mask = visibles[0]  # (T, N)
        trajs_3d_visible = trajs_3d_rotated[visible_mask]  # (V, 3)

        # Compute bbox over only visible points
        bbox_min = trajs_3d_visible.amin(dim=0)
        bbox_max = trajs_3d_visible.amax(dim=0)
        bbox_center = (bbox_min + bbox_max) / 2
        bbox_size = bbox_max - bbox_min

        # Target bounds (half-extent of desired cube)
        target_bounds = torch.tensor([10.0, 10.0, 6.0])
        scale = (target_bounds / bbox_size).min().item()
        translation = -bbox_center * scale
        rot = torch.from_numpy(rot)

        # Optional: clamp depth map if needed (max Z-depth defined in scaled space)
        logging.info(f"[datapoint_idx={index}] Scale={scale:.2f}, Translate={translation.tolist()}")
        # depth[depth > 50 / scale] = 50 / scale
        depth[depth > 20] = 20

        # Apply to scene
        (
            depth_trans, extrs_trans, query_points_3d_trans, trajs_3d_trans, trajs_w_z_trans
        ) = transform_scene(scale, rot, translation, depth, extrs, query_points_3d, trajs_3d, trajs_w_z)
        assert torch.allclose(trajs_w_z[..., :2], trajs_w_z_trans[..., :2])

        gotit = True
        return Datapoint(
            video=rgbs,
            videodepth=depth_trans,
            feats=None,
            segmentation=torch.ones(T, 1, H, W).float(),
            trajectory=trajs_w_z_trans,
            trajectory_3d=trajs_3d_trans,
            visibility=visibles,
            valid=torch.ones((T, N)),
            seq_name=str(video_name),
            intrs=intrs,
            extrs=extrs_trans,
            query_points=query_points_2d,
            query_points_3d=query_points_3d_trans,
        ), gotit

    def __len__(self):
        return len(self.points_dataset)


@torch.no_grad()
def zoedepth_nk(rgbs, batch_size=2, device="cuda", cached_file=None):
    if cached_file is not None and os.path.exists(cached_file):
        return np.load(cached_file)["depth"]

    # needs timm==0.6.7, but megasam needs timm==1.0.15
    model = torch.hub.load("isl-org/ZoeDepth", "ZoeD_NK", pretrained=True).to(device)
    model.eval()

    T, _, H, W = rgbs.shape
    depth = []
    for i in range(0, T, batch_size):
        rgbs_i = rgbs[i:i + batch_size].to(device) / 255.
        depth_i = model.infer(rgbs_i).clamp(0.01, 65.0).cpu()
        depth.append(depth_i)
    depth = torch.cat(depth, dim=0).numpy()[:, 0]

    if cached_file is not None:
        np.savez(cached_file, depth=depth)

    del model
    torch.cuda.empty_cache()

    return depth


def rigid_registration(
        p: np.ndarray,
        q: np.ndarray,
        w: np.ndarray = None,
        eps: float = 1e-12
) -> Tuple[float, np.ndarray, np.ndarray]:
    from moge.utils.geometry_numpy import weighted_mean_numpy

    if w is None:
        w = np.ones(p.shape[0])
    centroid_p = weighted_mean_numpy(p, w[:, None], axis=0)
    centroid_q = weighted_mean_numpy(q, w[:, None], axis=0)

    p_centered = p - centroid_p
    q_centered = q - centroid_q
    w = w / (np.sum(w) + eps)

    cov = (w[:, None] * p_centered).T @ q_centered
    U, S, Vh = np.linalg.svd(cov)
    R = Vh.T @ U.T
    if np.linalg.det(R) < 0:
        Vh[2, :] *= -1
        R = Vh.T @ U.T
    scale = np.sum(S) / np.trace((w[:, None] * p_centered).T @ p_centered)
    t = centroid_q - scale * (centroid_p @ R.T)
    return scale, R, t


def rigid_registration_ransac(
        p: np.ndarray,
        q: np.ndarray,
        w: np.ndarray = None,
        max_iters: int = 20,
        hypothetical_size: int = 10,
        inlier_thresh: float = 0.02
) -> Tuple[Tuple[float, np.ndarray, np.ndarray], np.ndarray]:
    n = p.shape[0]
    if w is None:
        w = np.ones(p.shape[0])

    best_score, best_inlines = 0., np.zeros(n, dtype=bool)
    best_solution = (np.array(1.), np.eye(3), np.zeros(3))

    for _ in range(max_iters):
        maybe_inliers = np.random.choice(n, size=hypothetical_size, replace=False)
        try:
            s, R, t = rigid_registration(p[maybe_inliers], q[maybe_inliers], w[maybe_inliers])
        except np.linalg.LinAlgError:
            continue
        transformed_p = s * p @ R.T + t
        errors = w * np.linalg.norm(transformed_p - q, axis=1)
        inliers = errors < inlier_thresh

        score = inlier_thresh * n - np.clip(errors, None, inlier_thresh).sum()
        if score > best_score:
            best_score, best_inlines = score, inliers
            best_solution = rigid_registration(p[inliers], q[inliers], w[inliers])

    return best_solution, best_inlines


def to_homogeneous(x):
    return np.concatenate([x, np.ones_like(x[..., :1])], axis=-1)


def from_homogeneous(x, assert_homogeneous_part_is_equal_to_1=False, eps=0.001):
    if assert_homogeneous_part_is_equal_to_1:
        assert np.allclose(x[..., -1:], 1, atol=eps), f"Expected homogeneous part to be 1, got {x[..., -1:]}"
    return x[..., :-1] / x[..., -1:]


def to_homogenous_torch(x):
    return torch.cat([x, torch.ones_like(x[..., :1])], axis=-1)


@torch.no_grad()
def moge(rgbs, batch_size=10, device="cuda", cached_file=None, intrinsics=None):
    if cached_file is not None and os.path.exists(cached_file):
        cached_data = np.load(cached_file)
        depths_with_normalized_scale = cached_data["depth"]
        points_in_world_space = cached_data["points"]
        w2c = cached_data["w2c"]
        intrinsics = cached_data["intrinsics"]
        mask = cached_data["mask"]
        return depths_with_normalized_scale, intrinsics, w2c, points_in_world_space, mask

    # git clone https://github.com/microsoft/MoGe.git ../moge
    # cd ../moge
    # git checkout dd158c0
    sys.path.append("../moge")  # TODO: Find a clean way to do this so that it is not hardcoded
    from moge.model import MoGeModel
    import utils3d
    model = MoGeModel.from_pretrained("Ruicheng/moge-vitl").to(device)

    T, _, H, W = rgbs.shape
    assert rgbs.shape == (T, 3, H, W)

    points = []
    depth = []
    mask = []
    for rgb in rgbs:
        rgb = rgb.to(device)
        output = model.infer(
            image=rgb / 255,
            resolution_level=9,
            force_projection=True,
            apply_mask=True,
            fov_x=np.rad2deg(utils3d.intrinsics_to_fov(intrinsics)[0]) if intrinsics is not None else None,
        )
        points.append(output["points"].cpu().numpy())
        depth.append(output["depth"].cpu().numpy())
        mask.append(output["mask"].cpu().numpy())
        if intrinsics is None:
            intrinsics = output["intrinsics"].cpu().numpy()
        assert np.allclose(intrinsics, output["intrinsics"].cpu().numpy(), atol=0.01), "Intr. changed between frames"
    points = np.stack(points)
    depth = np.stack(depth)
    mask = np.stack(mask)
    intrinsics = np.diag([W, H, 1]) @ intrinsics

    # Assert we can reproduce the points from the depth maps already (should be enforced with force_projection=True)
    pixel_xy = np.stack(np.meshgrid(np.arange(W), np.arange(H)), axis=-1)
    pixel_xy_homo = to_homogeneous(pixel_xy)
    depthmap_camera_xyz = np.einsum('ij,HWj->HWi', np.linalg.inv(intrinsics), pixel_xy_homo)
    depthmap_camera_xyz = depthmap_camera_xyz[None, :, :, :] * depth[:, :, :, None]
    valid = mask & (depth > 0)
    assert np.allclose(points[valid], depthmap_camera_xyz[valid], atol=1, rtol=0.1)

    depths_with_normalized_scale = depth.copy()
    points_in_world_space = points.copy()
    w2c = np.eye(4)[None].repeat(T, axis=0)
    for t in range(1, T):
        valid_p = mask[t] & (depth[t] > 0)  # & (depth[t] <= 4.20)  # TODO: magic number here!
        valid_q = mask[t - 1] & (depth[t] > 0)  # & (depth[t] <= 4.20)  # TODO: magic number here!
        valid = valid_p & valid_q
        (scale, rotation, translation), inliers = rigid_registration_ransac(
            p=points[t][valid].reshape(-1, 3),
            q=points_in_world_space[t - 1][valid].reshape(-1, 3),
            w=(1 / depths_with_normalized_scale[t - 1][valid]).reshape(-1),
            max_iters=20,
            hypothetical_size=10,
            inlier_thresh=0.02
        )
        depths_with_normalized_scale[t] = scale * depths_with_normalized_scale[t]

        # Transforming points[t] -> points_in_world_space[t - 1] already tells us how to transform to the
        # world space since points_in_world_space[t - 1] had already been transformed to the world space
        points_in_world_space[t] = scale * points_in_world_space[t] @ rotation.T + translation

        # I prefer to use column vectors: Q = q.T, P = p.T
        # q = p @ R.T + t -> Q = R @ P + t.T
        # p = q @ R - t @ rotation -> P = R.T @ Q - R.T @ t.T
        w2c[t, :3, :3] = rotation.T
        w2c[t, :3, 3] = -rotation.T @ translation.T

    # Assert no nans
    assert not np.isnan(depths_with_normalized_scale).any()
    assert not np.isnan(w2c).any()
    assert np.allclose(w2c[:, 3, 3], 1)

    # Now let's make sure we can go from scale-normalized depth maps to the points in world space
    # Pixel --> Camera --> World
    pixel_xy = np.stack(np.meshgrid(np.arange(W), np.arange(H)), axis=-1)
    pixel_xy_homo = to_homogeneous(pixel_xy)
    depthmap_camera_xyz = np.einsum('ij,HWj->HWi', np.linalg.inv(intrinsics), pixel_xy_homo)
    depthmap_camera_xyz = depthmap_camera_xyz[None, :, :, :] * depths_with_normalized_scale[:, :, :, None]
    depthmap_camera_xyz_homo = to_homogeneous(depthmap_camera_xyz)
    depthmap_world_xyz_homo = np.einsum('Tij,THWj->THWi', np.linalg.inv(w2c), depthmap_camera_xyz_homo)
    depthmap_world_xyz = from_homogeneous(depthmap_world_xyz_homo)
    points_in_world_space_reproduced = depthmap_world_xyz
    valid = mask & (depths_with_normalized_scale > 0)
    assert np.allclose(points_in_world_space[valid], points_in_world_space_reproduced[valid], atol=0.1, rtol=0.1)

    if cached_file is not None:
        np.savez(
            cached_file,
            depth=depths_with_normalized_scale,
            points=points_in_world_space,
            w2c=w2c,
            intrinsics=intrinsics,
            mask=mask,
        )

    return depths_with_normalized_scale, intrinsics, w2c, points_in_world_space, mask


def megasam(rgbs: torch.Tensor, batch_size: int = 10, device: str = "cuda", cached_file: Optional[str] = None):
    if cached_file is not None and os.path.exists(cached_file):
        cached_data = np.load(cached_file)
        return (
            cached_data["depths"].astype(np.float32),
            cached_data["intrinsics"].astype(np.float32),
            cached_data["extrinsics"].astype(np.float32),
        )
    # else:
    #     raise NotImplementedError("TMP ERR")

    T, C, H, W = rgbs.shape
    assert C == 3, "Expected shape (T, 3, H, W)"

    # Convert to NumPy format for MegaSAM (T, H, W, 3), uint8 [0, 255]
    rgbs_np = (rgbs.permute(0, 2, 3, 1).cpu().numpy()).astype(np.uint8)

    # git clone https://github.com/zbw001/TAPIP3D.git ../tapip3d
    # cd ../tapip3d
    # git checkout 8871375
    sys.path.append("../tapip3d")
    from annotation.megasam import MegaSAMAnnotator
    megasam = MegaSAMAnnotator(
        script_path=Path("../tapip3d") / "third_party" / "megasam" / "inference.py",
        depth_model="moge",
        resolution=H * W
    )
    megasam.to(device)
    depths, intrinsics, extrinsics = megasam.process_video(
        rgbs=rgbs_np,
        gt_intrinsics=None,
        return_raw_depths=False,
    )

    if cached_file is not None:
        np.savez(cached_file, depths=depths, intrinsics=intrinsics, extrinsics=extrinsics)
    return depths, intrinsics, extrinsics
