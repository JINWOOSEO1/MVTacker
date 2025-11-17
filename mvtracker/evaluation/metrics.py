import logging
import warnings
from typing import Mapping

import numpy as np
import pandas as pd
import torch


def compute_metrics(
        query_points,
        gt_occluded,
        gt_tracks,
        pred_occluded,
        pred_tracks,
        distance_thresholds=[1, 2, 4, 8, 16],
        survival_distance_threshold=50,
        query_mode="first",
):
    n_batches, n_frames, n_points, n_point_dim = gt_tracks.shape

    # First, we compute the original TAP-Vid metrics
    tapvid_metrics = compute_tapvid_metrics(query_points, gt_occluded, gt_tracks, pred_occluded,
                                            pred_tracks, distance_thresholds, query_mode)


    # Compute distances only for visible points
    visible_mask = ~gt_occluded
    distances = torch.norm(pred_tracks - gt_tracks, dim=-1)
    distances[~visible_mask] = float('nan')
    distances[torch.arange(n_frames)[None, :, None] < query_points[:, :, 0].long()[:, None, :]] = float('nan')

    # Compute Median Trajectory Error (MTE) and Average Trajectory Error (ATE) for visible points
    mte_per_track = torch.nanmedian(distances, dim=1).values
    ate_per_track = torch.nanmean(distances, dim=1)
    assert torch.isnan(mte_per_track).sum() == 0
    assert torch.isnan(ate_per_track).sum() == 0

    # Compute Final Trajectory Error (FDE) for the last visible frame
    last_visible_idx = torch.argmax(visible_mask * np.arange(n_frames)[None, :, None], dim=1)
    fde_per_track = distances[torch.arange(n_batches)[:, None], last_visible_idx, torch.arange(n_points)]

    # Compute "Survival" rate for visible points
    tracking_failed = (distances > survival_distance_threshold) * visible_mask
    failure_index = tracking_failed.float().argmax(dim=1)
    failure_index[(~tracking_failed).all(dim=1)] = n_frames  # If all points survived, survival is 1.0
    survival_per_track = (failure_index - query_points[:, :, 0].long()) / (n_frames - query_points[:, :, 0].long())

    assert mte_per_track.shape == ate_per_track.shape == survival_per_track.shape == fde_per_track.shape

    metrics = {
        'mte_visible_per_track': mte_per_track,
        'ate_visible_per_track': ate_per_track,
        'fde_visible_per_track': fde_per_track,
        'survival_per_track': survival_per_track,
        **tapvid_metrics,
    }

    return metrics


def compute_tapvid_metrics(
        query_points,
        gt_occluded,
        gt_tracks,
        pred_occluded,
        pred_tracks,
        distance_thresholds,
        query_mode="first",
):
    """
    Computes metrics from TAP-Vid (https://arxiv.org/abs/2211.03726) based on given ground truth and predictions.
    The computations are performed separately for each video in the batch.

    Parameters
    ----------
    query_points : torch.Tensor
        Tensor of shape (n_batches, n_points, 3) representing the query points.
    gt_occluded : torch.Tensor
        Boolean tensor of shape (n_batches, n_frames, n_points) indicating if a point is occluded in the ground truth.
    gt_tracks : torch.Tensor
        Tensor of shape (n_batches, n_frames, n_points, n_point_dim) representing the ground truth tracks.
    pred_occluded : torch.Tensor
        Boolean tensor of shape (n_batches, n_frames, n_points) indicating if a point is occluded in the predictions.
    pred_tracks : torch.Tensor
        Tensor of shape (n_batches, n_frames, n_points, n_point_dim) representing the predicted tracks.
    query_mode : str, optional
        Either "first" or "strided", default is "first". Indicates how the query points are sampled.

    Returns
    -------
    dict
        A dictionary containing:
        - 'occlusion_accuracy_per_track': Accuracy at predicting occlusion, per track.
        - 'pts_within_{x}_per_track' for x in [1, 2, 4, 8, 16]: Fraction of points predicted to be within the given pixel threshold, ignoring occlusion prediction, per track.
        - 'jaccard_{x}_per_track' for x in [1, 2, 4, 8, 16]: Jaccard metric for the given pixel threshold. Combines occlusion and point prediction accuracy, per track.
        - 'average_jaccard_per_track': Average Jaccard metric across thresholds, per track.
        - 'average_pts_within_thresh_per_track': Average fraction of points within threshold across thresholds, per track.
    """

    metrics = {}

    # Check shapes.
    n_batches, n_frames, n_points, n_point_dim = gt_tracks.shape
    assert n_point_dim in [2, 3]
    assert query_points.shape == (n_batches, n_points, n_point_dim + 1)
    assert gt_occluded.shape == (n_batches, n_frames, n_points)
    assert gt_tracks.shape == (n_batches, n_frames, n_points, n_point_dim)
    assert pred_occluded.shape == (n_batches, n_frames, n_points)
    assert pred_tracks.shape == (n_batches, n_frames, n_points, n_point_dim)
    assert query_mode in ["first", "strided"]
    assert query_points.dtype == torch.float32
    assert gt_occluded.dtype == torch.bool
    assert gt_tracks.dtype == torch.float32
    assert pred_occluded.dtype == torch.bool
    assert pred_tracks.dtype == torch.float32

    # Don't evaluate the query point.
    evaluation_points = torch.ones_like(gt_occluded, dtype=torch.bool)
    for batch_idx in range(n_batches):
        t = query_points[batch_idx, :, 0].long()
        evaluation_points[batch_idx, t, torch.arange(n_points)] = False

    # In first query mode, don't evaluate points before the query point.
    if query_mode == "first":
        t = query_points[:, :, 0].long()
        mask = torch.arange(n_frames).unsqueeze(-1) < t.unsqueeze(1)
        evaluation_points[mask] = False

    # Compute occlusion accuracy per track.
    occ_acc = ((pred_occluded == gt_occluded) & evaluation_points).float().sum(dim=1) / evaluation_points.sum(dim=1)
    metrics["occlusion_accuracy_per_track"] = occ_acc

    # Let's report the numbers separately for gt=0 and gt=1
    numer0 = ((pred_occluded == gt_occluded) & (gt_occluded == 1) & evaluation_points).float().sum(dim=1)
    numer1 = ((pred_occluded == gt_occluded) & (gt_occluded == 0) & evaluation_points).float().sum(dim=1)
    denom0 = ((gt_occluded == 1) & evaluation_points).float().sum(dim=1)
    denom1 = ((gt_occluded == 0) & evaluation_points).float().sum(dim=1)
    occ_acc_for_vis0 = numer0 / denom0
    occ_acc_for_vis1 = numer1 / denom1
    metrics["occlusion_accuracy_for_vis0_per_track"] = occ_acc_for_vis0
    metrics["occlusion_accuracy_for_vis1_per_track"] = occ_acc_for_vis1

    # Compute position metrics per track.
    distances = torch.norm(pred_tracks - gt_tracks, dim=-1)
    thresholds = torch.tensor(distance_thresholds, device=distances.device)
    for thresh in thresholds:
        within_threshold = distances < thresh
        correct_positions = (within_threshold & ~gt_occluded & evaluation_points).float().sum(dim=1)
        visible_points = (~gt_occluded & evaluation_points).float().sum(dim=1)
        assert visible_points.min() > 0, "No visible points to evaluate. Make sure at least two timesteps were visible."
        metrics[f"pts_within_{thresh:.2f}_per_track"] = correct_positions / visible_points

        true_positives = (within_threshold & ~pred_occluded & ~gt_occluded & evaluation_points).float().sum(dim=1)
        gt_positives = (~gt_occluded & evaluation_points).float().sum(dim=1)
        false_positives = (~within_threshold & ~pred_occluded) | (~pred_occluded & gt_occluded)
        false_positives = (false_positives & evaluation_points).float().sum(dim=1)
        jaccard = true_positives / (gt_positives + false_positives)
        metrics[f"jaccard_{thresh:.2f}_per_track"] = jaccard

    metrics["average_jaccard_per_track"] = torch.stack([metrics[f"jaccard_{thresh:.2f}_per_track"]
                                                        for thresh in thresholds], dim=-1).mean(dim=-1)
    metrics["average_pts_within_thresh_per_track"] = torch.stack([metrics[f"pts_within_{thresh:.2f}_per_track"]
                                                                  for thresh in thresholds], dim=-1).mean(dim=-1)

    # Assert no nans
    for k, v in metrics.items():
        if k in ["occlusion_accuracy_for_vis0_per_track", "occlusion_accuracy_for_vis1_per_track"]:
            continue  # They can have nans and will be handled later
        assert not torch.isnan(v).any(), f"NaN found in {k}"

    return metrics


def compute_tapvid_metrics_original(
        query_points: np.ndarray,
        gt_occluded: np.ndarray,
        gt_tracks: np.ndarray,
        pred_occluded: np.ndarray,
        pred_tracks: np.ndarray,
        query_mode: str,
) -> Mapping[str, np.ndarray]:
    """Computes TAP-Vid metrics (Jaccard, Pts. Within Thresh, Occ. Acc.)
    See the TAP-Vid paper for details on the metric computation.  All inputs are
    given in raster coordinates.  The first three arguments should be the direct
    outputs of the reader: the 'query_points', 'occluded', and 'target_points'.
    The paper metrics assume these are scaled relative to 256x256 images.
    pred_occluded and pred_tracks are your algorithm's predictions.
    This function takes a batch of inputs, and computes metrics separately for
    each video.  The metrics for the full benchmark are a simple mean of the
    metrics across the full set of videos.  These numbers are between 0 and 1,
    but the paper multiplies them by 100 to ease reading.
    Args:
       query_points: The query points, an in the format [t, y, x].  Its size is
         [b, n, 3], where b is the batch size and n is the number of queries
       gt_occluded: A boolean array of shape [b, n, t], where t is the number
         of frames.  True indicates that the point is occluded.
       gt_tracks: The target points, of shape [b, n, t, 2].  Each point is
         in the format [x, y]
       pred_occluded: A boolean array of predicted occlusions, in the same
         format as gt_occluded.
       pred_tracks: An array of track predictions from your algorithm, in the
         same format as gt_tracks.
       query_mode: Either 'first' or 'strided', depending on how queries are
         sampled.  If 'first', we assume the prior knowledge that all points
         before the query point are occluded, and these are removed from the
         evaluation.
    Returns:
        A dict with the following keys:
        occlusion_accuracy: Accuracy at predicting occlusion.
        pts_within_{x} for x in [1, 2, 4, 8, 16]: Fraction of points
          predicted to be within the given pixel threshold, ignoring occlusion
          prediction.
        jaccard_{x} for x in [1, 2, 4, 8, 16]: Jaccard metric for the given
          threshold
        average_pts_within_thresh: average across pts_within_{x}
        average_jaccard: average across jaccard_{x}
    """

    metrics = {}
    # Fixed bug is described in:
    # https://github.com/facebookresearch/co-tracker/issues/20
    eye = np.eye(gt_tracks.shape[2], dtype=np.int32)

    if query_mode == "first":
        # evaluate frames after the query frame
        query_frame_to_eval_frames = np.cumsum(eye, axis=1) - eye
    elif query_mode == "strided":
        # evaluate all frames except the query frame
        query_frame_to_eval_frames = 1 - eye
    else:
        raise ValueError("Unknown query mode " + query_mode)

    query_frame = query_points[..., 0]
    query_frame = np.round(query_frame).astype(np.int32)
    evaluation_points = query_frame_to_eval_frames[query_frame] > 0

    # Occlusion accuracy is simply how often the predicted occlusion equals the
    # ground truth.
    occ_acc = np.sum(
        np.equal(pred_occluded, gt_occluded) & evaluation_points,
        axis=(1, 2),
    ) / np.sum(evaluation_points)
    metrics["occlusion_accuracy"] = occ_acc

    # Next, convert the predictions and ground truth positions into pixel
    # coordinates.
    visible = np.logical_not(gt_occluded)
    pred_visible = np.logical_not(pred_occluded)
    all_frac_within = []
    all_jaccard = []
    for thresh in [1, 2, 4, 8, 16]:
        # True positives are points that are within the threshold and where both
        # the prediction and the ground truth are listed as visible.
        within_dist = np.sum(
            np.square(pred_tracks - gt_tracks),
            axis=-1,
        ) < np.square(thresh)
        is_correct = np.logical_and(within_dist, visible)

        # Compute the frac_within_threshold, which is the fraction of points
        # within the threshold among points that are visible in the ground truth,
        # ignoring whether they're predicted to be visible.
        count_correct = np.sum(
            is_correct & evaluation_points,
            axis=(1, 2),
        )
        count_visible_points = np.sum(visible & evaluation_points, axis=(1, 2))
        frac_correct = count_correct / count_visible_points
        metrics["pts_within_" + str(thresh)] = frac_correct
        all_frac_within.append(frac_correct)

        true_positives = np.sum(
            is_correct & pred_visible & evaluation_points, axis=(1, 2)
        )

        # The denominator of the jaccard metric is the true positives plus
        # false positives plus false negatives.  However, note that true positives
        # plus false negatives is simply the number of points in the ground truth
        # which is easier to compute than trying to compute all three quantities.
        # Thus we just add the number of points in the ground truth to the number
        # of false positives.
        #
        # False positives are simply points that are predicted to be visible,
        # but the ground truth is not visible or too far from the prediction.
        gt_positives = np.sum(visible & evaluation_points, axis=(1, 2))
        false_positives = (~visible) & pred_visible
        false_positives = false_positives | ((~within_dist) & pred_visible)
        false_positives = np.sum(false_positives & evaluation_points, axis=(1, 2))
        jaccard = true_positives / (gt_positives + false_positives)
        metrics["jaccard_" + str(thresh)] = jaccard
        all_jaccard.append(jaccard)
    metrics["average_jaccard"] = np.mean(
        np.stack(all_jaccard, axis=1),
        axis=1,
    )
    metrics["average_pts_within_thresh"] = np.mean(
        np.stack(all_frac_within, axis=1),
        axis=1,
    )
    return metrics


def evaluate_predictions(
        gt_tracks,
        gt_visibilities,
        pred_tracks,
        pred_occluded,
        query_points=None,
        distance_thresholds=[0.01, 0.02, 0.04, 0.08, 0.16],  # 1 cm, 2 cm, 4 cm, 8 cm, 16 cm
        survival_distance_threshold=0.5,  # 50 cm
        static_threshold=0.01,  # < 0.01 cm
        dynamic_threshold=0.1,  # > 10 cm
        very_dynamic_threshold=2.0,  # > 2 m
):
    n_frames, n_points, n_point_dim = gt_tracks.shape

    if query_points is None:
        warnings.warn("Query points are not provided. Using the first visible frame as query points.")
        query_points_t = np.argmax(gt_visibilities, axis=0)
        query_points_xyz = gt_tracks[query_points_t, np.arange(n_points)]
        query_points = np.concatenate([query_points_t[:, None], query_points_xyz], axis=-1)

    at_query_timestep_or_later = (np.arange(n_frames)[:, None] >= query_points[:, 0][None, :])
    gt_visibilities = gt_visibilities.copy() * at_query_timestep_or_later

    movement = np.zeros(n_points)
    for point_idx in range(n_points):
        point_track = gt_tracks[gt_visibilities[:, point_idx], point_idx, :]
        movement[point_idx] = np.linalg.norm(point_track[1:] - point_track[:-1], axis=-1).sum()

    point_types = ["any"]
    static_points = None
    dynamic_points = None
    very_dynamic_points = None
    if static_threshold is not None:
        point_types += ["static"]
        static_points = movement < static_threshold
    if dynamic_threshold is not None:
        point_types += ["dynamic"]
        dynamic_points = movement > dynamic_threshold
    if very_dynamic_threshold is not None:
        point_types += ["very_dynamic"]
        very_dynamic_points = movement > very_dynamic_threshold

    mask_1 = gt_visibilities.sum(axis=0) >= 2  # At least two visible, the first one is a query

    results = {}
    results_per_track = {}
    for short_name, mask_a in [
        ("all", mask_1),
    ]:
        for point_type in point_types:
            if point_type == "any":
                mask_b = np.ones_like(mask_a)
            elif point_type == "static":
                mask_b = static_points
            elif point_type == "dynamic":
                mask_b = dynamic_points
            elif point_type == "very_dynamic":
                mask_b = very_dynamic_points
            else:
                raise ValueError
            mask_ab = mask_a & mask_b
            short_name_ = f"{short_name}_{point_type}"

            if mask_ab.sum() == 0:
                logging.info(f"No points for {short_name_} (empty mask).")
                continue

            pred_tracks_ = pred_tracks[:, mask_ab, :][None]
            out_metrics_3d = compute_metrics(
                torch.from_numpy(query_points[mask_ab, :][None]).float(),
                torch.from_numpy(~gt_visibilities[:, mask_ab][None]),
                torch.from_numpy(gt_tracks[:, mask_ab, :][None]).float(),
                torch.from_numpy(pred_occluded[:, mask_ab][None]),
                torch.from_numpy(pred_tracks_).float(),
                distance_thresholds=distance_thresholds,
                survival_distance_threshold=survival_distance_threshold,
                query_mode="first",
            )
            results[short_name_] = {}
            for k, v in out_metrics_3d.items():
                assert "_per_track" in k
                results[short_name_][k.replace("_per_track", "")] = v.nanmean().item() * 100
            results[short_name_]["n"] = mask_ab.sum() / n_points * 100
            results[short_name_]["v"] = (gt_visibilities[:, mask_ab].sum() / mask_ab.sum() / n_frames) * 100

            results_per_track[short_name_] = {}
            for k, v in out_metrics_3d.items():
                assert v.ndim == 2 and v.shape[0] == 1
                v = v[0]
                results_per_track[short_name_][k] = v.cpu().numpy() * 100
            results_per_track[short_name_]["indices"] = np.where(mask_ab)[0]

    if "all_static" in results.keys() and "all_dynamic" in results.keys():
        results["all_dynamic-static-mean"] = {}
        for k in results["all_static"].keys():
            results["all_dynamic-static-mean"][k] = (results["all_dynamic"][k] + results["all_static"][k]) / 2

    df = pd.DataFrame(results)
    df = df.round(2)

    df_per_track = pd.DataFrame(results_per_track)
    df_per_track = df_per_track.round(2)


    return df, df_per_track
