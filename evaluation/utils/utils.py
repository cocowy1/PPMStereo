# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
import configparser
import os
import math
from typing import Optional, List
import torch
import cv2
import numpy as np
from dataclasses import dataclass
from tabulate import tabulate
import matplotlib.pyplot as plt

from pytorch3d.structures import Pointclouds
from pytorch3d.transforms import RotateAxisAngle
from pytorch3d.utils import (
    opencv_from_cameras_projection,
)
from pytorch3d.renderer import (
    AlphaCompositor,
    PointsRasterizationSettings,
    PointsRasterizer,
    PointsRenderer,
)
from evaluation.utils.eval_utils import depth_to_pcd

# Copyright (2025) Bytedance Ltd. and/or its affiliates 

# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License. 
# You may obtain a copy of the License at 

#     http://www.apache.org/licenses/LICENSE-2.0 

# Unless required by applicable law or agreed to in writing, software 
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
# See the License for the specific language governing permissions and 
# limitations under the License. 

def compute_scale_and_shift(prediction, target, mask, scale_only=False):
    if scale_only:
        return compute_scale(prediction, target, mask), 0
    else:
        return compute_scale_and_shift_full(prediction, target, mask)


def compute_scale(prediction, target, mask):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    prediction = prediction.astype(np.float32)
    target = target.astype(np.float32)
    mask = mask.astype(np.float32)

    a_00 = np.sum(mask * prediction * prediction)
    a_01 = np.sum(mask * prediction)
    a_11 = np.sum(mask)

    # right hand side: b = [b_0, b_1]
    b_0 = np.sum(mask * prediction * target)

    x_0 = b_0 / (a_00 + 1e-6)

    return x_0

def compute_scale_and_shift_full(prediction, target, mask):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    prediction = prediction.astype(np.float32)
    target = target.astype(np.float32)
    mask = mask.astype(np.float32)

    a_00 = np.sum(mask * prediction * prediction)
    a_01 = np.sum(mask * prediction)
    a_11 = np.sum(mask)

    b_0 = np.sum(mask * prediction * target)
    b_1 = np.sum(mask * target)

    x_0 = 1
    x_1 = 0

    det = a_00 * a_11 - a_01 * a_01

    if det != 0:
        x_0 = (a_11 * b_0 - a_01 * b_1) / det
        x_1 = (-a_01 * b_0 + a_00 * b_1) / det

    return x_0, x_1


def get_interpolate_frames(frame_list_pre, frame_list_post):
    assert len(frame_list_pre) == len(frame_list_post)
    min_w = 0.0
    max_w = 1.0
    step = (max_w - min_w) / (len(frame_list_pre)-1)
    post_w_list = [min_w] + [i * step for i in range(1,len(frame_list_pre)-1)] + [max_w]
    interpolated_frames = []
    for i in range(len(frame_list_pre)):
        interpolated_frames.append(frame_list_pre[i] * (1-post_w_list[i]) + frame_list_post[i] * post_w_list[i])
    return interpolated_frames


@dataclass
class PerceptionPrediction:
    """
    Holds the tensors that describe a result of any perception module.
    """

    depth_map: Optional[torch.Tensor] = None
    disparity: Optional[torch.Tensor] = None
    image_rgb: Optional[torch.Tensor] = None
    fg_probability: Optional[torch.Tensor] = None


def aggregate_eval_results(per_batch_eval_results, reduction="mean"):

    total_length = 0
    aggregate_results = defaultdict(list)
    for result in per_batch_eval_results:
        if isinstance(result, tuple):
            reduction = "sum"
            length = result[1]
            total_length += length
            result = result[0]
        for metric, val in result.items():
            if reduction == "sum":
                aggregate_results[metric].append(val * length)

    if reduction == "mean":
        return {k: torch.cat(v).mean().item() for k, v in aggregate_results.items()}
    elif reduction == "sum":
        return {
            k: torch.cat(v).sum().item() / float(total_length)
            for k, v in aggregate_results.items()
        }


def aggregate_and_print_results(
    per_batch_eval_results: List[dict],
):
    print("")
    result = aggregate_eval_results(
        per_batch_eval_results,
    )
    pretty_print_perception_metrics(result)
    result = {str(k): v for k, v in result.items()}

    print("")
    return result


def pretty_print_perception_metrics(results):

    metrics = sorted(list(results.keys()), key=lambda x: x.metric)

    print("===== Perception results =====")
    print(
        tabulate(
            [[metric, results[metric]] for metric in metrics],
        )
    )


def read_calibration(calibration_file, resolution_string):
    # ported from https://github.com/stereolabs/zed-open-capture/
    # blob/dfa0aee51ccd2297782230a05ca59e697df496b2/examples/include/calibration.hpp#L4172

    zed_resolutions = {
        "2K": (1242, 2208),
        "FHD": (1080, 1920),
        "HD": (720, 1280),
        # "qHD": (540, 960),
        "VGA": (376, 672),
    }
    assert resolution_string in zed_resolutions.keys()
    image_height, image_width = zed_resolutions[resolution_string]

    # Open camera configuration file
    assert os.path.isfile(calibration_file)
    calib = configparser.ConfigParser()
    calib.read(calibration_file)

    # Get translations
    T = np.zeros((3, 1))
    T[0] = float(calib["STEREO"]["baseline"])
    T[1] = float(calib["STEREO"]["ty"])
    T[2] = float(calib["STEREO"]["tz"])

    baseline = T[0]

    # Get left parameters
    left_cam_cx = float(calib[f"LEFT_CAM_{resolution_string}"]["cx"])
    left_cam_cy = float(calib[f"LEFT_CAM_{resolution_string}"]["cy"])
    left_cam_fx = float(calib[f"LEFT_CAM_{resolution_string}"]["fx"])
    left_cam_fy = float(calib[f"LEFT_CAM_{resolution_string}"]["fy"])
    left_cam_k1 = float(calib[f"LEFT_CAM_{resolution_string}"]["k1"])
    left_cam_k2 = float(calib[f"LEFT_CAM_{resolution_string}"]["k2"])
    left_cam_p1 = float(calib[f"LEFT_CAM_{resolution_string}"]["p1"])
    left_cam_p2 = float(calib[f"LEFT_CAM_{resolution_string}"]["p2"])
    left_cam_k3 = float(calib[f"LEFT_CAM_{resolution_string}"]["k3"])

    # Get right parameters
    right_cam_cx = float(calib[f"RIGHT_CAM_{resolution_string}"]["cx"])
    right_cam_cy = float(calib[f"RIGHT_CAM_{resolution_string}"]["cy"])
    right_cam_fx = float(calib[f"RIGHT_CAM_{resolution_string}"]["fx"])
    right_cam_fy = float(calib[f"RIGHT_CAM_{resolution_string}"]["fy"])
    right_cam_k1 = float(calib[f"RIGHT_CAM_{resolution_string}"]["k1"])
    right_cam_k2 = float(calib[f"RIGHT_CAM_{resolution_string}"]["k2"])
    right_cam_p1 = float(calib[f"RIGHT_CAM_{resolution_string}"]["p1"])
    right_cam_p2 = float(calib[f"RIGHT_CAM_{resolution_string}"]["p2"])
    right_cam_k3 = float(calib[f"RIGHT_CAM_{resolution_string}"]["k3"])

    # Get rotations
    R_zed = np.zeros(3)
    R_zed[0] = float(calib["STEREO"][f"rx_{resolution_string.lower()}"])
    R_zed[1] = float(calib["STEREO"][f"cv_{resolution_string.lower()}"])
    R_zed[2] = float(calib["STEREO"][f"rz_{resolution_string.lower()}"])

    R = cv2.Rodrigues(R_zed)[0]

    # Left
    cameraMatrix_left = np.array(
        [[left_cam_fx, 0, left_cam_cx], [0, left_cam_fy, left_cam_cy], [0, 0, 1]]
    )
    distCoeffs_left = np.array(
        [left_cam_k1, left_cam_k2, left_cam_p1, left_cam_p2, left_cam_k3]
    )

    # Right
    cameraMatrix_right = np.array(
        [
            [right_cam_fx, 0, right_cam_cx],
            [0, right_cam_fy, right_cam_cy],
            [0, 0, 1],
        ]
    )
    distCoeffs_right = np.array(
        [right_cam_k1, right_cam_k2, right_cam_p1, right_cam_p2, right_cam_k3]
    )

    # Stereo
    R1, R2, P1, P2, Q = cv2.stereoRectify(
        cameraMatrix1=cameraMatrix_left,
        distCoeffs1=distCoeffs_left,
        cameraMatrix2=cameraMatrix_right,
        distCoeffs2=distCoeffs_right,
        imageSize=(image_width, image_height),
        R=R,
        T=T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        newImageSize=(image_width, image_height),
        alpha=0,
    )[:5]

    # Precompute maps for cv::remap()
    map_left_x, map_left_y = cv2.initUndistortRectifyMap(
        cameraMatrix_left,
        distCoeffs_left,
        R1,
        P1,
        (image_width, image_height),
        cv2.CV_32FC1,
    )
    map_right_x, map_right_y = cv2.initUndistortRectifyMap(
        cameraMatrix_right,
        distCoeffs_right,
        R2,
        P2,
        (image_width, image_height),
        cv2.CV_32FC1,
    )

    zed_calib = {
        "map_left_x": map_left_x,
        "map_left_y": map_left_y,
        "map_right_x": map_right_x,
        "map_right_y": map_right_y,
        "pose_left": P1,
        "pose_right": P2,
        "baseline": baseline,
        "image_width": image_width,
        "image_height": image_height,
    }

    return zed_calib


def visualize_batch(
    batch_dict: dict,
    preds: PerceptionPrediction,
    output_dir: str,
    ref_frame: int = 0,
    only_foreground=False,
    step=0,
    sequence_name=None,
    writer=None,
):
    os.makedirs(output_dir, exist_ok=True)

    outputs = {}

    if preds.depth_map is not None:
        device = preds.depth_map.device

        pcd_global_seq = []
        H, W = batch_dict["stereo_video"].shape[3:]

        for i in range(len(batch_dict["stereo_video"])):
            R, T, K = opencv_from_cameras_projection(
                preds.perspective_cameras[i],
                torch.tensor([H, W])[None].to(device),
            )

            extrinsic_3x4_0 = torch.cat([R[0], T[0, :, None]], dim=1)

            extr_matrix = torch.cat(
                [
                    extrinsic_3x4_0,
                    torch.Tensor([[0, 0, 0, 1]]).to(extrinsic_3x4_0.device),
                ],
                dim=0,
            )

            inv_extr_matrix = extr_matrix.inverse().to(device)
            pcd, colors = depth_to_pcd(
                preds.depth_map[i, 0],
                batch_dict["stereo_video"][i][0].permute(1, 2, 0),
                K[0][0][0],
                K[0][0][2],
                K[0][1][2],
                step=1,
                inv_extrinsic=inv_extr_matrix,
                mask=batch_dict["fg_mask"][i, 0] if only_foreground else None,
                filter=False,
            )

            R, T = inv_extr_matrix[None, :3, :3], inv_extr_matrix[None, :3, 3]
            pcd_global_seq.append((pcd, colors, (R, T, preds.perspective_cameras[i])))

        raster_settings = PointsRasterizationSettings(
            image_size=[H, W], radius=0.003, points_per_pixel=10
        )
        R, T, cam_ = pcd_global_seq[ref_frame][2]

        median_depth = preds.depth_map.median()
        cam_.cuda()

        for mode in ["angle_15", "angle_-15", "changing_angle"]:
            res = []

            for t, (pcd, color, __) in enumerate(pcd_global_seq):

                if mode == "changing_angle":
                    angle = math.cos((math.pi) * (t / 15)) * 15
                elif mode == "angle_15":
                    angle = 15
                elif mode == "angle_-15":
                    angle = -15

                delta_x = median_depth * math.sin(math.radians(angle))
                delta_z = median_depth * (1 - math.cos(math.radians(angle)))

                cam = cam_.clone()
                cam.R = torch.bmm(
                    cam.R,
                    RotateAxisAngle(angle=angle, axis="Y", device=device).get_matrix()[
                        :, :3, :3
                    ],
                )
                cam.T[0, 0] = cam.T[0, 0] - delta_x
                cam.T[0, 2] = cam.T[0, 2] - delta_z + median_depth / 2.0

                rasterizer = PointsRasterizer(
                    cameras=cam, raster_settings=raster_settings
                )
                renderer = PointsRenderer(
                    rasterizer=rasterizer,
                    compositor=AlphaCompositor(background_color=(1, 1, 1)),
                )
                pcd_copy = pcd.clone()

                point_cloud = Pointclouds(points=[pcd_copy], features=[color / 255.0])
                images = renderer(point_cloud)
                res.append(images[0, ..., :3].cpu())
            res = torch.stack(res)

            ######################## mean and var image visualization ##########################
            mean = res.mean(dim=0)
            var = res.var(dim=0)
            h, w , _ = mean.shape
            red_mask = torch.zeros((h, w, 3), dtype=torch.uint8)

            threshold = 40
            # set red mask
            red_mask[:, :, 0] = 1 
            red_mask[:, :, 1] = 0   
            red_mask[:, :, 2] = 0   
            red_mask = red_mask.to(device)
            
            var_mask = (var * 255) > threshold
            
            mean_img_filter = mean.cpu() * (~var_mask).cpu().float() + red_mask.cpu() * var_mask.cpu().float()
            ##################################################################################
            
            video = (res * 255).numpy().astype(np.uint8)
            save_name = f"{sequence_name}_reconstruction_{step}_mode_{mode}_"
            
            if writer is None:
                outputs[mode] = video
            if only_foreground:
                save_name += "fg_only"
            else:
                save_name += "full_scene"
            
            video_out = cv2.VideoWriter(
                os.path.join(
                    output_dir,
                    f"{save_name}.mp4",
                ),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps=10,
                frameSize=(res.shape[2], res.shape[1]),
                isColor=True,
            )

            for i in range(len(video)):
                video_out.write(cv2.cvtColor(video[i], cv2.COLOR_BGR2RGB))
            video_out.release()

            if writer is not None:
                writer.add_video(
                    f"{sequence_name}_reconstruction_mode_{mode}",
                    (res * 255).permute(0, 3, 1, 2).to(torch.uint8)[None],
                    global_step=step,
                    fps=8,
                )

    return outputs
