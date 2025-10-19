import sys

import argparse
import os
import cv2
import glob
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict

from PIL import Image
from matplotlib import pyplot as plt
from pathlib import Path

DEVICE = 'cuda'

def load_image(imfile):
    img = np.array(Image.open(imfile).convert('RGB')).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img.to(DEVICE)

def demo(args):
    from models.stereoanyvideo_model import StereoAnyVideoModel
    from models.bidastereo_model import BiDAStereoModel
    from models.dynamic_stereo_model import DynamicStereoModel
    from models.ppm_stereo_model import PPMStereoModel
    
    if  args.model_name == 'bidastereo':
        model = BiDAStereoModel()
    elif args.model_name == 'dynamic':
        model = DynamicStereoModel()
    elif args.model_name == 'ppmstereo':
         model = PPMStereoModel()    
    elif args.model_name == 'stereoanyvideo':
        model = StereoAnyVideoModel()  

    if args.ckpt is not None:
        assert args.ckpt.endswith(".pth") or args.ckpt.endswith(
            ".pt"
        )
        strict = True
        state_dict = torch.load(args.ckpt)
        if "model" in state_dict:
            state_dict = state_dict["model"]
        if list(state_dict.keys())[0].startswith("module."):
            state_dict = {
                k.replace("module.", ""): v for k, v in state_dict.items()
            }
        model.model.load_state_dict(state_dict, strict=strict)
        print("Done loading model checkpoint", args.ckpt)

    model.to(DEVICE)
    model.eval()

    output_directory = args.output_path
    parent_directory = os.path.dirname(output_directory)
    if not os.path.exists(parent_directory):
        os.makedirs(parent_directory)
    if not os.path.isdir(output_directory):
        os.mkdir(output_directory)

    with torch.no_grad():
        images_left = sorted(glob.glob(os.path.join(args.path, 'left/*.png')) + glob.glob(os.path.join(args.path, 'left/*.jpg')))
        images_right = sorted(glob.glob(os.path.join(args.path, 'right/*.png')) + glob.glob(os.path.join(args.path, 'right/*.jpg')))
        assert len(images_left) == len(images_right), [len(images_left), len(images_right)]
        assert len(images_left) > 0, args.path
        print(f"Found {len(images_left)} frames. Saving files to {args.output_path}")

        num_frames = len(images_left)
        frame_size = args.frame_size

        disparities_ori_all = []

        for start_idx in range(0, num_frames, frame_size):
            end_idx = min(start_idx + frame_size, num_frames)

            image_left_list = []
            image_right_list = []

            for imfile1, imfile2 in zip(images_left[start_idx:end_idx], images_right[start_idx:end_idx]):
                image_left = load_image(imfile1)
                image_right = load_image(imfile2)
                image_left = F.interpolate(image_left[None], size=args.resize, mode="bilinear", align_corners=True)
                image_right = F.interpolate(image_right[None], size=args.resize, mode="bilinear", align_corners=True)
                image_left_list.append(image_left[0])
                image_right_list.append(image_right[0])

            video_left = torch.stack(image_left_list, dim=0)
            video_right = torch.stack(image_right_list, dim=0)

            batch_dict = defaultdict(list)
            batch_dict["stereo_video"] = torch.stack([video_left, video_right], dim=1)

            predictions = model(batch_dict)

            assert "disparity" in predictions
            disparities = predictions["disparity"][:, :1].clone().data.cpu().abs().numpy()
            disparities_ori = disparities.astype(np.uint8)
            disparities_ori_all.extend(disparities_ori)

        disparities_ori_all = np.array(disparities_ori_all)

        epsilon = 1e-5  # Smallest allowable disparity
        disparities_ori_all[disparities_ori_all < epsilon] = epsilon

        disparities_all = ((disparities_ori_all - disparities_ori_all.min()) / (disparities_ori_all.max() - disparities_ori_all.min()) * 255).astype(np.uint8)

        video_ori_disparity = cv2.VideoWriter(
            os.path.join(args.output_path, "disparity.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps=args.fps,
            frameSize=(disparities_all.shape[3], disparities_all.shape[2]),
            isColor=True,
        )
        video_disparity = cv2.VideoWriter(
            os.path.join(args.output_path, "disparity_norm.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps=args.fps,
            frameSize=(disparities_all.shape[3], disparities_all.shape[2]),
            isColor=True,
        )


        video_imgs = cv2.VideoWriter(
            os.path.join(args.output_path, "left_right.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps=args.fps,
            frameSize=(disparities_all.shape[3]*2, disparities_all.shape[2]),
            isColor=True,
        )

        for i in range(num_frames):
            imfile1 = images_left[i]
            imfile2 = images_right[i]

            left_img = cv2.imread(imfile1)
            right_img = cv2.imread(imfile2)
    
            # 水平拼接两帧
            combined_frame = cv2.hconcat([left_img, right_img])
            video_imgs.write(combined_frame)
            
            disparity_norm = disparities_all[i]
            disparity_norm = disparity_norm.transpose(1, 2, 0)
            disparity_norm_vis = cv2.applyColorMap(disparity_norm, cv2.COLORMAP_INFERNO)
            video_disparity.write(disparity_norm_vis)

            disparity_ori = disparities_ori_all[i]
            disparity_ori = disparity_ori.transpose(1, 2, 0)
            disparity_ori_vis = cv2.applyColorMap(disparity_ori, cv2.COLORMAP_INFERNO)
            video_ori_disparity.write(disparity_ori_vis)

            if args.save_png:
                filename_temp = args.output_path + '/disparity_norm_' + str(i).zfill(3) + '.png'
                cv2.imwrite(filename_temp, disparity_norm_vis)
                filename_temp = args.output_path + '/disparity_ori_' + str(i).zfill(3) + '.png'
                cv2.imwrite(filename_temp, disparity_ori_vis)

        video_ori_disparity.release()
        video_disparity.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default="bidastereo", choices=["dynamic","bidastereo","ppmstereo","stereoanyvideo"],help="name to specify model")
    parser.add_argument('--ckpt', default="/home/ywang/my_projects/MemStereo/ckpt/bidastereo_sf_dr.pth", help="checkpoint of stereo model")
    parser.add_argument('--resize', default=(720, 1280), help="image size input to the model")
    parser.add_argument("--fps", type=int, default=30, help="frame rate for video visualization")
    parser.add_argument('--path', default="/home/ywang/dataset/SouthKensington/indoor/video030/images/", help="dataset for evaluation")
    parser.add_argument("--save_png", action="store_true")
    parser.add_argument("--frame_size", type=int, default=150, help="number of updates in each forward pass.")
    parser.add_argument("--iters",type=int, default=20, help="number of updates in each forward pass.")
    parser.add_argument("--kernel_size", type=int, default=20, help="number of frames in each forward pass.")
    parser.add_argument('--output_path', help="directory to save output", default="/home/ywang/my_projects/MemStereo/demo_ouput/")

    # Validation parameters
    parser.add_argument(
        "--valid_iters",
        type=int,
        default=20,
        help="number of updates to the disparity field in each forward pass during validation.",
    )
    # Architecure choices
    parser.add_argument(
        "--different_update_blocks",
        action="store_false",
        help="use different update blocks for each resolution",
    )
    parser.add_argument(
        "--attention_type",
        type=str,
        default="self_stereo_temporal_update_time_update_space",
        help="attention type of the SST and update blocks. \
            Any combination of 'self_stereo', 'temporal', 'update_time', 'update_space' connected by an underscore.",
    )
    parser.add_argument(
        "--update_block_3d", action="store_false", 
        help="use Conv3D update block"
    )
    parser.add_argument(
        "--use_convex_3d", action="store_false", 
        help="different upsampling strategy"
    )
    
    args = parser.parse_args()

    demo(args)