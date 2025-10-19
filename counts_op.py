# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
os.environ["CUDA_VISIBLE_DEVICES"] ="2"
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import thop

torch.set_float32_matmul_precision("high")

import argparse
import logging
from pathlib import Path
from tqdm import tqdm
from thop import profile, clever_format
from collections import defaultdict
    
from pytorch_lightning.lite import LightningLite
from torch.cuda.amp import GradScaler

from train_utils.utils import (
    run_test_eval,
    save_ims_to_tb,
    count_parameters,
)
from train_utils.logger import Logger
from models.core.dynamic_stereo import DynamicStereo
from PPMStereo.models.core.ppmstereo import MemStereo
from PPMStereo.models.core.ppmstereo_VDA import MemStereo_vpt
from models.core.bidastereo import BiDAStereo

from evaluation.core.evaluator import Evaluator
from train_utils.losses import sequence_loss
import datasets.dynamic_stereo_datasets as datasets


class Lite(LightningLite):
    def run(self, args):
        self.seed_everything(0)
   
        if  args.name == 'bidastereo':
            model = BiDAStereo(
                mixed_precision=args.mixed_precision,
            )

        elif args.name == 'dynamic':
            model = DynamicStereo(
            max_disp=256,
            mixed_precision=args.mixed_precision,
            num_frames=args.sample_len,
            attention_type=args.attention_type,
            use_3d_update_block=args.update_block_3d,
            different_update_blocks=args.different_update_blocks,
        )

        elif args.name == 'memstereo':
            model = MemStereo(
            max_disp=256,
            mixed_precision=args.mixed_precision,
            num_frames=args.sample_len,
            attention_type=args.attention_type,
            use_3d_update_block=args.update_block_3d,
            use_convex_3d=args.use_convex_3d,
            different_update_blocks=args.different_update_blocks,
        )

        elif args.name == 'memstereo_vpt':
            model = MemStereo_vpt(
            max_disp=256,
            mixed_precision=args.mixed_precision,
            num_frames=args.sample_len,
            attention_type=args.attention_type,
            use_3d_update_block=args.update_block_3d,
            different_update_blocks=args.different_update_blocks,
        )
            
        logging.info(f"Parameter Count: {count_parameters(model)}")
        
        model.cuda()
        model.eval()
        model.freeze_bn()  # We keep BatchNorm frozen

        img = torch.randn(1, 20, 3, 768, 1024).cuda()
        predictions = defaultdict(list)
        predictions["stereo_video"] = img
        macs, params = profile(model, inputs=(img, img))
        macs, params = clever_format([macs, params], "%.3f")  # to be consistent with neapeak
        
        print("Input size:", img.size())
        print("Macs:", macs)
        print("params:", params)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="dynamic", choices=["bidastereo", "memstereo", "memstereo_vpt", "dynamic"])
    parser.add_argument("--ckpt_model", default="/home/ywang/my_projects/MemStereo/exp/mem/model_memstereo_119490.pth", type=str, help="restore checkpoint")
    parser.add_argument("--ckpt_path", default='/home/ywang/my_projects/MemStereo/exp/mem_mlp/', help="path to save checkpoints")
    parser.add_argument(
        "--mixed_precision", action="store_true", help="use mixed precision"
    )

    # Training parameters
    parser.add_argument(
        "--batch_size", type=int, default=2, help="batch size used during training."
    )
    parser.add_argument(
        "--train_datasets",
        nargs="+",
        # default=["dynamic_replica"],
        default=["things", "monkaa", "driving"],
        help="training datasets.",
    )
    parser.add_argument("--lr", type=float, default=0.0003, help="max learning rate.")

    parser.add_argument(
        "--num_steps", type=int, default=200000, help="length of training schedule."
    )
    parser.add_argument(
        "--image_size",
        type=int,
        nargs="+",
        default=[256, 448],
        help="size of the random image crops used during training.",
    )
    parser.add_argument(
        "--train_iters",
        type=int,
        default=10,
        help="number of updates to the disparity field in each forward pass.",
    )
    parser.add_argument(
        "--wdecay", type=float, default=0.00001, help="Weight decay in optimizer."
    )
    parser.add_argument(
        "--sample_len", type=int, default=5, help="length of training video samples"
    )
    parser.add_argument(
        "--validate_at_start", action="store_true", help="validate the model at start"
    )
    parser.add_argument("--save_freq", type=int, default=1000, help="save frequency")

    parser.add_argument(
        "--evaluate_every_n_epoch",
        type=int,
        default=1,
        help="evaluate every n epoch",
    )

    parser.add_argument(
        "--num_workers", type=int, default=16, help="number of dataloader workers."
    )
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
        "--use_convex_3d", action="store_true", 
        help="different upsampling strategy"
    )
    
    parser.add_argument(
        "--init_flow", action="store_true", 
        help="precit the 0-th disparity"
    )
    
    # Data augmentation
    parser.add_argument(
        "--img_gamma", type=float, nargs="+", default=None, help="gamma range"
    )
    parser.add_argument(
        "--saturation_range",
        type=float,
        nargs="+",
        default=[0, 1.4],
        help="color saturation",
    )
    parser.add_argument(
        "--do_flip",
        default=False,
        choices=["h", "v"],
        help="flip the images horizontally or vertically",
    )
    parser.add_argument(
        "--spatial_scale",
        type=float,
        nargs="+",
        default=[0.2, 0.4],
        help="re-scale the images randomly",
    )
    parser.add_argument(
        "--noyjitter",
        action="store_true",
        help="don't simulate imperfect rectification",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    )

    Path(args.ckpt_path).mkdir(exist_ok=True, parents=True)
    from pytorch_lightning.strategies import DDPStrategy

    Lite(
        strategy=DDPStrategy(find_unused_parameters=True),
        devices="auto",
        accelerator="gpu",
        precision=32,
    ).run(args)
