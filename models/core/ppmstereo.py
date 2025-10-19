# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List
from einops import rearrange
import torch
import torch.amp
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import math
import unfoldNd

from models.core.ppmtereo_update import (
    Attention,
    Attention_qk,
    SequenceUpdateBlock,
    SequenceUpdateBlock3D,
    TimeAttnBlock,
    get_temporal_positional_encoding,
)
from models.core.extractor import BasicEncoder, BasicEncoder_VFM, ResNetFPN, MultiLevelResNetFPN, conv3x3
from models.core.corr import CorrBlock1D, TFCL
from models.core.convnext import Feature

from models.core.attention import (
    PositionEncodingSine,
    LocalFeatureTransformer,
)
from models.core.utils.utils import InputPadder, interp

autocast = torch.cuda.amp.autocast


try:
    from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
except:
    print('no flash attention installed')
    
    
class PPMStereo(nn.Module):
    def __init__(
        self,
        max_disp: int = 192,
        mixed_precision: bool = False,
        num_frames: int = 5,
        attention_type: str = None,
        use_3d_update_block: bool = False,
        different_update_blocks: bool = False,
        use_convex_3d: bool = False,
        init_flow: bool = False,
    ):
        super(PPMStereo, self).__init__()

        self.max_flow = max_disp
        self.mixed_precision = mixed_precision
        self.use_cnet = True
        self.use_convex_3d = use_convex_3d
         
        self.hidden_dim = 128
        self.context_dim = 128
        dim = 256
        self.dim = dim
        self.dropout = 0
        self.use_3d_update_block = use_3d_update_block
        
        self.fnet = BasicEncoder(
        output_dim=dim, norm_fn="instance", dropout=self.dropout
        )
            
        if self.use_cnet == True:
            self.cnet = Feature(model_name="tiny", output_dim=self.dim)
        
        self.init_flow = init_flow
        
        hidden_dims = [128]*3
        context_dims = hidden_dims

        self.att = nn.ModuleList(
            [Attention_qk(num_heads=1, dim_head=self.context_dim),
             Attention_qk(num_heads=1,dim_head=self.context_dim),
             Attention_qk(num_heads=1, dim_head=self.context_dim)])
        
        # self.fusion_network = nn.Sequential(nn.Conv2d(dim, dim, 3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(dim, dim//2, 1, padding=0),
        # )
        
        # self.cnet = MultiBasicEncoder(output_dim=[hidden_dims, context_dims], norm_fn="instance", downsample=2)
        self.different_update_blocks = different_update_blocks
        cor_planes = 4 * 9
        self.depth = 4
        self.attention_type = attention_type
        # attention_type is a combination of the following attention types:
        # self_stereo, temporal, update_time, update_space
        # for example, self_stereo_temporal_update_time_update_space

        if self.use_3d_update_block:
            if self.different_update_blocks:
                self.update_block08 = SequenceUpdateBlock3D(
                    hidden_dim=self.hidden_dim, cor_planes=cor_planes, mask_size=4,
                    use_convex_3d=self.use_convex_3d,
                )
                self.update_block16 = SequenceUpdateBlock3D(
                    hidden_dim=self.hidden_dim,
                    cor_planes=cor_planes,
                    mask_size=4,
                    use_convex_3d=self.use_convex_3d,
                    attention_type=attention_type,
                )
                self.update_block04 = SequenceUpdateBlock3D(
                    hidden_dim=self.hidden_dim, cor_planes=cor_planes, mask_size=4,
                    use_convex_3d=self.use_convex_3d,
                )
            else:
                self.update_block = SequenceUpdateBlock3D(
                    hidden_dim=self.hidden_dim, cor_planes=cor_planes, mask_size=4
                )
        else:
            if self.different_update_blocks:
                self.update_block16 = SequenceUpdateBlock(
                    hidden_dim=self.hidden_dim,
                    cor_planes=cor_planes,
                    mask_size=4,
                )
                self.update_block08 = SequenceUpdateBlock(
                    hidden_dim=self.hidden_dim, cor_planes=cor_planes, mask_size=4
                )
                self.update_block04 = SequenceUpdateBlock(
                    hidden_dim=self.hidden_dim, cor_planes=cor_planes, mask_size=4
                )
            else:
                self.update_block = SequenceUpdateBlock(
                    hidden_dim=self.hidden_dim, cor_planes=cor_planes, mask_size=4
                )

        if attention_type is not None:
            if ("update_time" in attention_type) or ("temporal" in attention_type):
                self.time_embed = nn.Parameter(torch.zeros(1, num_frames, dim))
            if "temporal" in attention_type:
                self.time_attn_blocks = nn.ModuleList(
                    [TimeAttnBlock(dim=dim, num_heads=8) for _ in range(self.depth)]
                )

            if "self_stereo" in attention_type:
                self.self_attn_blocks = nn.ModuleList(
                    [
                        LocalFeatureTransformer(
                            d_model=dim,
                            nhead=8,
                            layer_names=["self"] * 1,
                            attention="linear",
                        )
                        for _ in range(self.depth)
                    ]
                )

                self.cross_attn_blocks = nn.ModuleList(
                    [
                        LocalFeatureTransformer(
                            d_model=dim,
                            nhead=8,
                            layer_names=["cross"] * 1,
                            attention="linear",
                        )
                        for _ in range(self.depth)
                    ]
                )

        self.num_frames = num_frames

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"time_embed"}

    def freeze_bn(self):
        for m in self.modules():
            
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def convex_upsample(self, flow: torch.Tensor, mask: torch.Tensor, rate: int = 4):
        """Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination"""
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, rate, rate, H, W).contiguous()
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(rate * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3).contiguous()
        
        return up_flow.reshape(N, 2, rate * H, rate * W)

    def convex_upsample_3d(self, flow, mask, rate, T):
        """Upsample flow field from [T, H/rate, W/rate, 2] to [T, H, W, 2] using convex combination.

        unfoldNd repo: https://github.com/f-dangel/unfoldNd
        Run: pip install --user unfoldNd

        Args:
            flow: (N*T, C_flow, H, W)
            mask: (N*T, C_mask, H, W) or (N, 1, 27, 1, rate, rate, T, H, W)
            rate: int
        """
        b = flow.shape[0] // T
        flow = rearrange(flow, "(b t) c h w -> b c t h w", b=b, t=T)
        mask = rearrange(mask, "(b t) c h w -> b c t h w", b=b, t=T)

        N, _, T, H, W = flow.shape

        mask = mask.view(N, 1, 27, 1, rate, rate, T, H, W)  # (N, 1, 27, rate, rate, rate, T, H, W) if upsample T
        mask = torch.softmax(mask, dim=2)

        upsample = unfoldNd.UnfoldNd([3, 3, 3], padding=1)
        flow_upsampled = upsample(rate * flow)
        flow_upsampled = flow_upsampled.view(N, 2, 27, 1, 1, 1, T, H, W)
        flow_upsampled = torch.sum(mask * flow_upsampled, dim=2)
        flow_upsampled = flow_upsampled.permute(0, 1, 5, 2, 6, 3, 7, 4)
        flow_upsampled = flow_upsampled.reshape(N, 2, T, rate * H,
                                                rate * W)  # (N, 2, rate*T, rate*H, rate*W) if upsample T
        up_flow = rearrange(flow_upsampled, "b c t h w -> (b t) c h w")

        return up_flow
    
    
    def zero_init(self, fmap: torch.Tensor):
        N, _, H, W = fmap.shape
        _x = torch.zeros([N, 1, H, W], dtype=torch.float32)
        _y = torch.zeros([N, 1, H, W], dtype=torch.float32)
        zero_flow = torch.cat((_x, _y), dim=1).to(fmap.device)
        return zero_flow

    def forward_batch_test(
        self, batch_dict: Dict, kernel_size: int = 20, iters: int = 20
    ):
        ## for visualization, stride can equal to kernel_size
        stride = kernel_size // 2
        predictions = defaultdict(list)

        uncertainties = []
        disp_preds = []
        video = batch_dict["stereo_video"]
        num_ims = len(video)
        print("iters", iters)
        # print("kernel_size", kernel_size)
        
        if kernel_size > num_ims:
            left_ims = video[:, 0]
            right_ims = video[:, 1]
            padder = InputPadder(left_ims.shape, divis_by=32)
            left_ims, right_ims = padder.pad(left_ims, right_ims)

            with autocast(enabled=self.mixed_precision, dtype=torch.bfloat16):
                # disparities_forw; [b, t, c, h, w]
                disparities_forw, uncertainty = self.forward(
                    left_ims[None].cuda(),
                    right_ims[None].cuda(),
                    iters=iters,
                    test_mode=True,
                )

            disparities_forw = padder.unpad(disparities_forw[0])[:, None].cpu()
            uncertainty = padder.unpad(uncertainty[0])[:, None].cpu()
            
            predictions["disparity"] = (disparities_forw).squeeze(1).abs()[:, :1]
            predictions["uncertainties"] = (uncertainty).squeeze(1).abs()[:, :1]
            
            return predictions
        
        
        
        for i in range(0, num_ims, stride):
            left_ims = video[i : min(i + kernel_size, num_ims), 0]
            padder = InputPadder(left_ims.shape, divis_by=32)

            right_ims = video[i : min(i + kernel_size, num_ims), 1]
            left_ims, right_ims = padder.pad(left_ims, right_ims)

            with autocast(enabled=self.mixed_precision, dtype=torch.bfloat16):
                # disparities_forw; [b, t, c, h, w]
                disparities_forw, uncertainty = self.forward(
                    left_ims[None].cuda(),
                    right_ims[None].cuda(),
                    iters=iters,
                    test_mode=True,
                )

            disparities_forw = padder.unpad(disparities_forw[0])[:, None].cpu()
            uncertainty = padder.unpad(uncertainty[0])[:, None].cpu()
            
            if len(disp_preds) > 0 and len(disparities_forw) >= stride:

                if len(disparities_forw) < kernel_size:
                    disp_preds.append(disparities_forw[stride // 2 :])
                    uncertainties.append(uncertainty[stride // 2 :])
                else:
                    disp_preds.append(disparities_forw[stride // 2 : -stride // 2])
                    uncertainties.append(uncertainty[stride // 2 : -stride // 2])
                    
            elif len(disp_preds) == 0:
                disp_preds.append(disparities_forw[: -stride // 2])
                uncertainties.append(uncertainty[: -stride // 2])
                
        predictions["disparity"] = (torch.cat(disp_preds).squeeze(1).abs())[:, :1]
        predictions["uncertainties"] = (torch.cat(uncertainties).squeeze(1).abs())[:, :1]
        
        # print(predictions["disparity"].shape)

        # ############ add visualization ##############
        # import matplotlib.pyplot as plt
        # for idx in range(len(predictions["disparity"])):
        #     plt.imsave(f'/home/ywang/my_projects/MemStereo/vis/sintel_clean_mem/disp_{idx}.svg', predictions["disparity"][idx].squeeze().cpu().numpy(), vmin=2, vmax=83, cmap='magma')
        # ############ add visualization ##############
            
        return predictions

    def forward_sst_block(
        self, fmap1_dw16: torch.Tensor, fmap2_dw16: torch.Tensor, T: int
    ):
        *_, h, w = fmap1_dw16.shape

        # positional encoding and self-attention
        pos_encoding_fn_small = PositionEncodingSine(d_model=self.dim, max_shape=(h, w))
        # 'n c h w -> n (h w) c'
        fmap1_dw16 = pos_encoding_fn_small(fmap1_dw16)
        # 'n c h w -> n (h w) c'
        fmap2_dw16 = pos_encoding_fn_small(fmap2_dw16)

        if self.attention_type is not None:
            # add time embeddings
            if (
                "temporal" in self.attention_type
                or "update_time" in self.attention_type
            ):
                fmap1_dw16 = rearrange(
                    fmap1_dw16, "(b t) m h w -> (b h w) t m", t=T, h=h, w=w
                )
                fmap2_dw16 = rearrange(
                    fmap2_dw16, "(b t) m h w -> (b h w) t m", t=T, h=h, w=w
                )

                # interpolate if video length doesn't match
                if T != self.num_frames:
                    time_embed = self.time_embed.transpose(1, 2)
                    new_time_embed = F.interpolate(time_embed, size=(T), mode="nearest")
                    new_time_embed = new_time_embed.transpose(1, 2).contiguous()
                else:
                    new_time_embed = self.time_embed

                fmap1_dw16 = fmap1_dw16 + new_time_embed
                fmap2_dw16 = fmap2_dw16 + new_time_embed

                fmap1_dw16 = rearrange(
                    fmap1_dw16, "(b h w) t m -> (b t) m h w", t=T, h=h, w=w
                )
                fmap2_dw16 = rearrange(
                    fmap2_dw16, "(b h w) t m -> (b t) m h w", t=T, h=h, w=w
                )

            if ("self_stereo" in self.attention_type) or (
                "temporal" in self.attention_type
            ):
                for att_ind in range(self.depth):
                    if "self_stereo" in self.attention_type:
                        fmap1_dw16 = rearrange(
                            fmap1_dw16, "(b t) m h w -> (b t) (h w) m", t=T, h=h, w=w
                        )
                        fmap2_dw16 = rearrange(
                            fmap2_dw16, "(b t) m h w -> (b t) (h w) m", t=T, h=h, w=w
                        )

                        fmap1_dw16, fmap2_dw16 = self.self_attn_blocks[att_ind](
                            fmap1_dw16, fmap2_dw16
                        )
                        fmap1_dw16, fmap2_dw16 = self.cross_attn_blocks[att_ind](
                            fmap1_dw16, fmap2_dw16
                        )

                        fmap1_dw16 = rearrange(
                            fmap1_dw16, "(b t) (h w) m -> (b t) m h w ", t=T, h=h, w=w
                        )
                        fmap2_dw16 = rearrange(
                            fmap2_dw16, "(b t) (h w) m -> (b t) m h w ", t=T, h=h, w=w
                        )

                    if "temporal" in self.attention_type:
                        fmap1_dw16 = self.time_attn_blocks[att_ind](fmap1_dw16, T=T)
                        fmap2_dw16 = self.time_attn_blocks[att_ind](fmap2_dw16, T=T)
                        
        return fmap1_dw16, fmap2_dw16

    def compute_qk_similarity(self, q, k, t=5):

        q = rearrange(q, "b c t h w -> (b t) c h w")
        k = rearrange(k, "b c t h w -> (b t) c h w")
        
        # # temporal attention
        bt, channels, height, width = q.shape
        b = bt // t
        hw = height * width
        global_max_pool = nn.AdaptiveMaxPool2d((height//4, width//4))
        q_, k_ = global_max_pool(q), global_max_pool(k)

        q_, k_ = q_.mean(dim=1).reshape(b, t, -1), k_.mean(dim=1).reshape(b, t, -1)
        # channel_similarity = torch.matmul(q_, k_.permute(0,1,3,2)) 

        sim = F.cosine_similarity(q_.unsqueeze(1), k_.unsqueeze(2), dim=-1).unsqueeze(1) 
  
        # # spatial attention
        # pool2d = nn.AvgPool2d(5, stride=2, padding=2)
        # q_, k_ = F.relu(pool2d(q)), F.relu(pool2d(k))
        # q_, k_ = q_.sum(1).reshape(b, 1, t, -1).contiguous(), k_.sum(1).reshape(b, 1, t, -1).permute(0, 1, 3, 2).contiguous()
        # q_, k_ = F.normalize(q_, p=2, dim=-1), F.normalize(k_, p=2, dim=-1)
        # spatial_similarity = torch.matmul(q_, k_) 
        
        # sim = 1.8 * (channel_similarity + spatial_similarity) / (channel_similarity + spatial_similarity).mean() 
        
        return sim

    # def compute_qk_similarity(self, q, k, t=5):

    #     # q = rearrange(q, "b c t h w -> (b t) c h w")
    #     # k = rearrange(k, "b c t h w -> (b t) c h w")
        
    #     # # temporal attention
    #     b, channels, t, height, width = q.shape

    #     pool3d = nn.AvgPool3d(kernel_size=(1, 9, 9), stride=(1, 4, 4), padding=(0, 4, 4))
    #     q_, k_ = torch.sigmoid(nn.AdaptiveAvgPool3d(output_size=1)(q)) * pool3d(q), torch.sigmoid(nn.AdaptiveAvgPool3d(output_size=1)(k)) * pool3d(k)

    #     q_ = rearrange(q_, "b c t h w -> b 1 t (c h w)")
    #     k_ = rearrange(k_, "b c t h w -> b 1 t (c h w)")
        
    #     q_, k_ = F.normalize(q_, p=2, dim=-1), F.normalize(k_, p=2, dim=-1)
    #     k_ = k_.permute(0, 1, 3, 2).contiguous()
        
    #     score = torch.matmul(q_, k_) + 1.
        
    #     return 1.2 * score
    
    def vis_attention(self, image1, current_query, selected_key, scale):
        import numpy as np
        import matplotlib.pyplot as plt
        import cv2
        attention_scores = torch.einsum('bsnd,bsmd->bnms', current_query.permute(0,2,1,3).contiguous().to(dtype=torch.bfloat16), \
                                         selected_key.permute(0,2,1,3).contiguous().to(dtype=torch.bfloat16)) / torch.sqrt(torch.tensor(128, dtype=torch.bfloat16)).float()
        
        attention_weights = torch.softmax(attention_scores, dim=2)
        
        _, _, h, w = image1.shape
        
        attention_map = attention_weights.reshape(h//(4*scale), w//(4*scale), -1, h//(4*scale), w//(4*scale))
        
        image_path = "/home/ywang/dataset/SouthKensington/outdoor/video119/images/left/left000082.png"  # 替换为你的图像路径
        current_image = cv2.imread(image_path)
        current_image = cv2.cvtColor(current_image, cv2.COLOR_BGR2RGB) 
        current_image_t = current_image.copy()
        current_image_t[30*8:31*8, 13*8:14*8]= (255,0,0)
        plt.imsave('current_image_t.svg',current_image_t)
        
        attention_map_t = attention_map[30, 13, 0, :].squeeze().detach().cpu().numpy()
        attention_map_t_resized = cv2.resize(attention_map_t * 1.5 / attention_map_t.max(), (w, h))[8:736-8,...]
        heatmap = cv2.applyColorMap(np.uint8(255 * attention_map_t_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        alpha = 0.1  # 透明度
        image_path = "/home/ywang/dataset/SouthKensington/outdoor/video119/images/left/left000085.png"  # 替换为你的图像路径
        original_image = cv2.imread(image_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB) 
        
        superimposed_image = cv2.addWeighted(original_image, 1 - alpha, heatmap, alpha, 0)
        
        
        #############################################################################
        # attention_mask = attention_map_t_resized > 0.1
        # original_image[attention_mask] = (255,0,0)
        
        plt.imsave('mask_image.svg',original_image)
        
        
    def forward_update_block(
        self,
        image1: torch.Tensor,
        update_block: nn.Module,
        corr_fn: CorrBlock1D,
        flow: torch.Tensor,
        net: torch.Tensor,  
        inp: torch.Tensor,
        motion_hidden_state: torch.Tensor,
        attn_block: nn.Module,
        predictions: List,
        uncertainties: List,
        iters: int,
        interp_scale: float,
        t: int,
    ):

        bt, c, h, w = inp.shape
        b = int(bt / t)
        top_k = 5
        
        query, key = attn_block.to_qk(inp).chunk(2, dim=1)
        query = rearrange(query, "(b t) c h w -> b c t h w", t=t)
        key = rearrange(key, "(b t) c h w -> b c t h w", t=t)
        
        pe = True
        if pe == True:
            temporal_encoding = get_temporal_positional_encoding(
                max_sequence_len = t,
                channels = c,
                device=inp.device,
                is_normalize=True,
                scale=1.,)
            
            temporal_encoding = temporal_encoding.unsqueeze(0).permute(0, 4, 1, 2, 3).contiguous()
            temporal_encoding = temporal_encoding.repeat(b, 1, 1, h, w)
            #
               
        if interp_scale == 4:
            alpha = -1
        elif interp_scale == 2:
            alpha = 2
        else:
            alpha = 3

        if interp_scale <= 4:
            sim_score = self.compute_qk_similarity(query, key, t=t)
            strive_time = torch.ones_like(sim_score)
        
            key = torch.cat([key, temporal_encoding], dim=1)
            
        else:
            if pe:
                key = key + temporal_encoding
                query = query + temporal_encoding
                
        for itr in range(iters):
            # if itr % 2 == 0:
            #     small_patch = False
            # else:
            #     small_patch = True

            flow = flow.detach()
            out_corrs = corr_fn(flow)
            # out_corrs = corr_fn(flow, None, small_patch=small_patch)
                        
            motion_features, motion_hidden_state, current_value = update_block.get_motion_and_value(flow, out_corrs, motion_hidden_state, inp)
            
            scale = c ** -0.5 * math.log(key.shape[1], 12000)
            uncertainty_score = update_block.get_uncertainty(torch.cat([net, current_value], dim=1))
              
            motion_features = rearrange(motion_features, "(b t) c h w -> b c t h w", t=t)
            current_value = rearrange(current_value, "(b t) c h w -> b c t h w", t=t)
                
             ### only activate at the high-resolution
            if interp_scale <= 4:
                add_item = t
                
                # quality-aware memory assessment module
                penality_score = torch.exp(-(strive_time) / (strive_time.sum(3, keepdim=True) + add_item)) 
                frame_confidence = uncertainty_score.reshape(b, 1, 1, t, -1).contiguous().mean(-1)
                frame_score = penality_score * sim_score + frame_confidence

                frame_idx = (torch.argsort(frame_score, dim=-1, descending=True))
                topk_indices = frame_idx[:,:, :, :top_k]
                mask = torch.zeros_like(frame_score, dtype=torch.bool)
                frame_mask = mask.scatter_(3, topk_indices, True)
                strive_time[frame_mask == True] += 1
                        
                motion_features_global = []
                
                for clip in range(t):
                    current_query = query[:, :, clip, ...]
                    if pe:
                        current_query = current_query + temporal_encoding[:, :, clip, ...]
                            
                    current_query = current_query.flatten(start_dim=2).permute(0, 2, 1).contiguous().unsqueeze(2)

                    temporal_mask  = frame_mask[:, :, clip, ...]
                    expanded_mask_key = temporal_mask.reshape(b, 1, t, 1, 1).expand_as(key)
                    expanded_mask_value = temporal_mask.reshape(b, 1, t, 1, 1).expand_as(current_value)
                      
                    selected_key = torch.masked_select(key, expanded_mask_key).view(b, 2*c, -1, h, w).contiguous()
                    selected_value = torch.masked_select(current_value, expanded_mask_value).view(b, c, -1, h, w).contiguous()
                    
                    # selected_idx = torch.masked_select(frame_idx[:, :, clip, ...], temporal_mask).view(b, 1, -1).contiguous()
                    selected_score = torch.masked_select(frame_score[:, :, clip, ...], temporal_mask).view(b, 1, -1).contiguous()
                    selected_score_norm = selected_score / selected_score.mean()
                    
                    # standard_weight_frame = torch.linspace(scaling_low, scaling_high, selected_key.size(2)).to(query) # num_frame
                    # new_weight_frame = torch.zeros(b, 1, selected_key.size(2)).to(query)
                    # new_weight_frame.scatter_(2, torch.argsort(selected_idx.reshape(b, 1, -1), dim=-1),
                    #                     selected_score_norm.view(1, 1, -1).repeat(b, 1, 1))

                    # dynamic memory modulation
                    selected_key = selected_key[:, :c,...] * selected_score_norm.view(b, 1, -1, 1, 1).contiguous() + selected_key[:, c:,...]
                    
                    # if pe == True:
                    #     selected_temporal_encoding = torch.masked_select(temporal_encoding, temporal_mask.unsqueeze(-1).unsqueeze(-1)).contiguous().reshape(b, c, -1, 1 ,1)
                    #     selected_key = selected_key + selected_temporal_encoding     
                                     
                    selected_key = selected_key.flatten(start_dim=2).permute(0, 2, 1).contiguous().unsqueeze(2)
                    selected_value = selected_value.flatten(start_dim=2).permute(0, 2, 1).contiguous().unsqueeze(2)
                        
                    hidden_states = flash_attn_func(current_query.to(dtype=torch.bfloat16), selected_key.to(dtype=torch.bfloat16), selected_value.to(dtype=torch.bfloat16), dropout_p=0.0, softmax_scale=scale, causal=False).float()
                    hidden_states = hidden_states.squeeze(2).permute(0, 2, 1).contiguous().view(b, c, h, w)
                    motion_features_global.append(motion_features[:, :, clip, ...] + update_block.aggregator.beta * hidden_states)
                
                motion_features = rearrange(motion_features, "b c t h w -> (b t) c h w")
                motion_features_global = torch.stack(motion_features_global, dim=1).reshape(-1, c, h, w).contiguous()
            
            else:                    
                current_query = query.flatten(start_dim=2).permute(0, 2, 1).contiguous().unsqueeze(2)
                current_key = key.flatten(start_dim=2).permute(0, 2, 1).contiguous().unsqueeze(2)
                current_value = current_value.flatten(start_dim=2).permute(0, 2, 1).contiguous().unsqueeze(2)

                hidden_states = flash_attn_func(current_query.to(dtype=torch.bfloat16), current_key.to(dtype=torch.bfloat16), current_value.to(dtype=torch.bfloat16), dropout_p=0.0, softmax_scale=scale, causal=False).float()
                hidden_states = hidden_states.squeeze(2).reshape(b, t, h, w, -1).permute(0, 1, 4, 2, 3).contiguous().reshape(bt, -1, h, w).contiguous()
                
                motion_features = rearrange(motion_features, "b c t h w -> (b t) c h w")
                motion_features_global = motion_features + update_block.aggregator.beta * hidden_states

            with autocast(enabled=self.mixed_precision, dtype=torch.bfloat16):
                net, up_mask, delta_flow = update_block(net, inp, motion_features, motion_features_global, t=t)
            
            flow = flow + delta_flow
            
            if self.use_convex_3d == True:
                flow_up = flow_out = self.convex_upsample_3d(flow, up_mask, rate=4, T=t)
            else:
                flow_up = flow_out = self.convex_upsample(flow, up_mask, rate=4)
                
            uncertainty_up = F.interpolate(uncertainty_score, scale_factor= 4 * interp_scale, mode='bilinear')
            
            if interp_scale > 1:
                flow_up = interp_scale * interp(
                    flow_out,
                    (
                        interp_scale * flow_out.shape[2],
                        interp_scale * flow_out.shape[3],
                    ),
                    )
                     
            flow_up = flow_up[:, :1]
            predictions.append(flow_up)
            uncertainties.append(uncertainty_up)
            
        del query, current_value, motion_features, motion_features_global, flow_up, uncertainty_up
        return flow_out, net, motion_hidden_state
   
    
    def channel_length(self, x):
        return torch.sqrt(torch.sum(torch.pow(x, 2), dim=1, keepdim=True) + 1e-3)


    def forward(self, image1, image2, flow_init=None, iters=10, test_mode=False):
        """Estimate optical flow between pair of frames"""
        # if input is list,
        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0

        b, T, c, h, w = image1.shape

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        hdim = self.hidden_dim

        image1 = rearrange(image1, "b t c h w -> (b t) c h w")
        image2 = rearrange(image2, "b t c h w -> (b t) c h w")
        
        with autocast(enabled=self.mixed_precision, dtype=torch.bfloat16):
            fmap1, fmap2 = self.fnet([image1, image2])
                
            net, inp = torch.split(fmap1, [hdim, hdim], dim=1)

            if self.use_cnet:    
                ## use convnet to estimate
                cnet, cnet_8x, cnet_16x = self.cnet(image1)
                net_4x, inp_4x = torch.split(cnet, [hdim, hdim], dim=1)

                net = (net + net_4x) / 2.0
                inp = (inp + inp_4x) / 2.0
                # inp = self.fusion_network(torch.cat([inp, inp_4x], dim=1))
                
            net = torch.tanh(net)
            inp = F.relu(inp)
            
            if self.init_flow:
                # init flow
                net_4x = rearrange(net, "(b t) c h w -> b c t h w", t=T)
                flow_init = self.update_block04.flow_head(net_4x)
                flow_init = rearrange(flow_init, " b c t h w -> (b t) c h w")
                
                net_4x = rearrange(net_4x, " b c t h w -> (b t) c h w")

                weight_update = .25 * self.update_block04.mask(net_4x)
                flow_init_up = self.convex_upsample(flow_init, weight_update, rate=4)
                 
            *_, h, w = fmap1.shape
                 
            # 1/4 -> 1/16
            # feature
            fmap1_dw16 = F.avg_pool2d(fmap1, 4, stride=4)
            fmap2_dw16 = F.avg_pool2d(fmap2, 4, stride=4)

            fmap1_dw16, fmap2_dw16 = self.forward_sst_block(fmap1_dw16, fmap2_dw16, T=T)
            
            net_dw16, inp_dw16 = torch.split(fmap1_dw16, [hdim, hdim], dim=1)
            
            if self.use_cnet:    
                ## use convnet to estimate
                net_16x, inp_16x = torch.split(cnet_16x, [hdim, hdim], dim=1)

                net_dw16 = (net_dw16 + net_16x) / 2.0
                inp_dw16 = (inp_dw16 + inp_16x) / 2.0
            
            net_dw16 = torch.tanh(net_dw16)
            inp_dw16 = F.relu(inp_dw16)
            
            fmap1_dw8 = (
                F.avg_pool2d(fmap1, 2, stride=2) + interp(fmap1_dw16, (h // 2, w // 2))
            ) / 2.0
            fmap2_dw8 = (
                F.avg_pool2d(fmap2, 2, stride=2) + interp(fmap2_dw16, (h // 2, w // 2))
            ) / 2.0

            net_dw8, inp_dw8 = torch.split(fmap1_dw8, [hdim, hdim], dim=1)
            if self.use_cnet:    
                ## use convnet to estimate
                net_8x, inp_8x = torch.split(cnet_8x, [hdim, hdim], dim=1)

                net_dw8 = (net_dw8 + net_8x) / 2.0
                inp_dw8 = (inp_dw8 + inp_8x) / 2.0
            
            net_dw8 = torch.tanh(net_dw8)
            inp_dw8 = F.relu(inp_dw8)
            
            # Cascaded refinement (1/16 + 1/8 + 1/4)
            predictions = []
            uncertainties = []

            flow = None
            flow_up = None
            
            if flow_init is not None:
                scale = fmap1_dw16.shape[2] / flow_init.shape[2]
                flow_dw16 = -scale * interp(flow_init, (fmap1_dw16.shape[2], fmap1_dw16.shape[3]))
            else:
                # zero initialization
                flow_dw16 = self.zero_init(fmap1_dw16)
                motion_hidden_state_dw16 = None
                
                # Recurrent Update Module
                # Update 1/16
                update_block = (
                    self.update_block16
                    if self.different_update_blocks
                    else self.update_block
                )
                
                corr_fn_att_dw16 = CorrBlock1D(fmap1_dw16, fmap2_dw16)
                flow, net_dw16, motion_hidden_state_dw16 = self.forward_update_block(
                    image1 = image1,
                    update_block=update_block,
                    corr_fn=corr_fn_att_dw16,
                    flow=flow_dw16,
                    net=net_dw16,
                    inp=inp_dw16,
                    motion_hidden_state=motion_hidden_state_dw16,
                    attn_block=self.att[0],
                    predictions=predictions,
                    uncertainties=uncertainties,
                    iters=iters // 2,
                    interp_scale=4,
                    t=T,
                )
                
                scale = fmap1_dw8.shape[2] / flow.shape[2]
                flow_dw8 = -scale * interp(flow, (fmap1_dw8.shape[2], fmap1_dw8.shape[3]))
                motion_hidden_state_dw8 = F.interpolate(motion_hidden_state_dw16, scale_factor=2, mode='bilinear',
                                        align_corners=True)
            
                net_dw8 = (
                    net_dw8
                    + interp(net_dw16, (2 * net_dw16.shape[2], 2 * net_dw16.shape[3]))
                ) / 2.0
                # Update 1/8

                # del image1, image2, fmap1_dw16, fmap2_dw16, net_dw16, inp_dw16, update_block
                
                update_block = (
                    self.update_block08
                    if self.different_update_blocks
                    else self.update_block
                    )

                corr_fn_dw8 = CorrBlock1D(fmap1_dw8, fmap2_dw8)
                flow, net_dw8, motion_hidden_state_dw8 = self.forward_update_block(
                    image1 = image1,
                    update_block=update_block,
                    corr_fn=corr_fn_dw8,
                    flow=flow_dw8,
                    net=net_dw8,
                    inp=inp_dw8,
                    motion_hidden_state=motion_hidden_state_dw8,
                    attn_block=self.att[1],
                    predictions=predictions,
                    uncertainties=uncertainties,
                    iters=iters // 2,
                    interp_scale=2,
                    t=T,
                )

                scale = h / flow.shape[2]
                flow = -scale * interp(flow, (h, w))
        
            motion_hidden_state = F.interpolate(motion_hidden_state_dw8, scale_factor=2, mode='bilinear',
                                        align_corners=True)
            net = (
                net + interp(net_dw8, (2 * net_dw8.shape[2], 2 * net_dw8.shape[3]))
            ) / 2.0
            
            del fmap1_dw8, fmap2_dw8, net_dw8, inp_dw8, update_block
                
            # Update 1/4
            update_block = (
                self.update_block04 if self.different_update_blocks else self.update_block
            )
            
            corr_fn = CorrBlock1D(fmap1, fmap2)
            flow, __, _ = self.forward_update_block(
                image1 = image1,
                update_block=update_block,
                corr_fn=corr_fn,
                flow=flow,
                net=net,
                inp=inp,
                attn_block=self.att[2],
                predictions=predictions,
                motion_hidden_state=motion_hidden_state,
                uncertainties=uncertainties,
                iters=iters,
                interp_scale=1,
                t=T,
            )

            del fmap1, fmap2, net, flow, flow_dw8, flow_dw16, inp, update_block
            
            predictions = torch.stack(predictions)
            uncertainties = torch.stack(uncertainties)
            
            predictions = rearrange(predictions, "d (b t) c h w -> d b t c h w", b=b, t=T)
            uncertainties = rearrange(uncertainties, "d (b t) c h w -> d b t c h w", b=b, t=T)
            
            flow_up = predictions[-1]
            uncertainty = uncertainties[-1]
            if test_mode:
                return flow_up, uncertainty

            if self.init_flow:
                flow_init_up = rearrange(flow_init_up[:, :1], "(b t) c h w -> b t c h w", b=b, t=T)
                return flow_init_up, predictions, uncertainties
            else: 
                return predictions, uncertainties