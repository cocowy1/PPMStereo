# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import math
from einops import rearrange
import torch
import math
import unfoldNd

import torch.nn as nn
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange
from models.core.attention import LoFTREncoderLayer
try:
    from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
except:
    print('no flash attention installed')

# Ref: https://github.com/princeton-vl/RAFT/blob/master/core/update.py


@torch.no_grad()
def get_temporal_positional_encoding(
        max_sequence_len,
        channels,
        device,
        is_normalize=False,
        scale=2*math.pi,
        is_debug=False,
):
    position = torch.arange(max_sequence_len, device=device)
    if is_normalize:
        position = position / position[-1] * scale
    if is_debug:
        print(f"{position = }")
    position.unsqueeze_(1)
    div_term = 1.0 / (10000.0 ** (
        torch.arange(0, channels, 2, device=device).float() / channels))
    position_div_term = position * div_term

    temporal_encoding = torch.zeros(
        (max_sequence_len, 1, 1, channels), device=device)
    temporal_encoding_sin = torch.sin(position_div_term)
    temporal_encoding_cos = torch.cos(position_div_term)
    temporal_encoding[:, 0, 0, 0::2] = temporal_encoding_sin
    temporal_encoding[:, 0, 0, 1::2] = temporal_encoding_cos

    # if is_debug:
    #     position_np = position.detach().cpu().numpy()
    #     position_div_term_np = position_div_term.detach().cpu().numpy()
    #     for i, p in enumerate(position_np[:0x10]):
    #         plt.plot(
    #             np.arange(position_div_term_np.shape[1]),
    #             position_div_term_np[i],
    #             label=f"line {p}",
    #         )
    #     plt.legend()
    #     plt.savefig("position_div_term.png")
    #     plt.close()
    #     temporal_encoding_sin_np = temporal_encoding_sin.detach().cpu().numpy()
    #     for i, p in enumerate(position_np[:0x10]):
    #         plt.subplot(4, 4, i+1)
    #         plt.plot(
    #             np.arange(temporal_encoding_sin_np.shape[1]),
    #             temporal_encoding_sin_np[i],
    #             label=f"line {p}",
    #         )
    #         plt.ylim(-1.05, 1.05)
    #         # plt.legend()
    #     plt.savefig("temporal_encoding_sin.png", dpi=300)
    #     plt.close()
    #     temporal_encoding_cos_np = temporal_encoding_cos.detach().cpu().numpy()
    #     for i, p in enumerate(position_np[:0x10]):
    #         plt.subplot(4, 4, i+1)
    #         plt.plot(
    #             np.arange(temporal_encoding_cos_np.shape[1]),
    #             temporal_encoding_cos_np[i],
    #             label=f"line {p}",
    #         )
    #         plt.ylim(-1.05, 1.05)
    #         # plt.legend()
    #     plt.savefig("temporal_encoding_cos.png", dpi=300)
    #     plt.close()

    return temporal_encoding


class PCBlock4_Deep_nopool_res(nn.Module):
    def __init__(self, C_in, C_out, k_conv, factor=1.5):
        super().__init__()
        self.conv_list = nn.ModuleList([
            nn.Conv2d(C_in, C_in, kernel, stride=1, padding=kernel//2, groups=C_in) for kernel in k_conv])

        self.ffn1 = nn.Sequential(
            nn.Conv2d(C_in, int(factor*C_in), 1, padding=0),
            nn.GELU(),
            nn.Conv2d(int(factor*C_in), C_in, 1, padding=0),
        )
        self.pw = nn.Conv2d(C_in, C_in, 1, padding=0)
        self.ffn2 = nn.Sequential(
            nn.Conv2d(C_in, int(factor*C_in), 1, padding=0),
            nn.GELU(),
            nn.Conv2d(int(factor*C_in), C_out, 1, padding=0),
        )

    def forward(self, x):
        x = F.gelu(x + self.ffn1(x))
        for conv in self.conv_list:
            x = F.gelu(x + conv(x))
        x = F.gelu(x + self.pw(x))
        x = self.ffn2(x)
        return x
    
    
class Attention_qk(nn.Module):
    def __init__(
        self,
        *,
        num_heads = 1,
        dim_head = 128,
    ):
        super().__init__()
        self.heads = num_heads
        self.scale = dim_head ** -0.5

        self.to_qk = nn.Conv2d(dim_head, dim_head*2, 1, bias=False)

    def forward(self, fmap):
        q, k = self.to_qk(fmap).chunk(2, dim=1)
        return q, k

        
class FlowHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192 + 128):
        super(SepConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(
            hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2)
        )
        self.convr1 = nn.Conv2d(
            hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2)
        )
        self.convq1 = nn.Conv2d(
            hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2)
        )

        self.convz2 = nn.Conv2d(
            hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0)
        )
        self.convr2 = nn.Conv2d(
            hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0)
        )
        self.convq2 = nn.Conv2d(
            hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0)
        )

    def forward(self, h, x):
        # horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q

        return h


class ConvGRU(nn.Module):
    def __init__(self, hidden_dim, input_dim, kernel_size=3):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(
            hidden_dim + input_dim, hidden_dim, kernel_size, padding=kernel_size // 2
        )
        self.convr = nn.Conv2d(
            hidden_dim + input_dim, hidden_dim, kernel_size, padding=kernel_size // 2
        )
        self.convq = nn.Conv2d(
            hidden_dim + input_dim, hidden_dim, kernel_size, padding=kernel_size // 2
        )

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)

        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r * h, x], dim=1)))

        h = (1 - z) * h + z * q
        return h


class SKSepConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(SKSepConvGRU, self).__init__()
        
        self.convz1 = nn.Sequential(
            nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1, 15), padding=(0, 7)),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, (1, 5), padding=(0, 2)),
        )
        self.convr1 = nn.Sequential(
            nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1, 15), padding=(0, 7)),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, (1, 5), padding=(0, 2)),
        )
        self.convq1 = nn.Conv2d(
            hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2)
        )

        self.convz2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convr2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convq2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))


    def forward(self, h, x):
        # horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r*h, x], dim=1)))
        h = (1-z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r*h, x], dim=1)))
        h = (1-z) * h + z * q

        return h



class SKSepConvGRU3D(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192 + 128 + 1):
        super(SKSepConvGRU3D, self).__init__()
        self.convz1 = nn.Sequential(
            nn.Conv3d(hidden_dim + input_dim, hidden_dim, (1, 1, 15), padding=(0, 0, 7)),
            nn.GELU(),
            nn.Conv3d(hidden_dim, hidden_dim, (1, 1, 5), padding=(0, 0, 2)),
        )
        self.convr1 = nn.Sequential(
            nn.Conv3d(hidden_dim + input_dim, hidden_dim, (1, 1, 15), padding=(0, 0, 7)),
            nn.GELU(),
            nn.Conv3d(hidden_dim, hidden_dim, (1, 1, 5), padding=(0, 0, 2)),
        )
        self.convq1 = nn.Conv3d(
            hidden_dim + input_dim, hidden_dim, (1, 1, 5), padding=(0, 0, 2)
        )

        self.convz2 = nn.Conv3d(
            hidden_dim + input_dim, hidden_dim, (1, 5, 1), padding=(0, 2, 0)
        )
        self.convr2 = nn.Conv3d(
            hidden_dim + input_dim, hidden_dim, (1, 5, 1), padding=(0, 2, 0)
        )
        self.convq2 = nn.Conv3d(
            hidden_dim + input_dim, hidden_dim, (1, 5, 1), padding=(0, 2, 0)
        )

        self.convz3 = nn.Conv3d(
            hidden_dim + input_dim, hidden_dim, (5, 1, 1), padding=(2, 0, 0)
        )
        self.convr3 = nn.Conv3d(
            hidden_dim + input_dim, hidden_dim, (5, 1, 1), padding=(2, 0, 0)
        )
        self.convq3 = nn.Conv3d(
            hidden_dim + input_dim, hidden_dim, (5, 1, 1), padding=(2, 0, 0)
        )

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q

        # time
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz3(hx))
        r = torch.sigmoid(self.convr3(hx))
        q = torch.tanh(self.convq3(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q

        return h


class SepConvGRU3D(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192 + 128):
        super(SepConvGRU3D, self).__init__()
        self.convz1 = nn.Conv3d(
            hidden_dim + input_dim, hidden_dim, (1, 1, 5), padding=(0, 0, 2)
        )
        self.convr1 = nn.Conv3d(
            hidden_dim + input_dim, hidden_dim, (1, 1, 5), padding=(0, 0, 2)
        )
        self.convq1 = nn.Conv3d(
            hidden_dim + input_dim, hidden_dim, (1, 1, 5), padding=(0, 0, 2)
        )

        self.convz2 = nn.Conv3d(
            hidden_dim + input_dim, hidden_dim, (1, 5, 1), padding=(0, 2, 0)
        )
        self.convr2 = nn.Conv3d(
            hidden_dim + input_dim, hidden_dim, (1, 5, 1), padding=(0, 2, 0)
        )
        self.convq2 = nn.Conv3d(
            hidden_dim + input_dim, hidden_dim, (1, 5, 1), padding=(0, 2, 0)
        )

        self.convz3 = nn.Conv3d(
            hidden_dim + input_dim, hidden_dim, (5, 1, 1), padding=(2, 0, 0)
        )
        self.convr3 = nn.Conv3d(
            hidden_dim + input_dim, hidden_dim, (5, 1, 1), padding=(2, 0, 0)
        )
        self.convq3 = nn.Conv3d(
            hidden_dim + input_dim, hidden_dim, (5, 1, 1), padding=(2, 0, 0)
        )

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q

        # time
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz3(hx))
        r = torch.sigmoid(self.convr3(hx))
        q = torch.tanh(self.convq3(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q

        return h


class RelPosEmb(nn.Module):
    def __init__(
            self,
            max_pos_size,
            dim_head
    ):
        super().__init__()
        self.rel_height = nn.Embedding(2 * max_pos_size - 1, dim_head)
        self.rel_width = nn.Embedding(2 * max_pos_size - 1, dim_head)

        deltas = torch.arange(max_pos_size).view(1, -1) - torch.arange(max_pos_size).view(-1, 1)
        rel_ind = deltas + max_pos_size - 1
        self.register_buffer('rel_ind', rel_ind)

    def forward(self, q):
        batch, heads, h, w, c = q.shape
        height_emb = self.rel_height(self.rel_ind[:h, :h].reshape(-1))
        width_emb = self.rel_width(self.rel_ind[:w, :w].reshape(-1))

        height_emb = rearrange(height_emb, '(x u) d -> x u () d', x=h)
        width_emb = rearrange(width_emb, '(y v) d -> y () v d', y=w)

        height_score = einsum('b h x y d, x u v d -> b h x y u v', q, height_emb)
        width_score = einsum('b h x y d, y u v d -> b h x y u v', q, width_emb)

        return height_score + width_score


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        q, k, v = qkv, qkv, qkv

        attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C).contiguous()
        x = self.proj(x)
        return x


class BasicMotionEncoder(nn.Module):
    def __init__(self, cor_planes):
        super(BasicMotionEncoder, self).__init__()

        self.convc1 = nn.Conv2d(cor_planes, 256, 1, padding=0)
        self.convc2 = nn.Conv2d(256, 192, 3, padding=1)
        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv = nn.Conv2d(64 + 192, 128 - 2, 3, padding=1)

    def forward(self, flow, corr, t):

        cor = F.gelu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))

        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)


class BasicMotionEncoder_v2(nn.Module):
    def __init__(self, cor_planes):
        super(BasicMotionEncoder_v2, self).__init__()
        self.k_conv = [1, 7]
        self.convc1 = PCBlock4_Deep_nopool_res(cor_planes, 256, k_conv=self.k_conv)
        # self.convc1 = nn.Conv2d(cor_planes, 256, 1, padding=0)
        
        self.convc2 = nn.Conv2d(256, 192, 3, padding=1)
        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)
        
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
        # self.convf2 = PCBlock4_Deep_nopool_res(128, 64, k_conv=self.k_conv)
        
        self.final_conv = nn.Conv2d(64 + 192 + 64, 128 - 2 + 64, 3, padding=1)
        #self.conv = PCBlock4_Deep_nopool_res(64 + 192 + 64, 128 - 2 + 64, k_conv=self.k_conv)

        # self.init_hidden_state = nn.Parameter(torch.randn(1, 64, 1, 1))
        self.init_conv = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(64, 64, 3, padding=1))
        
    def forward(self, flow, corr, motion_hidden_state, inp):
        BN, _, H, W = flow.shape
        
        if motion_hidden_state is None:
            # motion_hidden_state = self.init_hidden_state.repeat(BN, 1, H, W)
            motion_hidden_state =  self.init_conv(inp)
            
        cor = F.gelu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))

        cor_flo = torch.cat([cor, flo, motion_hidden_state], dim=1)
        out = F.relu(self.final_conv(cor_flo))
        out, motion_hidden_state = torch.split(out, [126, 64], dim=1)

        return torch.cat([out, flow], dim=1), motion_hidden_state
    
    
class SKMotionEncoder6_Deep_nopool_res_Mem_skflow(nn.Module):
    def __init__(self, cor_planes, k_conv):
        super().__init__()
        self.cor_planes = cor_planes
        self.k_conv = k_conv
        self.convc1 = PCBlock4_Deep_nopool_res(cor_planes, 256, k_conv=self.k_conv)
        self.convc2 = PCBlock4_Deep_nopool_res(256, 192, k_conv=self.k_conv)

        self.convf1 = nn.Conv2d(2, 128, 1, 1, 0)
        self.convf2 = PCBlock4_Deep_nopool_res(128, 64, k_conv=self.k_conv)

        self.conv = PCBlock4_Deep_nopool_res(64+192, 128-2, k_conv=self.k_conv)

    def forward(self, flow, corr):
        cor = F.gelu(self.convc1(corr))

        cor = self.convc2(cor)

        flo = self.convf1(flow)
        flo = self.convf2(flo)

        cor_flo = torch.cat([cor, flo], dim=1)
        out = self.conv(cor_flo)

        return torch.cat([out, flow], dim=1)
    
     
class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(ResidualBlock, self).__init__()
  
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        
        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.BatchNorm2d(planes)
        
        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.Sequential()

        if stride == 1 and in_planes == planes:
            self.downsample = None
        
        else:    
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)

    def forward(self, x):
        y = x
        y = self.conv1(y)
        y = self.norm1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.norm2(y)
        y = self.relu(y)

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x+y)
    
class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TimeAttnBlock(nn.Module):
    def __init__(self, dim=256, num_heads=8):
        super(TimeAttnBlock, self).__init__()
        self.temporal_attn = Attention(dim, num_heads=8, qkv_bias=False, qk_scale=None)
        self.temporal_fc = nn.Linear(dim, dim)
        self.temporal_norm1 = nn.LayerNorm(dim)

        nn.init.constant_(self.temporal_fc.weight, 0)
        nn.init.constant_(self.temporal_fc.bias, 0)

    def forward(self, x, T=1):
        _, _, h, w = x.shape

        x = rearrange(x, "(b t) m h w -> (b h w) t m", h=h, w=w, t=T)
        res_temporal1 = self.temporal_attn(self.temporal_norm1(x))
        res_temporal1 = rearrange(
            res_temporal1, "(b h w) t m -> b (h w t) m", h=h, w=w, t=T
        )
        res_temporal1 = self.temporal_fc(res_temporal1)
        res_temporal1 = rearrange(
            res_temporal1, " b (h w t) m -> b t m h w", h=h, w=w, t=T
        )
        x = rearrange(x, "(b h w) t m -> b t m h w", h=h, w=w, t=T)
        x = x + res_temporal1
        x = rearrange(x, "b t m h w -> (b t) m h w", h=h, w=w, t=T)
        return x


class SpaceAttnBlock(nn.Module):
    def __init__(self, dim=256, num_heads=8):
        super(SpaceAttnBlock, self).__init__()
        self.encoder_layer = LoFTREncoderLayer(dim, nhead=num_heads, attention="linear")

    def forward(self, x, T=1):
        _, _, h, w = x.shape
        x = rearrange(x, "(b t) m h w -> (b t) (h w) m", h=h, w=w, t=T)
        x = self.encoder_layer(x, x)
        x = rearrange(x, "(b t) (h w) m -> (b t) m h w", h=h, w=w, t=T)
        return x


class Aggregate(nn.Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 128,
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head

        self.to_v = nn.Conv2d(dim, inner_dim, 1, bias=False)
        self.beta = nn.Parameter(torch.zeros(1))
        
        if dim != inner_dim:
            self.project = nn.Conv2d(inner_dim, dim, 1, bias=False)
        else:
            self.project = None

    def forward(self, attn, fmap):
        heads, b, c, h, w = self.heads, *fmap.shape

        v = self.to_kv(fmap)
        v = rearrange(v, 'b (h d) x y -> b h (x y) d', h=heads)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)

        if self.project is not None:
            out = self.project(out)

        out = fmap + self.beta * out

        return out


class FlowHead3D(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super(FlowHead3D, self).__init__()
        self.conv1 = nn.Conv3d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv3d(hidden_dim, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class FlowHead3D_FFT(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, t=5):
        super(FlowHead3D_FFT, self).__init__()
        self.conv1 = nn.Conv3d(input_dim, hidden_dim, kernel_size=(1, 5, 5), padding=(0, 2, 2), bias=False)
        self.conv2 = nn.Conv3d(hidden_dim, 2, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.temporal_convolution = Temporal_FFT(input_dim=input_dim, hidden_dim=hidden_dim, clip_len=t)
    def forward(self, x):
        x_t = self.temporal_convolution(x).abs()
        out = self.conv2(self.relu(self.conv1(x_t)))
        
        return out

class FFTLMul(nn.Module):
    def __init__(self, dim, shape, bias=True):
        super().__init__()
        self.dim = dim
        self.shape = shape
        self.bias = bias
    
    def forward(self, x, w):
        x = x * torch.view_as_complex(w)
        return x

class FFTLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.complex_weight = nn.Parameter(torch.randn([out_features, in_features, 2], dtype=torch.float32) * 0.02)
        if bias:
            bias_shape = [1, out_features, 1, 1, 1, 2]
            self.complex_bias = nn.Parameter(torch.zeros(bias_shape, dtype=torch.float32))

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias
        )

    def forward(self, x):
        B, C, T, H, W = x.shape
        w = torch.view_as_complex(self.complex_weight).unsqueeze(0).repeat(B, 1, 1)
        x = torch.bmm(w, x.reshape(B, C, T * H * W)).reshape(B, self.out_features, T, H, W)
        if self.bias:
            x = x + torch.view_as_complex(self.complex_bias)
        return x

class FFTBatchNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.norm = nn.BatchNorm3d(dim, affine=False)

    def extra_repr(self) -> str:
        return 'dim={}'.format(
            self.dim
        )

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = torch.view_as_real(x).reshape(B, C, T, H, W * 2)
        x = self.norm(x)
        x = x.reshape(B, C, T, H, W, 2)
        x = torch.view_as_complex(x)
        return x


class Temporal_FFT(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, clip_len=5):
        super(Temporal_FFT, self).__init__()
        self.factor_G = 16
        
        # filter and spectrum learning 
        self.filter_g = nn.Sequential(
            nn.Conv3d(in_channels=input_dim, out_channels=input_dim*2, kernel_size=(3,3,3), padding=(1,1,1), bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=input_dim*2, out_channels=input_dim*2, kernel_size=(3,3,3), padding=(1,1,1), bias=False))
        
        self.filter1 = FFTLMul(dim=input_dim, shape=[clip_len, 1, 1], bias=False)
        self.linear1 = FFTLinear(in_features=input_dim, out_features=input_dim, bias=False)
        self.norm1 = FFTBatchNorm(dim=input_dim)
        
        self.alpha1 = nn.Parameter(torch.zeros([1, input_dim, 1, 1, 1]))
        self.bias = nn.Parameter(torch.zeros([1, input_dim, 1, 1, 1]))
        
    def forward(self, x):
        # compute spectrum of input feature by FFT
        # x shape: b, c, t, h, w
        b, c, clip_len, h, w = x.shape
        x_t = torch.fft.fft(x, dim=2, norm='ortho')
        # # concatenate input feature and correlation weights
        # x_c = torch.cat([x, w_o], dim=1)
        # predict intermediate filter
        
        iterm_filter = self.filter_g(x)
        # expand intermediate filter to final filter
        final_filter = iterm_filter.unsqueeze(2)
        final_filter = final_filter.reshape(b, c, 2, clip_len, h, w).permute(0,1,3,4,5,2).contiguous()
        
        # spectrum modulation
        y1 = self.norm1(self.linear1(self.filter1(x_t, final_filter)))
        # reconstruct visual feature by IFFT
        out = torch.fft.ifft(y1 * self.alpha1, n=clip_len, dim=2, norm='ortho')
        # skip connection
        out = x + out
        
        return out
    
    
### old version
# class SequenceUpdateBlock3D(nn.Module):
#     def __init__(self, hidden_dim, cor_planes, mask_size=8, attention_type=None):
#         super(SequenceUpdateBlock3D, self).__init__()

#         self.encoder = BasicMotionEncoder(cor_planes)
#         self.gru = SKSepConvGRU3D(hidden_dim=hidden_dim, input_dim=256 + hidden_dim)
#         self.flow_head = FlowHead3D_FFT(hidden_dim, hidden_dim=256, t=5)
                
#         self.k_conv = [3, 15]
#         # self.gru_new = PCBlock4_Deep_nopool_res_3d(256 + hidden_dim, hidden_dim, self.k_conv)
#         # self.motion_agg = MotionGate3D(hidden_dim=hidden_dim, input_dim=128 + hidden_dim)

#         # self.flow_head = PCBlock4_Deep_nopool_res_3d(128, 2, k_conv=self.k_conv)
        
#         self.mask = nn.Sequential(
#             nn.Conv2d(hidden_dim, hidden_dim + 128, 3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(hidden_dim + 128, (mask_size ** 2) * 9, 1, padding=0),
#         )
#         self.attention_type = attention_type
#         if attention_type is not None:
#             if "update_time" in attention_type:
#                 self.time_attn = TimeAttnBlock(dim=384, num_heads=8)
#             if "update_space" in attention_type:
#                 self.space_attn = SpaceAttnBlock(dim=384, num_heads=8)
        
#         self.aggregator = Aggregate(dim=128, dim_head=128, heads=1)


#     def get_motion_and_value(self, flow, corr):
#         motion_features = self.encoder(flow, corr)
#         value = self.aggregator.to_v(motion_features)
        
#         return motion_features, value


#     def update_query(self, query, key, kernel_size, scale, t):
#         b, c, t, h, w =query.shape
#         padding = tuple(int(x // 2) for x in kernel_size)
#         lib_module = unfoldNd.UnfoldNd(
#         kernel_size, padding=padding, dilation=1, stride=1)
        
#         key = lib_module(key) # b, c*kernel_t*kernel_h*kernel_w, L
#         points = math.prod(kernel_size)
#         key = key.reshape(b, c, points, -1).permute(0, 3, 1, 2).contiguous() # b, t*h*w, c, points
#         query = query.reshape(b, c, -1).unsqueeze(3).permute(0, 2, 3, 1).contiguous() # b, t*h*w, 1, c
#         value = key.transpose(3, 2) # b, t*h*w, points, c
        
#         sim = torch.matmul(query, key) * scale # b, t*h*w, 1, points
#         prob = F.softmax(sim, dim=-1) # b, t*h*w, 1, points

#         out = torch.matmul(prob, value)  #  b, t*h*w, 1, c

#         return out
    
    
#     def forward(self, net, inp, motion_features, motion_features_global, upsample=True, t=1):
        
#         # motion_features_aggregated = self.motion_agg(motion_features, motion_features_global, t=t)
#         # inp_tensor = []
        
#         inp_tensor = torch.cat([inp, motion_features, motion_features_global], dim=1)
        
#         del motion_features, motion_features_global
        
#         if self.attention_type is not None:
#             if "update_time" in self.attention_type:
#                 inp_tensor = self.time_attn(inp_tensor, T=t)
#             if "update_space" in self.attention_type:
#                 inp_tensor = self.space_attn(inp_tensor, T=t)

#         net = rearrange(net, "(b t) c h w -> b c t h w", t=t)
#         inp_tensor = rearrange(inp_tensor, "(b t) c h w -> b c t h w", t=t)

#         net = self.gru(net, inp_tensor)
#         # net = self.gru_new(torch.cat([net, inp_tensor], dim=1))
#         delta_flow = self.flow_head(net)[:, :2]

        
#         # scale mask to balance gradients
#         net = rearrange(net, " b c t h w -> (b t) c h w")
#         mask = 0.25 * self.mask(net)

#         delta_flow = rearrange(delta_flow, " b c t h w -> (b t) c h w")
#         return net, mask, delta_flow
    
    
## new version 1.23, add uncertainty
class SequenceUpdateBlock3D(nn.Module):
    def __init__(self, hidden_dim, cor_planes, mask_size=8, use_convex_3d=False, attention_type=None):
        super(SequenceUpdateBlock3D, self).__init__()

        self.encoder = BasicMotionEncoder_v2(cor_planes)
        self.gru = SKSepConvGRU3D(hidden_dim=hidden_dim, input_dim=256 + hidden_dim)
        self.flow_head = FlowHead3D(hidden_dim, hidden_dim=256,)
        
        # self.gamma = nn.Parameter(torch.ones(1))
        self.uncertainty = nn.Sequential(nn.Conv2d(hidden_dim + 128, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, 1, padding=0),
            nn.Sigmoid(),
        )
        
        # self.k_conv = [1, 7]
        # self.mlp = PCBlock4_Deep_nopool_res(cor_planes, cor_planes, self.k_conv)
        
        # self.gru = PCBlock4_Deep_nopool_res_3d(384 + hidden_dim, hidden_dim, self.k_conv)
        # self.motion_agg = MotionGate3D(hidden_dim=hidden_dim, input_dim=128 + hidden_dim)

        # self.flow_head = PCBlock4_Deep_nopool_res_3d(128, 2, k_conv=self.k_conv)
        self.use_convex_3d = use_convex_3d
        if self.use_convex_3d == True:
            self.mask_3d = nn.Sequential(
            nn.Conv3d(hidden_dim, hidden_dim + 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_dim + 128, (mask_size ** 2) * 27, 1, padding=0),
            )
        else:
            self.mask_2d = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim + 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim + 128, (mask_size ** 2) * 9, 1, padding=0),
            )
                        
        self.attention_type = attention_type
        if attention_type is not None:
            if "update_time" in attention_type:
                self.time_attn = TimeAttnBlock(dim=384, num_heads=8)
            if "update_space" in attention_type:
                self.space_attn = SpaceAttnBlock(dim=384, num_heads=8)
        
        self.aggregator = Aggregate(dim=128, dim_head=128, heads=1)

        # self.MotionScalar = torch.nn.Sequential(
        #                 torch.nn.Conv2d(in_channels=128 + 1, out_channels=128, kernel_size=5, stride=1, padding=2, bias=False),
        #                 nn.ReLU(inplace=True),
        #                 torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0)
        #             )
        # self.MotionOffset = torch.nn.Sequential(
        #                 torch.nn.Conv2d(in_channels=128 + 1, out_channels=128, kernel_size=5, stride=1, padding=2, bias=False),
        #                 nn.ReLU(inplace=True),
        #                 torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0)
        #             )
                    
    def get_uncertainty(self, net):
        uncertainty = self.uncertainty(net)
        return uncertainty

    def get_scalar_and_offset(self, motion):
        MotionScalar = self.MotionScalar(motion, dim=1)
        MotionOffset = self.MotionOffset(motion, dim=1)
        return MotionScalar, MotionOffset
       
    def get_motion_and_value(self, flow, corr, motion_hidden_state, inp):
        # corr = self.mlp(corr)
        motion_features, motion_hidden_state = self.encoder(flow, corr, motion_hidden_state, inp)
        value = self.aggregator.to_v(motion_features)
        
        return motion_features, motion_hidden_state, value

    def update_query(self, query, key, kernel_size, scale, t):
        b, c, t, h, w =query.shape
        padding = tuple(int(x // 2) for x in kernel_size)
        lib_module = unfoldNd.UnfoldNd(
        kernel_size, padding=padding, dilation=1, stride=1)
        
        key = lib_module(key) # b, c*kernel_t*kernel_h*kernel_w, L
        points = math.prod(kernel_size)
        key = key.reshape(b, c, points, -1).permute(0, 3, 1, 2).contiguous() # b, t*h*w, c, points
        query = query.reshape(b, c, -1).unsqueeze(3).permute(0, 2, 3, 1).contiguous() # b, t*h*w, 1, c
        value = key.transpose(3, 2) # b, t*h*w, points, c
        
        sim = torch.matmul(query, key) * scale # b, t*h*w, 1, points
        prob = F.softmax(sim, dim=-1) # b, t*h*w, 1, points

        out = torch.matmul(prob, value)  #  b, t*h*w, 1, c

        return out
    
    def forward(self, net, inp, motion_features, motion_features_global, t=1):
        
        # motion_features_aggregated = self.motion_agg(motion_features, motion_features_global, t=t)
        # inp_tensor = []
        
        inp_tensor = torch.cat([inp, motion_features, motion_features_global], dim=1)
        del motion_features, motion_features_global
        
        if self.attention_type is not None:
            if "update_time" in self.attention_type:
                inp_tensor = self.time_attn(inp_tensor, T=t)
            if "update_space" in self.attention_type:
                inp_tensor = self.space_attn(inp_tensor, T=t)

        net = rearrange(net, "(b t) c h w -> b c t h w", t=t)
        inp_tensor = rearrange(inp_tensor, "(b t) c h w -> b c t h w", t=t)
        
        net = self.gru(net, inp_tensor)
        delta_flow = self.flow_head(net)
        
        b, c, t, h, w = net.shape
        # scale mask to balance gradients
        if self.use_convex_3d == True:
            mask = 0.25 * self.mask_3d(net)
            mask = rearrange(mask, " b c t h w -> (b t) c h w")
            net = rearrange(net, " b c t h w -> (b t) c h w")
        else:
            net = rearrange(net, " b c t h w -> (b t) c h w")
            mask = 0.25 * self.mask_2d(net)

        delta_flow = rearrange(delta_flow, " b c t h w -> (b t) c h w")
        
        return net, mask, delta_flow
    

class PCBlock4_Deep_nopool_res(nn.Module):
    def __init__(self, C_in, C_out, k_conv):
        super().__init__()
        self.conv_list = nn.ModuleList([
            nn.Conv2d(C_in, C_in, kernel, stride=1, padding=kernel//2, groups=C_in) for kernel in k_conv])

        self.ffn1 = nn.Sequential(
            nn.Conv2d(C_in, int(1.5*C_in), 1, padding=0),
            nn.GELU(),
            nn.Conv2d(int(1.5*C_in), C_in, 1, padding=0),
        )
        self.pw = nn.Conv2d(C_in, C_in, 1, padding=0)
        self.ffn2 = nn.Sequential(
            nn.Conv2d(C_in, int(1.5*C_in), 1, padding=0),
            nn.GELU(),
            nn.Conv2d(int(1.5*C_in), C_out, 1, padding=0),
        )

    def forward(self, x):
        x = F.gelu(x + self.ffn1(x))
        for conv in self.conv_list:
            x = F.gelu(x + conv(x))
        x = F.gelu(x + self.pw(x))
        x = self.ffn2(x)
        return x
    
    
class SequenceUpdateBlock(nn.Module):
    def __init__(self, hidden_dim, cor_planes, mask_size=8):
        
        super(SequenceUpdateBlock, self).__init__()
        self.k_conv = [1, 15]
        
        # self.mlp = PCBlock4_Deep_nopool_res(cor_planes, cor_planes, self.k_conv)
        self.encoder = BasicMotionEncoder(cor_planes)
        self.gru = SKSepConvGRU(hidden_dim=hidden_dim, input_dim=256 + hidden_dim + 1)
        # self.flow_head_2d = FlowHead(hidden_dim, hidden_dim=256)

        # self.encoder_new = SKMotionEncoder6_Deep_nopool_res_Mem_skflow(cor_planes, self.k_conv)
        # self.gru = PCBlock4_Deep_nopool_res(256 + hidden_dim + hidden_dim, hidden_dim, k_conv=self.PCUpdater_conv)
        # self.flow_head_2d = PCBlock4_Deep_nopool_res(128, 2, k_conv=self.k_conv)
        
        self.flow_head_2d = FlowHead(hidden_dim, hidden_dim=256)
        
        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim + 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim + 128, (mask_size ** 2) * 9, 1, padding=0),
        )
        
        self.aggregator = Aggregate(dim=128, dim_head=128, heads=1)
        
    def get_motion_and_value(self, flow, corr):
        # corr = self.mlp(corr)
        motion_features = self.encoder(flow, corr)
        value = self.aggregator.to_v(motion_features)
        return motion_features, value
    
    def forward(self, net, inp, motion_features, motion_features_global, att, t):
        inp_cat = torch.cat([inp, motion_features, motion_features_global], dim=1)

        # Attentional update
        net = self.gru(att, net, inp_cat)
        # net = self.gru(net, inp_cat)
        delta_flow = self.flow_head_2d(net)

        # scale mask to balence gradients
        mask = .25 * self.mask(net)
        return net, mask, delta_flow
    