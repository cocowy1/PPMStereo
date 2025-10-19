# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.core.attention import LoFTREncoderLayer
try:
    from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
except:
    print('no flash attention installed')

# Ref: https://github.com/princeton-vl/RAFT/blob/master/core/update.py
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



class SKSepConvGRU3D(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192 + 128):
        super(SKSepConvGRU3D, self).__init__()
        self.convz1 = nn.Sequential(
            nn.Conv3d(hidden_dim+input_dim, hidden_dim, (1, 1, 15), padding=(0, 0, 7)),
            nn.GELU(),
            nn.Conv3d(hidden_dim, hidden_dim, (1, 1, 5), padding=(0, 0, 2)),
        )
        self.convr1 = nn.Sequential(
            nn.Conv3d(hidden_dim+input_dim, hidden_dim, (1, 1, 15), padding=(0, 0, 7)),
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


class BasicMotionEncoder(nn.Module):
    def __init__(self, cor_planes):
        super(BasicMotionEncoder, self).__init__()

        self.convc1 = nn.Conv2d(cor_planes, 256, 1, padding=0)
        self.convc2 = nn.Conv2d(256, 192, 3, padding=1)
        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv = nn.Conv2d(64 + 192, 128 - 2, 3, padding=1)

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))

        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)


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


class BasicUpdateBlock(nn.Module):
    def __init__(self, hidden_dim, cor_planes, mask_size=8, attention_type=None):
        super(BasicUpdateBlock, self).__init__()
        self.attention_type = attention_type
        if attention_type is not None:
            if "update_time" in attention_type:
                self.time_attn = TimeAttnBlock(dim=256, num_heads=8)

            if "update_space" in attention_type:
                self.space_attn = SpaceAttnBlock(dim=256, num_heads=8)

        self.encoder = BasicMotionEncoder(cor_planes)
        self.gru = SKSepConvGRU3D(hidden_dim=hidden_dim, input_dim=128 + hidden_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)

        self.mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, mask_size ** 2 * 9, 1, padding=0),
        )

    def forward(self, net, inp, corr, flow, upsample=True, t=1):
        motion_features = self.encoder(flow, corr)
        inp = torch.cat((inp, motion_features), dim=1)
        if self.attention_type is not None:
            if "update_time" in self.attention_type:
                inp = self.time_attn(inp, T=t)
            if "update_space" in self.attention_type:
                inp = self.space_attn(inp, T=t)
        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)

        # scale mask to balence gradients
        mask = 0.25 * self.mask(net)
        return net, mask, delta_flow


class FlowHead3D_FFT(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, t=5):
        super(FlowHead3D_FFT, self).__init__()
        self.conv1 = nn.Conv3d(input_dim, hidden_dim, (1, 3, 3), padding=1, bias=False)
        self.conv2 = nn.Conv3d(hidden_dim, 2, (1, 3, 3), padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.temporal_convolution = Temporal_FFT(input_dim=input_dim, hidden_dim=hidden_dim, clip_len=t)
        
    def forward(self, x):
        x_t = self.temporal_convolution(x)
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
        self.t = clip_len // 2 + 1
        
        # filter and spectrum learning 
        self.filter_g = nn.Conv2d(in_channels=input_dim*clip_len, out_channels=input_dim*self.t*2//self.factor_G, kernel_size=1, bias=False)
        self.filter1 = FFTLMul(dim=input_dim, shape=[self.t, 1, 1], bias=False)
        self.linear1 = FFTLinear(in_features=input_dim, out_features=input_dim, bias=False)
        self.norm1 = FFTBatchNorm(dim=input_dim)
        
        self.alpha1 = nn.Parameter(torch.zeros([1, input_dim, 1, 1, 1]))
        self.bias = nn.Parameter(torch.zeros([1, input_dim, 1, 1, 1]))
        
    def forward(self, x):
        # compute spectrum of input feature by FFT
        # x shape: b, c, t, h, w
        
        b, c, clip_len, h, w = x.shape
        x_t = torch.fft.rfft(x, dim=2, norm='ortho')
        # # concatenate input feature and correlation weights
        # x_c = torch.cat([x, w_o], dim=1)
        # predict intermediate filter
        
        iterm_filter = self.filter_g(x.reshape(b, -1, h, w))
        # expand intermediate filter to final filter
        final_filter = iterm_filter.unsqueeze(2).repeat(1, 1, self.factor_G, 1, 1)
        final_filter = final_filter.reshape(b, c, self.t, 2, h, w).permute(0,1,2,4,5,3).contiguous()
        # spectrum modulation
        y1 = self.norm1(self.linear1(self.filter1(x_t, final_filter)))
        # reconstruct visual feature by IFFT
        out = torch.fft.irfft(y1 * self.alpha1, n=clip_len, dim=2, norm='ortho')
        # skip connection
        out = x + out
        
        
class FlowHead3D(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super(FlowHead3D, self).__init__()
        self.conv1 = nn.Conv3d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv3d(hidden_dim, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class SequenceUpdateBlock3D(nn.Module):
    def __init__(self, hidden_dim, cor_planes, mask_size=8, attention_type=None):
        super(SequenceUpdateBlock3D, self).__init__()

        self.encoder = BasicMotionEncoder(cor_planes)
        self.gru = SepConvGRU3D(hidden_dim=hidden_dim, input_dim=128 + hidden_dim)
        self.flow_head = FlowHead3D(hidden_dim, hidden_dim=256)
        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim + 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim + 128, (mask_size ** 2) * 9, 1, padding=0),
        )
        self.attention_type = attention_type
        if attention_type is not None:
            if "update_time" in attention_type:
                self.time_attn = TimeAttnBlock(dim=256, num_heads=8)
            if "update_space" in attention_type:
                self.space_attn = SpaceAttnBlock(dim=256, num_heads=8)


    def forward(self, net, inp, corrs, flows, t, upsample=True):
        inp_tensor = []

        motion_features = self.encoder(flows, corrs)
        inp_tensor = torch.cat([inp, motion_features], dim=1)

        if self.attention_type is not None:
            if "update_time" in self.attention_type:
                inp_tensor = self.time_attn(inp_tensor, T=t)
            if "update_space" in self.attention_type:
                inp_tensor = self.space_attn(inp_tensor, T=t)

        net = rearrange(net, "(b t) c h w -> b c t h w", t=t)
        inp_tensor = rearrange(inp_tensor, "(b t) c h w -> b c t h w", t=t)

        net = self.gru(net, inp_tensor)

        delta_flow = self.flow_head(net)

        # scale mask to balance gradients
        net = rearrange(net, " b c t h w -> (b t) c h w")
        mask = 0.25 * self.mask(net)

        delta_flow = rearrange(delta_flow, " b c t h w -> (b t) c h w")
        return net, mask, delta_flow
