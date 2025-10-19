# Copyright (c) Meta Platforms, Inc. and affiliates.

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
import math



class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

class Block(nn.Module):
    """ ConvNeXtV2 Block.
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class ConvNeXtV2(nn.Module):
    """ ConvNeXt V2
        
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans=3, num_classes=1000, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], 
                 drop_path_rate=0., head_init_scale=1.
                 ):
        super().__init__()
        self.depths = depths
        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        x_list = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            x_list.append(x)
        return x_list # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x_list = self.forward_features(x)
        return x_list[0], x_list[1], x_list[2], x_list[3]

def convnextv2_atto(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[40, 80, 160, 320], **kwargs)
    return model

def convnextv2_femto(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[48, 96, 192, 384], **kwargs)
    return model

def convnext_pico(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[64, 128, 256, 512], **kwargs)
    return model

def convnextv2_nano(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 8, 2], dims=[80, 160, 320, 640], **kwargs)
    return model

def convnextv2_tiny(**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    return model

def convnextv2_base(**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    return model

def convnextv2_large(**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    return model

def convnextv2_huge(**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[352, 704, 1408, 2816], **kwargs)
    return model


class SubModule(nn.Module):
    def __init__(self):
        super(SubModule, self).__init__()

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1)

class Feature(SubModule):
    def __init__(self, model_name, output_dim=256):
        super(Feature, self).__init__()
        
        if model_name == 'base':
            self.convnext = convnextv2_base()
        elif model_name == 'large':
            self.convnext = convnextv2_large()
        elif model_name == 'huge':
            self.convnext = convnextv2_huge()
        elif model_name == 'tiny':
            block_dims = [96, 192, 384, 768]
            self.convnext = convnextv2_tiny()
        elif model_name == 'nano':
            block_dims = [80, 160, 320]
            self.convnext = convnextv2_nano()
        else:
            raise NotImplementedError

        model_path = "/home/ywang/my_projects/MemStereo/ckpt/convnext/convnextv2_tiny_22k_384_ema.pt"
        self.convnext.load_state_dict(torch.load(model_path)["model"], strict=True)
        self.convnext.eval()

        self.upconv_16 = nn.Sequential(nn.Upsample(scale_factor=2),
                                        nn.Conv2d(block_dims[3], output_dim, 3, 1, 1),
                                        nn.InstanceNorm2d(output_dim, affine=False),
                                        nn.ReLU())

        self.upconv_8 = nn.Sequential(nn.Upsample(scale_factor=2),
                                        nn.Conv2d(output_dim, output_dim, 3, 1, 1),
                                        nn.InstanceNorm2d(output_dim, affine=False),
                                        nn.ReLU(),)
        
        self.upconv_4 = nn.Sequential(nn.Upsample(scale_factor=2),
                                        nn.Conv2d(output_dim, output_dim, 3, 1, 1),
                                        nn.InstanceNorm2d(output_dim, affine=False),
                                        nn.ReLU(),)
             
        self.decode_16x = nn.Sequential(nn.Conv2d(block_dims[2] + output_dim, output_dim, 1, 1, 0),
                                        nn.InstanceNorm2d(output_dim, affine=False),
                                        nn.ReLU(),
                                        nn.Conv2d(output_dim, output_dim, 3, 1, 1, 1))
        
        self.decode_8x = nn.Sequential(nn.Conv2d(block_dims[1] + output_dim, output_dim, 1, 1, 0),
                                        nn.InstanceNorm2d(output_dim, affine=False),
                                        nn.ReLU(),
                                        nn.Conv2d(output_dim, output_dim, 3, 1, 1, 1))
        
        self.decode_4x = nn.Sequential(nn.Conv2d(block_dims[0] + output_dim, output_dim, 1, 1, 0),
                                        nn.InstanceNorm2d(output_dim, affine=False),
                                        nn.ReLU(),
                                        nn.Conv2d(output_dim, output_dim, 3, 1, 1, 1))
                 
                 
    def forward(self, x):
        with torch.no_grad():
            x4, x8, x16, x32 = self.convnext(x)
        
        x16 = self.decode_16x(torch.cat([x16, self.upconv_16(x32)], dim=1))
        x8 = self.decode_8x(torch.cat([x8, self.upconv_8(x16)], dim=1))
        x4 = self.decode_4x(torch.cat([x4, self.upconv_4(x8)], dim=1))
        
        return x4, x8, x16


# class Feature(SubModule):
#     def __init__(self):
#         super(Feature, self).__init__()
#         model = timm.create_model('mobilenetv2_100', pretrained=True, features_only=True)
        
#         layers = [1,2,3,5,6]
#         chans = [16, 24, 32, 96, 160]
#         self.conv_stem = model.conv_stem
#         self.bn1 = model.bn1
#         self.act1 = model.act1

#         self.block0 = torch.nn.Sequential(*model.blocks[0:layers[0]])
#         self.block1 = torch.nn.Sequential(*model.blocks[layers[0]:layers[1]])
#         self.block2 = torch.nn.Sequential(*model.blocks[layers[1]:layers[2]])
#         self.block3 = torch.nn.Sequential(*model.blocks[layers[2]:layers[3]])
#         self.block4 = torch.nn.Sequential(*model.blocks[layers[3]:layers[4]])

#         self.deconv32_16 = Conv2x_IN(chans[4], chans[3], deconv=True, concat=True)
#         self.deconv16_8 = Conv2x_IN(chans[3]*2, chans[2], deconv=True, concat=True)
#         self.deconv8_4 = Conv2x_IN(chans[2]*2, chans[1], deconv=True, concat=True)
#         self.conv4 = BasicConv_IN(chans[1]*2, chans[1]*2, kernel_size=3, stride=1, padding=1)

#         self.stem_2 = nn.Sequential(
#             BasicConv(3, 32, kernel_size=3, stride=2, padding=1),
#             nn.Conv2d(32, 32, 3, 1, 1, bias=False),
#             nn.BatchNorm2d(32), nn.ReLU()
#             )
        
#         self.stem_4 = nn.Sequential(
#             BasicConv(32, 64, kernel_size=3, stride=2, padding=1),
#             nn.Conv2d(64, 64, 3, 1, 1, bias=False),
#             nn.BatchNorm2d(64), nn.ReLU()
#             )
        
        # self.conv = BasicConv(112, 112, kernel_size=3, padding=1, stride=1)
        # self.desc = nn.Conv2d(112, 112, kernel_size=1, padding=0, stride=1)
#     def forward(self, x):
#         stem_2 = self.stem_2(x)
#         stem_4 = self.stem_4(stem_2)

#         x = self.act1(self.bn1(self.conv_stem(x)))
#         x2 = self.block0(x)
#         x4 = self.block1(x2)
#         x8 = self.block2(x4)
#         x16 = self.block3(x8)
#         x32 = self.block4(x16)

#         x16 = self.deconv32_16(x32, x16)
#         x8 = self.deconv16_8(x16, x8)
#         x4 = self.deconv8_4(x8, x4)
#         x4 = self.conv4(x4)

#         x4 = torch.cat((x4, stem_4), 1)
#         x4 = self.desc(self.conv(x4))

#         return [x8, x4]


# class feature_backbone(nn.Module):
#     def __init__(self, m=0.999):
#         super(feature_backbone, self).__init__() 
#         self.m = m
#         print('m:{}'.format(m))

#         self.decoder_q = Feature()
#         self.decoder_k = Feature()

#         for param_q, param_k in zip(self.decoder_q.parameters(), self.decoder_k.parameters()):
#             param_k.data.copy_(param_q.data)  # initialize
#             param_k.requires_grad = False  # not update by gradient

#     @torch.no_grad()
#     def _momentum_update_key_encoder(self):
#         """
#         Momentum update of the key encoder
#         """
#         for param_q, param_k in zip(self.decoder_q.parameters(), self.decoder_k.parameters()):
#             param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

#     def forward(self, left, right):
#         l_fms = self.decoder_q(left)

#         if self.training:
#             with torch.no_grad():                    # no gradient to keys
#                 self._momentum_update_key_encoder()  # update the key encoder
#                 r_fms = self.decoder_k(right)
#                 r_fms = [r_fm.detach() for r_fm in r_fms]
#         else:
#             r_fms = self.decoder_q(right)

#         return  l_fms, r_fms