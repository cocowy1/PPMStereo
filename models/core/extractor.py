# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1, norm_layer=nn.BatchNorm2d):
        super().__init__()

        # self.sparse = sparse
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.conv2 = conv3x3(planes, planes)
        self.bn1 = norm_layer(planes)
        self.bn2 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        if stride == 1 and in_planes == planes:
            self.downsample = None
        else:
            self.bn3 = norm_layer(planes)
            self.downsample = nn.Sequential(
                conv1x1(in_planes, planes, stride=stride),
                self.bn3
            )

    def forward(self, x):
        y = x
        y = self.relu(self.bn1(self.conv1(y)))
        y = self.relu(self.bn2(self.conv2(y)))
        if self.downsample is not None:
            x = self.downsample(x)
        return self.relu(x+y)
    
    
def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution without padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1)



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
                
class Feature(SubModule):
    def __init__(self, output_dim=256):
        super(Feature, self).__init__()
        # if model_name == 'base':
        #     self.model = convnextv2_base()
        # elif model_name == 'large':
        #     self.model = convnextv2_large()
        # elif model_name == 'huge':
        #     self.model = convnextv2_huge()
        # elif model_name == 'tiny':
        #     self.model = convnextv2_tiny()
        # else:
        #     raise NotImplementedError
        block_dims = [96, 192, 384]
        self.convnext = timm.create_model('convnext_tiny_in22k', pretrained=True)
        self.convnext.eval()
        self.decode_16 = conv3x3(block_dims[2], output_dim) # 1/16
        self.decode_8 = conv3x3(block_dims[1], output_dim) # 1/8
        self.decode_4 = conv3x3(block_dims[0], output_dim) # 1/4
                    
    def forward(self, x):
        with torch.no_grad():
            x4, x8, x16, _ = self.convnext(x)
            
        x4 = self.decode_4(x4)
        x8 = self.decode_8(x8)
        x16 = self.decode_16(x16)
            
        return x4, x8, x16




class ResNetFPN(nn.Module):
    """
    ResNet18, output resolution is 1/16.
    Each block has 2 layers.
    """
    def __init__(self, input_dim=3, output_dim=256, 
                 ratio=1, norm_layer=nn.BatchNorm2d, init_weight=False):
        super().__init__()
        # Config
        block = BasicBlock
        block_dims = [64, 128, 256]
        initial_dim = 64
        self.init_weight = init_weight
        self.input_dim = input_dim
        # Class Variable
        self.in_planes = initial_dim
        for i in range(len(block_dims)):
            block_dims[i] = int(block_dims[i] * ratio)
            
        # Networks
        self.conv1 = nn.Conv2d(input_dim, initial_dim, kernel_size=7, stride=2, padding=3)
        self.bn1 = norm_layer(initial_dim)
        self.relu = nn.ReLU(inplace=True)
        self.name = 'resnet18'
        if self.name == 'resnet34':
            n_block = [3, 4, 6]
        elif self.name == 'resnet18':
            n_block = [2, 2, 2]
        else:
            raise NotImplementedError       
        
        self.layer1 = self._make_layer(block, block_dims[0], stride=1, norm_layer=norm_layer, num=n_block[0])  # 1/2
        self.layer2 = self._make_layer(block, block_dims[1], stride=2, norm_layer=norm_layer, num=n_block[1])  # 1/4
        self.layer3 = self._make_layer(block, block_dims[2], stride=1, norm_layer=norm_layer, num=n_block[2])  # 1/8

        self.final_conv = conv3x3(block_dims[2], output_dim)
        self._init_weights(self.name)

    def _init_weights(self, name):

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        if self.init_weight:
            from torchvision.models import resnet18, ResNet18_Weights, resnet34, ResNet34_Weights
            if name == 'resnet18':
                pretrained_dict = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).state_dict()
            else:
                pretrained_dict = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1).state_dict()
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            # if self.input_dim == 6:
            #     for k, v in pretrained_dict.items():
            #         if k == 'conv1.weight':
            #             pretrained_dict[k] = torch.cat((v, v), dim=1)
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict, strict=False)
        

    def _make_layer(self, block, dim, stride=1, norm_layer=nn.BatchNorm2d, num=2):
        layers = []
        layers.append(block(self.in_planes, dim, stride=stride, norm_layer=norm_layer))
        for i in range(num - 1):
            layers.append(block(dim, dim, stride=1, norm_layer=norm_layer))
        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        # ResNet Backbone
        x = self.relu(self.bn1(self.conv1(x)))
        for i in range(len(self.layer1)):
            x = self.layer1[i](x)
        for i in range(len(self.layer2)):
            x = self.layer2[i](x)
        for i in range(len(self.layer3)):
            x = self.layer3[i](x)
            
        # # Output
        output = self.final_conv(x)
        
        return output

class MultiLevelResNetFPN(nn.Module):
    """
    ResNet18, output resolution is 1/16.
    Each block has 2 layers.
    """
    def __init__(self, input_dim=3, output_dim=256, 
                 ratio=1, norm_layer=nn.BatchNorm2d, init_weight=False):
        super().__init__()
        # Config
        block = BasicBlock
        block_dims = [64, 128, 256, 512]
        initial_dim = 64
        self.init_weight = init_weight
        self.input_dim = input_dim
        # Class Variable
        self.in_planes = initial_dim
        for i in range(len(block_dims)):
            block_dims[i] = int(block_dims[i] * ratio)
            
        # Networks
        self.conv1 = nn.Conv2d(input_dim, initial_dim, kernel_size=7, stride=2, padding=3)
        self.bn1 = norm_layer(initial_dim)
        self.relu = nn.ReLU(inplace=True)
        self.name = 'resnet18'
        if self.name == 'resnet34':
            n_block = [3, 4, 6]
        elif self.name == 'resnet18':
            n_block = [2, 2, 2, 2]
        else:
            raise NotImplementedError       
        
        self.layer1 = self._make_layer(block, block_dims[0], stride=1, norm_layer=norm_layer, num=n_block[0])  # 1/2
        self.layer2 = self._make_layer(block, block_dims[1], stride=2, norm_layer=norm_layer, num=n_block[1])  # 1/4
        self.layer3 = self._make_layer(block, block_dims[2], stride=2, norm_layer=norm_layer, num=n_block[2])  # 1/8
        self.layer4 = self._make_layer(block, block_dims[3], stride=2, norm_layer=norm_layer, num=n_block[3])  # 1/16

        self.decode_16 = conv3x3(block_dims[3], output_dim) # 1/16
        self.decode_8 = conv3x3(block_dims[2], output_dim) # 1/8
        self.decode_4 = conv3x3(block_dims[1], output_dim) # 1/4
        
        self._init_weights(self.name)

    def _init_weights(self, name):

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        if self.init_weight:
            from torchvision.models import resnet18, ResNet18_Weights, resnet34, ResNet34_Weights
            if name == 'resnet18':
                pretrained_dict = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).state_dict()
            else:
                pretrained_dict = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1).state_dict()
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            # if self.input_dim == 6:
            #     for k, v in pretrained_dict.items():
            #         if k == 'conv1.weight':
            #             pretrained_dict[k] = torch.cat((v, v), dim=1)
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict, strict=False)
        

    def _make_layer(self, block, dim, stride=1, norm_layer=nn.BatchNorm2d, num=2):
        layers = []
        layers.append(block(self.in_planes, dim, stride=stride, norm_layer=norm_layer))
        for i in range(num - 1):
            layers.append(block(dim, dim, stride=1, norm_layer=norm_layer))
        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        
        # ResNet Backbone
        x = self.relu(self.bn1(self.conv1(x)))
        for i in range(len(self.layer1)):
            x = self.layer1[i](x)
        
        for i in range(len(self.layer2)):
            x = self.layer2[i](x)
            
        output_4 = self.decode_4(x)
            
        for i in range(len(self.layer3)):
            x = self.layer3[i](x)

        output_8 = self.decode_8(x)
        
        for i in range(len(self.layer4)):
            x = self.layer4[i](x)

        output_16 = self.decode_16(x)
      
        return output_4, output_8, output_16
    
       
    
class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn="group", stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, padding=1, stride=stride
        )
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == "group":
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)

        elif norm_fn == "batch":
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            self.norm3 = nn.BatchNorm2d(planes)

        elif norm_fn == "instance":
            self.norm1 = nn.InstanceNorm2d(planes, affine=False)
            self.norm2 = nn.InstanceNorm2d(planes, affine=False)
            self.norm3 = nn.InstanceNorm2d(planes, affine=False)

        elif norm_fn == "none":
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            self.norm3 = nn.Sequential()

        self.downsample = nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3
        )

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        x = self.downsample(x)

        return self.relu(x + y)


class BasicEncoder(nn.Module):
    def __init__(self, output_dim=128, norm_fn="batch", dropout=0.0):
        super(BasicEncoder, self).__init__()
        self.norm_fn = norm_fn

        if self.norm_fn == "group":
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)

        elif self.norm_fn == "batch":
            self.norm1 = nn.BatchNorm2d(64)

        elif self.norm_fn == "instance":
            self.norm1 = nn.InstanceNorm2d(64, affine=False)

        elif self.norm_fn == "none":
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 64
        self.layer1 = self._make_layer(64, stride=1)
        self.layer2 = self._make_layer(96, stride=2)
        self.layer3 = self._make_layer(128, stride=1)

        # output convolution
        self.conv2 = nn.Conv2d(128, output_dim, kernel_size=1)

        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):

        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)
    
            
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.conv2(x)

        if self.dropout is not None:
            x = self.dropout(x)

        if is_list:
            x = torch.split(x, x.shape[0] // 2, dim=0)

        return x



class BasicEncoder_VFM(nn.Module):
    def __init__(self, output_dim=128, norm_fn="batch", dropout=0.0):
        super(BasicEncoder_VFM, self).__init__()
        self.norm_fn = norm_fn

        if self.norm_fn == "group":
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)

        elif self.norm_fn == "batch":
            self.norm1 = nn.BatchNorm2d(64)

        elif self.norm_fn == "instance":
            self.norm1 = nn.InstanceNorm2d(64, affine=False)

        elif self.norm_fn == "none":
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 64
        self.layer1 = self._make_layer(64, stride=1)
        self.layer2 = self._make_layer(96, stride=2)
        self.layer3 = self._make_layer(128, stride=1)

        # output convolution
        self.conv2 = nn.Conv2d(128 + 768, output_dim, kernel_size=1)

        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x, vfm_features):

        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)
            
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.conv2(torch.cat([x, vfm_features], dim=1))

        if self.dropout is not None:
            x = self.dropout(x)

        if is_list:
            x = torch.split(x, x.shape[0] // 2, dim=0)

        return x

class MultiLevelEncoder_VFM(nn.Module):
    def __init__(self, output_dim=256, norm_fn="batch", dropout=0.0,  downsample=2):
        super(MultiLevelEncoder_VFM, self).__init__()
        self.norm_fn = norm_fn

        if self.norm_fn == "group":
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)

        elif self.norm_fn == "batch":
            self.norm1 = nn.BatchNorm2d(64)

        elif self.norm_fn == "instance":
            self.norm1 = nn.InstanceNorm2d(64, affine=False)

        elif self.norm_fn == "none":
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1 + (downsample > 2), padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 64
        self.layer1 = self._make_layer(64, stride=1)
        self.layer2 = self._make_layer(96, stride=1 + (downsample > 1))
        self.layer3 = self._make_layer(128, stride=1 + (downsample > 0))
        self.layer4 = self._make_layer(128, stride=2)
        self.layer5 = self._make_layer(128, stride=2)

        self.upconv_16 = nn.Sequential(nn.Upsample(scale_factor=2),
                                        nn.Conv2d(64, 64, 3, 1, 1, 1),
                                        nn.InstanceNorm2d(64, affine=False),
                                        nn.ReLU(),)

        self.upconv_8 = nn.Sequential(nn.Upsample(scale_factor=2),
                                        nn.Conv2d(256, 128, 3, 1, 1, 1),
                                        nn.InstanceNorm2d(128, affine=False),
                                        nn.ReLU(),)

        self.upconv_4 = nn.Sequential(nn.Upsample(scale_factor=2),
                                        nn.Conv2d(256, 128, 3, 1, 1, 1),
                                        nn.InstanceNorm2d(128, affine=False),
                                        nn.ReLU(),)


        self.decode_16x = nn.Sequential(nn.Conv2d(128 + 64*2, output_dim, 3, 1, 1, 1),
                                        nn.InstanceNorm2d(output_dim, affine=False),
                                        nn.ReLU(),
                                        nn.Conv2d(output_dim, output_dim, 3, 1, 1, 1),
                                        )
        
        self.decode_8x = nn.Sequential(nn.Conv2d(128*2 + 64, output_dim, 3, 1, 1, 1),
                                        nn.InstanceNorm2d(output_dim, affine=False),
                                        nn.ReLU(),
                                        nn.Conv2d(output_dim, output_dim, 3, 1, 1, 1),)
        
        self.decode_4x = nn.Sequential(nn.Conv2d(128*2 + 64, output_dim, 3, 1, 1, 1),
                                        nn.InstanceNorm2d(output_dim, affine=False),
                                        nn.ReLU(),
                                        nn.Conv2d(output_dim, output_dim, 3, 1, 1, 1),)

        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x, vfm_features):

        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)
            
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)  ## 1

        x = self.layer1(x)    ## 1, 64
        x = self.layer2(x)    ## 1/2, 96
        x_4 = self.layer3(x)    ## 1/4, 128
        
        x_8 = self.layer4(x_4)   ## 1/16, 128
        x_16 = self.layer5(x_8)   ## 1/16, 128
        
        # 16x
        decode_16 = torch.cat((x_16, vfm_features[2], self.upconv_16(vfm_features[3])), dim=1)
        decode_16 = self.decode_16x(decode_16) 

        # 8x
        decode_8 = torch.cat((x_8, vfm_features[1], self.upconv_8(decode_16)), dim=1)
        decode_8 = self.decode_8x(decode_8) 
        
        # 4x
        decode_4 = torch.cat((x_4, vfm_features[0], self.upconv_4(decode_8)), dim=1)
        decode_4 = self.decode_4x(decode_4) 
        
        if self.dropout is not None:
            x = self.dropout(x)

        if is_list:
            fmap1_dw16, fmap2_dw16 = torch.split(decode_16, x.shape[0] // 2, dim=0)
            fmap1_dw8, fmap2_dw8 = torch.split(decode_8, x.shape[0] // 2, dim=0)
            fmap1, fmap2 = torch.split(decode_4, x.shape[0] // 2, dim=0)
            
        return [fmap1, fmap2], [fmap1_dw8, fmap2_dw8], [fmap1_dw16, fmap2_dw16]
    
    
class MultiBasicEncoder(nn.Module):
    def __init__(self, output_dim=[128], norm_fn='batch', dropout=0.0, downsample=2):
        super(MultiBasicEncoder, self).__init__()
        self.norm_fn = norm_fn
        self.downsample = downsample
        
        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)

        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(64)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(64)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1 + (downsample > 2), padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 64
        self.layer1 = self._make_layer(64, stride=1)
        self.layer2 = self._make_layer(96, stride=1 + (downsample > 1))
        self.layer3 = self._make_layer(128, stride=1 + (downsample > 0))
        self.layer4 = self._make_layer(128, stride=2)
        self.layer5 = self._make_layer(128, stride=2)

        output_list = []
        
        for dim in output_dim:
            conv_out = nn.Sequential(
                ResidualBlock(128, 128, self.norm_fn, stride=1),
                nn.Conv2d(128, dim[2], 3, padding=1))
            output_list.append(conv_out)

        self.outputs04 = nn.ModuleList(output_list)

        output_list = []
        for dim in output_dim:
            conv_out = nn.Sequential(
                ResidualBlock(128, 128, self.norm_fn, stride=1),
                nn.Conv2d(128, dim[1], 3, padding=1))
            output_list.append(conv_out)

        self.outputs08 = nn.ModuleList(output_list)

        output_list = []
        for dim in output_dim:
            conv_out = nn.Conv2d(128, dim[0], 3, padding=1)
            output_list.append(conv_out)

        self.outputs16 = nn.ModuleList(output_list)

        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)
        else:
            self.dropout = None

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x, dual_inp=False, num_layers=3):
        
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)
            
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        if dual_inp:
            v = x
            x = x[:(x.shape[0]//2)]

        outputs04 = [f(x) for f in self.outputs04]
        if num_layers == 1:
            return (outputs04, v) if dual_inp else (outputs04,)

        y = self.layer4(x)
        outputs08 = [f(y) for f in self.outputs08]

        if num_layers == 2:
            return (outputs04, outputs08, v) if dual_inp else (outputs04, outputs08)

        z = self.layer5(y)
        outputs16 = [f(z) for f in self.outputs16]


        if is_list:
            fmap1_dw16, fmap2_dw16 = torch.split(outputs16, x.shape[0] // 2, dim=0)
            fmap1_dw8, fmap2_dw8 = torch.split(outputs08, x.shape[0] // 2, dim=0)
            fmap1, fmap2 = torch.split(outputs04, x.shape[0] // 2, dim=0)
            
        return [fmap1_dw16, fmap2_dw16], [fmap1_dw8, fmap2_dw8], [fmap1, fmap2]
    
