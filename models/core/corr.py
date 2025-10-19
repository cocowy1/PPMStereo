# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F

def bilinear_sampler(img, coords, mode="bilinear", mask=False, stereo=True):
    """Wrapper for grid_sample, uses pixel coordinates"""
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1, 1], dim=-1)
    xgrid = 2 * xgrid / (W - 1) - 1
    if not stereo:
        ygrid = 2 * ygrid / (H - 1) - 1
    else:
        assert torch.unique(ygrid).numel() == 1 and H == 1  # This is a stereo problem
    img = img.contiguous().float()
    grid = torch.cat([xgrid, ygrid], dim=-1).contiguous().float()
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img


def bilinear_sampler_bidastereo(img, coords, mode='bilinear', mask=False):
    """Wrapper for grid_sample, uses pixel coordinates"""
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1, 1], dim=-1)
    xgrid = 2 * xgrid / (W - 1) - 1
    if H > 1:
        ygrid = 2 * ygrid/(H - 1) - 1
    img = img.contiguous()
    grid = torch.cat([xgrid, ygrid], dim=-1).contiguous()
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img

def coords_grid(batch, ht, wd, device):
    coords = torch.meshgrid(
        torch.arange(ht, device=device), torch.arange(wd, device=device), indexing="ij"
    )
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


class CorrBlock1D:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []
        self.coords = coords_grid(
            fmap1.shape[0], fmap1.shape[2], fmap1.shape[3], fmap1.device
        )
        # all pairs correlation
        corr = CorrBlock1D.corr(fmap1, fmap2)

        batch, h1, w1, dim, w2 = corr.shape
        corr = corr.reshape(batch * h1 * w1, dim, 1, w2)

        self.corr_pyramid.append(corr)
        for i in range(self.num_levels):
            corr = F.avg_pool2d(corr, [1, 2], stride=[1, 2])
            self.corr_pyramid.append(corr)

    def __call__(self, flow):
        r = self.radius
        coords = self.coords + flow
        coords = coords[:, :1].permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            dx = torch.linspace(-r, r, 2 * r + 1)
            dx = dx.view(1, 1, 2 * r + 1, 1).to(coords.device)
            x0 = dx + coords.reshape(batch * h1 * w1, 1, 1, 1) / 2 ** i
            y0 = torch.zeros_like(x0)

            coords_lvl = torch.cat([x0, y0], dim=-1)
            corr = bilinear_sampler(corr, coords_lvl)
            corr = corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()

    @staticmethod
    def corr(fmap1, fmap2):
        B, D, H, W1 = fmap1.shape
        _, _, _, W2 = fmap2.shape
        fmap1 = fmap1.view(B, D, H, W1)
        fmap2 = fmap2.view(B, D, H, W2)
        corr = torch.einsum("aijk, aijh->ajkh", fmap1, fmap2)
        corr = corr.reshape(B, H, W1, 1, W2).contiguous()
        return corr / torch.sqrt(torch.tensor(D).float())




class TFCL:
    """
    Implementation of Triple-frame Correlation Layer (TFCL).
    """
    def __init__(self, fmap1, fmap2):
        self.fmap1 = fmap1
        self.fmap2 = fmap2
        self.coords = coords_grid(fmap1.shape[0], fmap1.shape[2], fmap1.shape[3], fmap1.device)

    def __call__(self, flow, extra_offset, small_patch=False):

        corr = self.correlation(self.fmap1, self.fmap2, flow, small_patch)

        return corr

    def correlation(self, left_feature, right_feature, flow, small_patch):
        flow[:, 1:] = 0
        coords = self.coords + flow
        coords = coords.permute(0, 2, 3, 1)
        coords = coords.repeat(3,1,1,1)
        right_feature = bilinear_sampler_bidastereo(right_feature, coords)

        if small_patch:
            psize_list = [(3, 3), (3, 3), (3, 3), (3, 3)]
            dilate_list = [(1, 1), (1, 1), (1, 1), (1, 1)]
        else:
            psize_list = [(1, 9), (1, 9), (1, 9), (1, 9)]
            dilate_list = [(1, 1), (1, 1), (1, 1), (1, 1)]

        N, C, H, W = right_feature.size()
        rights = torch.split(right_feature, [N // 3] * 3, dim=0)
        corrs = []
        for i in range(3):
            corr = self.get_correlation(left_feature, rights[i], psize_list[i], dilate_list[i])
            corrs.append(corr)

        final_corr = torch.cat(corrs, dim=1)

        return final_corr

    def get_correlation(self, left_feature, right_feature, psize=(3, 3), dilate=(1, 1)):

        N, C, H, W = left_feature.size()

        di_y, di_x = dilate[0], dilate[1]
        pady, padx = psize[0] // 2 * di_y, psize[1] // 2 * di_x

        right_pad = F.pad(right_feature, [padx, padx, pady, pady], mode='replicate')

        corr_list = []
        for h in range(0, pady * 2 + 1, di_y):
            for w in range(0, padx * 2 + 1, di_x):
                right_crop = right_pad[:, :, h:h + H, w:w + W]
                assert right_crop.size() == left_feature.size()
                corr = (left_feature * right_crop).mean(dim=1, keepdim=True)
                corr_list.append(corr)

        corr_final = torch.cat(corr_list, dim=1)

        return corr_final
    


class AAPC:
    """
    Implementation of All-in-All-Pair Correlation.
    """
    def __init__(self, fmap1, fmap2, att=None):
        self.fmap1 = fmap1
        self.fmap2 = fmap2

        self.att = att
        self.coords = coords_grid(fmap1.shape[0], fmap1.shape[2], fmap1.shape[3], fmap1.device)

    def __call__(self, flow, extra_offset, small_patch=False):

        corr = self.correlation(self.fmap1, self.fmap2, flow, small_patch)

        return corr

    def correlation(self, left_feature, right_feature, flow, small_patch):
        flow[:, 1:] = 0
        coords = self.coords - flow
        coords = coords.permute(0, 2, 3, 1)
        right_feature = bilinear_sampler_bidastereo(right_feature, coords)

        if small_patch:
            psize_list = [(3, 3), (3, 3), (3, 3), (3, 3)]
            dilate_list = [(1, 1), (1, 1), (1, 1), (1, 1)]
        else:
            psize_list = [(1, 9), (1, 9), (1, 9), (1, 9)]
            dilate_list = [(1, 1), (1, 1), (1, 1), (1, 1)]

        N, C, H, W = left_feature.size()
        lefts = torch.split(left_feature, [C // 4] * 4, dim=1)
        rights = torch.split(right_feature, [C // 4] * 4, dim=1)
        corrs = []
        for i in range(len(psize_list)):
            corr = self.get_correlation(lefts[i], rights[i], psize_list[i], dilate_list[i])
            corrs.append(corr)

        final_corr = torch.cat(corrs, dim=1)
        return final_corr

    def get_correlation(self, left_feature, right_feature, psize=(3, 3), dilate=(1, 1)):

        N, C, H, W = left_feature.size()

        di_y, di_x = dilate[0], dilate[1]
        pady, padx = psize[0] // 2 * di_y, psize[1] // 2 * di_x

        left_pad = F.pad(left_feature, [padx, padx, pady, pady], mode='replicate')
        right_pad = F.pad(right_feature, [padx, padx, pady, pady], mode='replicate')

        corr_list = []
        for dy1 in range(0, pady * 2 + 1, di_y):
            for dx1 in range(0, padx * 2 + 1, di_x):
                left_crop = left_pad[:, :, dy1:dy1 + H, dx1:dx1 + W]

                for dy2 in range(0, pady * 2 + 1, di_y):
                    for dx2 in range(0, padx * 2 + 1, di_x):
                        right_crop = right_pad[:, :, dy2:dy2 + H, dx2:dx2 + W]
                        assert right_crop.size() == left_crop.size()
                        corr = (left_crop * right_crop).sum(dim=1, keepdim=True)  # Sum over channels
                        corr_list.append(corr)

        corr_final = torch.cat(corr_list, dim=1)

        return corr_final