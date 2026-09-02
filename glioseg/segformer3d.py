"""SegFormer3D -- lightweight hierarchical transformer for 3D segmentation.

Reference: Perera, Navard & Yilmaz, "SegFormer3D: an Efficient Transformer for
3D Medical Image Segmentation", CVPR 2024 Workshops, arXiv:2404.10156.
Official code: github.com/OSUPCVLab/SegFormer3D

This is a from-scratch reimplementation of the described architecture:
  - overlapping patch embedding at 4 hierarchical stages
  - efficient self-attention with spatial-reduction ratio R per stage
  - Mix-FFN (depthwise 3x3x3 conv inside the FFN, no positional embedding)
  - all-MLP decoder fusing all four stages

It is the ONLY arm not sourced from MONAI, so it is the one to sanity-check
first. Target ~4.5M params at the default width.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_ch, embed_dim, patch=7, stride=4):
        super().__init__()
        self.proj = nn.Conv3d(in_ch, embed_dim, kernel_size=patch,
                              stride=stride, padding=patch // 2)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)                      # B,C,D,H,W
        shape = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)      # B,N,C
        return self.norm(x), shape


class EfficientAttention(nn.Module):
    """Self-attention with spatial reduction on K,V -- the cost saver."""

    def __init__(self, dim, heads, sr_ratio=1):
        super().__init__()
        assert dim % heads == 0, f"dim {dim} not divisible by heads {heads}"
        self.heads, self.scale = heads, (dim // heads) ** -0.5
        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, dim * 2, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv3d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, shape):
        B, N, C = x.shape
        d, h, w = shape
        q = self.q(x).reshape(B, N, self.heads, C // self.heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.transpose(1, 2).reshape(B, C, d, h, w)
            x_ = self.sr(x_).flatten(2).transpose(1, 2)
            x_ = self.norm(x_)
        else:
            x_ = x
        kv = self.kv(x_).reshape(B, -1, 2, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        x = F.scaled_dot_product_attention(q, k, v)          # flash path when available
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class MixFFN(nn.Module):
    """FFN with a depthwise conv -- supplies positional information implicitly,
    which is why SegFormer needs no positional embedding."""

    def __init__(self, dim, expansion=4):
        super().__init__()
        hidden = dim * expansion
        self.fc1 = nn.Linear(dim, hidden)
        self.dw = nn.Conv3d(hidden, hidden, 3, padding=1, groups=hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x, shape):
        B, N, C = x.shape
        d, h, w = shape
        x = self.fc1(x)
        x = self.dw(x.transpose(1, 2).reshape(B, -1, d, h, w)).flatten(2).transpose(1, 2)
        return self.fc2(F.gelu(x))


class Block(nn.Module):
    def __init__(self, dim, heads, sr_ratio, expansion=4):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.attn = EfficientAttention(dim, heads, sr_ratio)
        self.n2 = nn.LayerNorm(dim)
        self.ffn = MixFFN(dim, expansion)

    def forward(self, x, shape):
        x = x + self.attn(self.n1(x), shape)
        x = x + self.ffn(self.n2(x), shape)
        return x


class SegFormer3D(nn.Module):
    def __init__(self, in_channels=4, num_classes=4,
                 dims=(32, 64, 160, 256), heads=(1, 2, 5, 8),
                 depths=(2, 2, 2, 2), sr_ratios=(4, 2, 1, 1),
                 decoder_dim=256):
        super().__init__()
        patches = [(7, 4), (3, 2), (3, 2), (3, 2)]
        self.stages = nn.ModuleList()
        cin = in_channels
        for i, (dim, hd, dp, sr) in enumerate(zip(dims, heads, depths, sr_ratios)):
            p, s = patches[i]
            self.stages.append(nn.ModuleDict({
                "embed": OverlapPatchEmbed(cin, dim, p, s),
                "blocks": nn.ModuleList([Block(dim, hd, sr) for _ in range(dp)]),
                "norm": nn.LayerNorm(dim),
            }))
            cin = dim

        # all-MLP decoder: project every stage to a common width, fuse, predict
        self.linears = nn.ModuleList([nn.Linear(d, decoder_dim) for d in dims])
        self.fuse = nn.Sequential(
            nn.Conv3d(decoder_dim * len(dims), decoder_dim, 1, bias=False),
            nn.BatchNorm3d(decoder_dim), nn.ReLU(inplace=True),
        )
        self.head = nn.Conv3d(decoder_dim, num_classes, 1)

    def forward(self, x):
        size = x.shape[2:]
        feats = []
        for st in self.stages:
            x, shape = st["embed"](x)
            for blk in st["blocks"]:
                x = blk(x, shape)
            x = st["norm"](x)
            B, N, C = x.shape
            x = x.transpose(1, 2).reshape(B, C, *shape)
            feats.append(x)

        ref = feats[0].shape[2:]
        outs = []
        for f, lin in zip(feats, self.linears):
            B, C = f.shape[:2]
            sh = f.shape[2:]
            f = lin(f.flatten(2).transpose(1, 2)).transpose(1, 2).reshape(B, -1, *sh)
            outs.append(F.interpolate(f, size=ref, mode="trilinear", align_corners=False))

        y = self.head(self.fuse(torch.cat(outs, dim=1)))
        return F.interpolate(y, size=size, mode="trilinear", align_corners=False)
