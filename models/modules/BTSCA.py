import torch
import torch.nn as nn
from thop import profile
from torchsummaryX import summary

# Self Attention
class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,):
        super(SelfAttention, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        # B N C -> B N num_head C//num_head -> B C//num_head N num_heads
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x_atten = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x_out = self.proj(x_atten + x)
        x_out = self.proj_drop(x_out)

        return x_out

# Cross Attention
class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super(CrossAttention, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.q1 = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv2 = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x1, x2):
        B1, N1, C1 = x1.shape
        B2, N2, C2 = x2.shape
        assert B1 == B2 and C1 == C2 and N1 == N2, "x1 and x2 should have the same dimensions"

        # B N C -> B N num_head C//num_head -> B C//num_head N num_heads
        q1 = self.q1(x1).reshape(B1, N1, self.num_heads, C1 // self.num_heads).permute(0, 2, 1, 3)
        kv2 = self.kv2(x2).reshape(B2, -1, 2, self.num_heads, C2 // self.num_heads).permute(2, 0, 3, 1, 4)
        k2, v2 = kv2[0], kv2[1]

        attn = (q1 @ k2.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x_atten = (attn @ v2).transpose(1, 2).reshape(B2, N2, C2)
        x_out = self.proj(x_atten + x1)
        x_out = self.proj_drop(x_out)

        return x_out

# Bi-Temporal Self Cycle Attention
class BTSCA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super(BTSCA, self).__init__()
        self.SA_x1 = SelfAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                   attn_drop=attn_drop, proj_drop=proj_drop)
        self.SA_x2 = SelfAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                   attn_drop=attn_drop,proj_drop=proj_drop)
        self.CA_x1toX2 = CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                        attn_drop=attn_drop, proj_drop=proj_drop)
        self.CA_x2toX1 = CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                        attn_drop=attn_drop, proj_drop=proj_drop)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x1, x2):
        B1, C1, H1, W1 = x1.shape
        B2, C2, H2, W2 = x2.shape
        assert B1 == B2 and C1 == C2 and H1 == H2 and W1 == W2, "x1 and x2 should have the same dimensions"

        x1 = x1.flatten(2).transpose(1, 2)  # B HXW C
        x2 = x2.flatten(2).transpose(1, 2)

        x1_self = self.SA_x1(x1)
        x2_cross = self.CA_x1toX2(x2, x1_self)
        x2_self = self.SA_x2(x2_cross)
        x1_cross = self.CA_x2toX1(x1_self, x2_self)  ##B HXW C
        x = self.proj(x1_cross)  # B HXW C

        # Ablation Experiment CL
        # x = self.proj(x2_cross)  # B HXW C

        x = x.permute(0, 2, 1).reshape(B1, C1, H1, W1).contiguous()

        return x

if __name__ == '__main__':
    x1 = torch.randn(1, 64, 8, 8)
    x2 = torch.randn(1, 64, 8, 8)
    print("input shape:", x1.shape, x2.shape)
    btsca = BTSCA(64)
    x = btsca(x1, x2)
    print("output shape:", x.shape)
    summary(btsca, torch.zeros((1, 64, 8, 8)), torch.zeros((1, 64, 8, 8)))
    flops, params = profile(btsca, inputs=(x1, x2))
    print('params', params)
    print('flops', flops)