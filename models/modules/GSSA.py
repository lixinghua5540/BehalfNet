import torch
import torch.nn as nn
from thop import profile
from torchsummaryX import summary

# Spectral Attention
class SpectralAttention(nn.Module):
    def __init__(self, dim, reduction=4):
        super(SpectralAttention, self).__init__()
        self.dim = dim
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(self.dim * 2, self.dim * 2 // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(self.dim * 2 // reduction, self.dim * 2))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):

        B, _, H, W = x1.shape

        avg_v1 = self.avg_pool(x1).view(B, self.dim) # B  C
        avg_v2 = self.avg_pool(x2).view(B, self.dim) # B  C
        max_v1 = self.max_pool(x1).view(B, self.dim) # B  C
        max_v2 = self.max_pool(x2).view(B, self.dim) # B  C

        avg_v = torch.cat((avg_v1, avg_v2), dim=1) # B  2*C
        max_v = torch.cat((max_v1, max_v2), dim=1) # B  2*C

        avg_se = self.mlp(avg_v).view(B, self.dim * 2, 1) # B  2*C 1
        max_se = self.mlp(max_v).view(B, self.dim * 2, 1) # B  2*C 1

        Stat_out = self.sigmoid(avg_se + max_se).view(B, self.dim * 2, 1) # B 2*C 1
        spectral_weights = Stat_out.reshape(B, 2, self.dim, 1, 1).permute(1, 0, 2, 3, 4)  # 2 B C 1 1

        return spectral_weights

# Spatial Attention
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=1, reduction=4):
        super(SpatialAttention, self).__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(6, 4 * reduction, kernel_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(4 * reduction, 2, kernel_size),
            nn.Sigmoid())

    def forward(self, x1, x2):
        device = x1.device
        B, _, H, W = x1.shape

        mean_v1 = torch.mean(x1, dim=1, keepdim=True)  # B  1  H  W
        max_v1, _ = torch.max(x1, dim=1, keepdim=True)  # B  1  H  W
        conv_v1 = nn.Conv2d(x1.size(1), 1, kernel_size=3, padding=1).to(device)(x1)

        mean_v2= torch.mean(x2, dim=1, keepdim=True)  # B  1  H  W
        max_v2, _ = torch.max(x2, dim=1, keepdim=True)  # B  1  H  W
        conv_v2 = nn.Conv2d(x2.size(1), 1, kernel_size=3, padding=1).to(device)(x2)

        x_cat = torch.cat((mean_v1, max_v1, conv_v1, mean_v2, max_v2, conv_v2), dim=1)  # B 6 H W
        spatial_weights = self.mlp(x_cat).reshape(B, 2, 1, H, W).permute(1, 0, 2, 3, 4)  # 2 B 1 H W

        return spatial_weights

# Attention Interaction
class AttentionInteraction(nn.Module):
    def __init__(self, dim, reduction=1):
        super(AttentionInteraction, self).__init__()
        self.dim = dim
        self.spectral_gate = SpectralAttention(self.dim)
        self.spatial_gate = SpatialAttention(reduction=4)

    def forward(self, x1, x2):
        spectral_out = self.spectral_gate(x1, x2)  # 2 B C 1 1
        spatial_out = self.spatial_gate(x1, x2)  # 2 B 1 H W

        mix_out = spectral_out.mul(spatial_out)

        return mix_out

# Gate Spatial-Spectral Attention
class GSSA(nn.Module):
    def __init__(self, dim, reduction=4):
        super(GSSA, self).__init__()
        self.SSA = AttentionInteraction(dim)
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim * 2 // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(dim * 2 // reduction, dim),
            nn.Sigmoid())

    def forward(self, x1, x2):
        B1, C1, H1, W1 = x1.shape
        B2, C2, H2, W2 = x2.shape
        assert B1 == B2 and C1 == C2 and H1 == H2 and W1 == W2, "x1 and x2 should have the same dimensions"

        mix_out = self.SSA(x1, x2)  # 2 B C H W

        x1_flat = x1.flatten(2).transpose(1, 2)  # B HXW C
        x2_flat = x2.flatten(2).transpose(1, 2)  # B HXW C

        gated_weight = self.gate(torch.cat((x1_flat, x2_flat), dim=2))  # B HXW C
        gated_weight = gated_weight.reshape(B1, H1, W1, C1).permute(0, 3, 1, 2).contiguous()  # B C H W

        GA_x1 = gated_weight * mix_out[0]
        GA_x2 = (1 - gated_weight) * mix_out[1]

        out_x1 = x2 + GA_x1 * x1  # B C H W
        out_x2 = x1 + GA_x2 * x2  # B C H W

        # Ablation Experiment GM
        # out_x1 = x2 + mix_out[0] * x1  # B C H W
        # out_x2 = x1 + mix_out[1] * x2  # B C H W

        return out_x1, out_x2


if __name__ == '__main__':
    x1 = torch.randn(1, 64, 8, 8)
    x2 = torch.randn(1, 64, 8, 8)
    print("input shape:", x1.shape, x2.shape)
    gssa = GSSA(64)
    x1, x2 = gssa(x1, x2)
    print("output shape:", x1.shape, x2.shape)
    summary(gssa, torch.zeros((1, 64, 8, 8)), torch.zeros((1, 64, 8, 8)))
    flops, params = profile(gssa, inputs=(x1, x2))
    print('params', params)
    print('flops', flops)
