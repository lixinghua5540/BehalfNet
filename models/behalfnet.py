import os
import time
import torch
import torch.nn as nn
from thop import profile
import torch.nn.functional as F
from torchsummaryX import summary
from einops.layers.torch import Reduce
from models.modules.PAF import PAF
from models.modules.GSSA import GSSA
from models.modules.BTSCA import BTSCA

# Channel Calibration
class ChannelCalibration(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.conv = nn.Conv2d(dim_in, dim_out, 1)
        self.bn = nn.BatchNorm2d(dim_out)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

# Group In Group Convolution
class GIGC(nn.Module):
    def __init__(self, dim_in, dim_out, padding=1, num_groups=8):
        super().__init__()
        self.gpwc = nn.Conv2d(dim_in, dim_out, groups=num_groups, kernel_size=1)
        self.gc = nn.Conv2d(dim_out, dim_out, kernel_size=3, groups=num_groups, padding=padding, stride=1)
        self.bn = nn.BatchNorm2d(dim_out)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.gc(self.gpwc(x))))

class BehalfNet(nn.Module):
    def __init__(
            self,
            embed_dim=512,
            num_groups=8,
            num_classes=12,
            channels=[126, 124],
            dim=[256, 128, 64],
            ndiff=[64, 32, 16]
    ):
        super(BehalfNet, self).__init__()
        self.deep = len(dim)

        # Channel Calibration for input
        self.cc_x1 = ChannelCalibration(channels[0], dim[0])
        self.cc_x2 = ChannelCalibration(channels[1], dim[0])

        # The structure of the MyNet network.
        # First: x1 and x2 Pass PAF respectively
        self.PAF_x1 = nn.ModuleList([PAF(dim[i], ndiff[i]) for i in range(self.deep)])
        self.PAF_x2 = nn.ModuleList([PAF(dim[i], ndiff[i]) for i in range(self.deep)])

        # Second: x1 and x2 Pass GSSA Block together
        self.GSSA_blocks = nn.ModuleList([GSSA(dim[i], reduction=4) for i in range(self.deep)])

        # Third: x1 and x2 Pass BTSCA Block together
        self.BTSCA_blocks = nn.ModuleList([
            BTSCA(dim[i], num_heads=1, qkv_bias=False, qk_scale=None, attn_drop=0.1, proj_drop=0.1)
            for i in range(self.deep)
        ])

        # Fourth: If more than one layer, x1 and x2 Pass GIGC between layers respectively
        self.GIGC_blocks_x1 = nn.ModuleList([
            GIGC(dim[i], dim[i + 1], num_groups=num_groups) for i in range(self.deep - 1)
        ])
        self.GIGC_blocks_x2 = nn.ModuleList([
            GIGC(dim[i], dim[i + 1], num_groups=num_groups) for i in range(self.deep - 1)
        ])

        # Hyperspectral Image Classification Head
        self.cc_blocks = nn.ModuleList([ChannelCalibration(dim[i], embed_dim) for i in range(self.deep)])
        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels=embed_dim * self.deep, out_channels=embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )
        self.pred = nn.Sequential(
            Reduce('b d h w -> b d', 'mean'),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x1, x2):
        x1 = x1.squeeze(dim=1)
        x2 = x2.squeeze(dim=1)

        x1 = self.cc_x1(x1)
        x2 = self.cc_x2(x2)

        fused_features = []
        for i in range(self.deep):
            x1 = self.PAF_x1[i](x1)
            x2 = self.PAF_x2[i](x2)

            x1, x2 = self.GSSA_blocks[i](x1, x2)
            fused = self.BTSCA_blocks[i](x1, x2)

            if i < self.deep - 1:
                x1 = self.GIGC_blocks_x1[i](x1)
                x2 = self.GIGC_blocks_x2[i](x2)

            fused_features.append(self.cc_blocks[i](fused))

        x_fuse = self.fuse(torch.cat(fused_features, dim=1))

        x_out = self.pred(x_fuse)

        return x_out

def behalfnet(dataset):
    model = None

    # Anji Dataset
    # Only T1
    if dataset == 'T1':
        model = BehalfNet(
            num_groups = 8,
            num_classes = 12,
            channels = [126, 126],
            dim = [256, 128, 64],
            ndiff = [64, 32, 16]
        )

    # Only T2
    if dataset == 'T2':
        model = BehalfNet(
            num_groups = 8,
            num_classes = 12,
            channels = [124, 124],
            dim = [256, 128, 64],
            ndiff = [64, 32, 16]
        )

    # Cat T1 and T2
    if dataset == 'Bi-Temporal-cat':
        model = BehalfNet(
            num_groups = 8,
            num_classes = 12,
            channels = [250, 250],
            dim = [256, 128, 64],
            ndiff = [64, 32, 16]
        )

    # Split T1 and T2
    if dataset == 'Bi-Temporal-split':
        model = BehalfNet(
            num_groups = 8,
            num_classes = 12,
            channels = [126, 124],
            dim = [256, 128, 64],
            ndiff = [64, 32, 16]
        )

    # Viareggio Dataset
    # Only T1
    if dataset == 'T1_V':
        model = BehalfNet(
            num_groups = 8,
            num_classes = 8,
            channels = [127, 127],
            dim = [256, 128, 64],
            ndiff = [64, 32, 16]
        )

    # Only T2
    if dataset == 'T2_V':
        model = BehalfNet(
            num_groups = 8,
            num_classes = 8,
            channels = [127, 127],
            dim = [256, 128, 64],
            ndiff = [64, 32, 16]
        )

    # Cat T1 and T2
    if dataset == 'Bi-Temporal_V-cat':
        model = BehalfNet(
            num_groups = 8,
            num_classes = 8,
            channels = [254, 254],
            dim = [256, 128, 64],
            ndiff = [64, 32, 16]
        )

    # Split T1 and T2
    if dataset == 'Bi-Temporal_V-split':
        model = BehalfNet(
            num_groups = 8,
            num_classes = 8,
            channels = [127, 127],
            dim = [256, 128, 64],
            ndiff = [64, 32, 16]
        )

    # when N = 5:
    # dim = [512, 384, 256, 128, 64],
    # ndiff = [128, 96, 64, 32, 16]

    return model

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda")

    x1 = torch.randn(1, 126, 16, 16).to(device)
    x2 = torch.randn(1, 124, 16, 16).to(device)
    print("input shape:", x1.shape, x2.shape)
    net = behalfnet(dataset='Bi-Temporal-split').to(device)
    x = net(x1, x2)
    print("output shape:", x.shape)
    summary(net, torch.randn(1, 126, 16, 16).to(device), torch.zeros((1, 124, 16, 16)).to(device))
    flops, params = profile(net, inputs=(x1, x2))
    print(f'params: {params / 1e6:.4f} M')
    print(f'flops:  {flops / 1e9:.4f} G')

    print("Testing speed...")
    with torch.no_grad():
        for _ in range(50): net(x1, x2)

        t_start = time.time()

        for _ in range(1000): net(x1, x2)

        elapsed = time.time() - t_start

    print(f'Inference Time: {elapsed / 1000 * 1000:.4f} ms')
    print(f'FPS: {1000 / elapsed:.2f}')