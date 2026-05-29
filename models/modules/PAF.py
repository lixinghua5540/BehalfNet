import torch
import torch.nn as nn
from thop import profile
from torchsummaryX import summary

# Fusion Block
class FusionBlock(nn.Module):
    def __init__(self, channels, emb_dim):
        super(FusionBlock, self).__init__()
        self.act = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(channels, emb_dim, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(emb_dim)
        self.conv2 = nn.Conv2d(emb_dim, emb_dim, kernel_size=3, padding=1, groups=emb_dim, bias=False)
        self.bn3 = nn.BatchNorm2d(3 * emb_dim)
        self.conv3 = nn.Conv2d(3 * emb_dim, channels, kernel_size=1, bias=False)

    def forward(self, x):
        x0 = self.conv1(x)
        x1 = self.conv2(self.act(self.bn2(x0)))
        x2 = torch.cat([x0, x1, x0], axis=1)
        x = self.conv3(self.act(self.bn3(x2))) + x
        return x

# Combination Coefficient
class CC(nn.Module):
    def __init__(self, channel, output_size=1, reduction=16):
        super(CC, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size)
        self.conv_mean = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, kernel_size=1, padding=0, bias=True),
            nn.Sigmoid(),
        )
        self.conv_std = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, kernel_size=1, padding=0, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # mean
        x_avg = self.avg_pool(x)
        x_mean = self.conv_mean(x_avg)

        # std
        batch, channel, height, width = x.size()
        x = x.view(batch, channel, -1)
        x_std = torch.std(x, dim=2, keepdim=True)
        x_std = x_std.view(batch, channel, 1, 1)
        x_std = self.conv_std(x_std)

        # cc
        cc = (x_mean + x_std)/2.0
        return cc

# Progressive Adaptive Fusion
class PAF(nn.Module):
    def __init__(self, nDim=64, nDiff=16):
        super(PAF, self).__init__()

        self.block_shot = FusionBlock(nDim, nDim + nDiff)

        self.ca1 = CC(nDim)
        self.cb1 = CC(nDim)

        self.block_long = FusionBlock(nDim, nDim + nDiff)

        self.ca2 = CC(nDim)
        self.cb2 = CC(nDim)

        self.conv_out = nn.Sequential(
            nn.Conv2d(2 * nDim, nDim, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(nDim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # shot
        x_shot = self.block_shot(x)
        a1 = self.ca1(x_shot)
        b1 = self.cb1(x)

        p1x = x + a1 * x_shot
        q1x = x_shot + b1 * x

        # long
        x_long = self.block_long(p1x)
        a2 = self.ca2(q1x)
        b2 = self.cb2(x_long)

        p2x = x_long + a2 * q1x
        q2x = q1x + b2 * x_long

        # output
        x = torch.cat((p2x, q2x), dim=1)

        # Ablation Experiment LF
        # x = torch.cat((p1x, q1x), dim=1)

        out = self.conv_out(x)

        return out

if __name__ == '__main__':
    x = torch.randn(1, 64, 8, 8)
    print("input shape:", x.shape)
    block = PAF()
    x = block(x)
    print("output shape:", x.shape)
    summary(block, torch.zeros((1, 64, 8, 8)))
    flops, params = profile(block, inputs=(x,))
    print('params', params)
    print('flops', flops)
