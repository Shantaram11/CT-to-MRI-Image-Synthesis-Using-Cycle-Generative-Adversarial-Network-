import torch
import torch.nn as nn


class PatchDiscriminator(nn.Module):
    """
    简化版 PatchGAN 判别器
    输入: [B, 1, 256, 256] 灰度图
    输出: [B, 1, H, W] 的真/假评分图
    """

    def __init__(self, in_channels=1):
        super().__init__()

        def conv_block(in_c, out_c, normalize=True):
            layers = [nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *conv_block(in_channels, 64, normalize=False),
            *conv_block(64, 128),
            *conv_block(128, 256),
            *conv_block(256, 512),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, x):
        return self.model(x)