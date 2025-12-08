
import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """
    两次 Conv + IN + ReLU
    （原来是 BatchNorm，这里改成 InstanceNorm，更适合 GAN + batch_size=1）
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    """
    下采样: MaxPool + DoubleConv
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch),
        )

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    """
    上采样: ConvTranspose2d + 拼接(skip) + DoubleConv
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # 注意：in_ch 是拼接后的通道数
        self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        # x1: 来自下方的特征，x2: 来自 skip 的特征
        x1 = self.up(x1)

        # 对齐大小（有时候可能差一像素，这里简单裁剪）
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        x2 = x2[:, :, diff_y // 2: x2.size(2) - diff_y // 2,
                    diff_x // 2: x2.size(3) - diff_x // 2]

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    简单 2D UNet，用于 CT -> MRI 图像翻译
    输入: [B, 1, 256, 256]
    输出: [B, 1, 256, 256]，范围 [-1, 1] (tanh)
    """
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()

        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)

        self.up1 = Up(512 + 512, 256)
        self.up2 = Up(256 + 256, 128)
        self.up3 = Up(128 + 128, 64)
        self.up4 = Up(64 + 64, 64)

        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x1 = self.inc(x)       # [B, 64, 256, 256]
        x2 = self.down1(x1)    # [B, 128, 128, 128]
        x3 = self.down2(x2)    # [B, 256, 64, 64]
        x4 = self.down3(x3)    # [B, 512, 32, 32]
        x5 = self.down4(x4)    # [B, 512, 16, 16]

        x = self.up1(x5, x4)   # [B, 256, 32, 32]
        x = self.up2(x, x3)    # [B, 128, 64, 64]
        x = self.up3(x, x2)    # [B, 64, 128, 128]
        x = self.up4(x, x1)    # [B, 64, 256, 256]

        x = self.outc(x)       # [B, 1, 256, 256]
        x = self.tanh(x)       # [-1, 1]
        return x


if __name__ == "__main__":
    net = UNet(in_channels=1, out_channels=1)
    x = torch.randn(2, 1, 256, 256)
    y = net(x)
    print("输入:", x.shape, "输出:", y.shape)
