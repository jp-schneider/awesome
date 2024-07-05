# full assembly of the sub-parts to form the complete net
import torch
import torch.nn as nn
import torch.nn.functional as F

from awesome.model.cnn_net import concat_input
from awesome.util.batcherize import batcherize


class UNet(nn.Module):
    def __init__(self, 
                 in_chn: int = None, 
                 out_chn: int = 1,
                 dtype: torch.dtype = torch.float32,
                 decoding: bool = False,
                 ):
        super(UNet, self).__init__()
        if decoding:
            return
        self.inc = InConv(in_chn, 64, dtype=dtype)
        self.down1 = Down(64, 128, dtype=dtype)
        self.down2 = Down(128, 256, dtype=dtype)
        self.down3 = Down(256, 512, dtype=dtype)
        self.down4 = Down(512, 512, dtype=dtype)
        self.up1 = Up(1024, 256, dtype=dtype)
        self.up2 = Up(512, 128, dtype=dtype)
        self.up3 = Up(256, 64, dtype=dtype)
        self.up4 = Up(128, 64, dtype=dtype)
        self.outc = OutConv(64, out_chn, dtype=dtype)

 
    @batcherize(keep=True)
    def forward(self, image, feature_encoding, *args, **kwargs):
        x = torch.cat((image, feature_encoding), dim=1)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x

# sub-parts of the U-Net model

class DoubleConv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch, dtype: torch.dtype = torch.float32):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, dtype=dtype),
            nn.BatchNorm2d(out_ch, dtype=dtype),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, dtype=dtype),
            nn.BatchNorm2d(out_ch, dtype=dtype),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class InConv(nn.Module):
    def __init__(self, in_ch, out_ch, dtype: torch.dtype = torch.float32):
        super(InConv, self).__init__()
        self.conv = DoubleConv(in_ch, out_ch, dtype=dtype)

    def forward(self, x):
        x = self.conv(x)
        return x


class Down(nn.Module):
    def __init__(self, in_ch, out_ch, dtype: torch.dtype = torch.float32):
        super(Down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch, dtype=dtype)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class Up(nn.Module):
    def __init__(self, 
            in_ch, 
            out_ch, 
            bilinear=True, 
            dtype: torch.dtype = torch.float32):
        super(Up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2, dtype=dtype)

        self.conv = DoubleConv(in_ch, out_ch, dtype=dtype)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))
        
        # for padding issues, see 
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch, dtype: torch.dtype = torch.float32):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1, dtype=dtype)

    def forward(self, x):
        x = self.conv(x)
        return x