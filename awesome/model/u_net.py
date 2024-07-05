
# Unet code of scribble segmentation
# from awesome.model.cnn_net import concat_input
# from torch.nn import nn
# import torch
# import torch.nn.functional as F

# class UNet(nn.Module):
#     ''' U-Net-Network'''


#     def __init__(self, n_channels, n_classes, width=64, bilinear=True, in_type="rgb"):
#         ''' Initializes the Module UNet.

#         Args:
#             n_channels: 
#                 number of input channel.
#             n_classes: 
#                 number of output channel.
#             width: 
#                 width of Downscaling and Upscaling layers.
#             in_type: 
#                 can become: 'rgb', 'xy' or 'rgbxy'. Decides if the network uses 
#                 the plain image data, the plain feature or both concatenated as input.
#         '''

#         super(UNet, self).__init__()

#         n_classes = n_classes

#         self.in_chn = n_channels
#         self.out_chn = n_classes
#         self.in_type = in_type

#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         self.bilinear = bilinear

#         width = width

#         self.inc = DoubleConv(n_channels, width)
#         self.down1 = Down(width, width*2)
#         self.down2 = Down(width*2, width*4)
#         self.down3 = Down(width*4, width*8)
#         self.down4 = Down(width*8, width*8)

#         self.up1 = Up(width*16, width*4, bilinear)
#         self.up2 = Up(width*8, width*2, bilinear)
#         self.up3 = Up(width*4, width, bilinear)
#         self.up4 = Up(width*2, width, bilinear)
#         self.outc = OutConv(width, n_classes)

#     def forward(self, patch_image, patch_grid):
#         x = concat_input(self.in_type, patch_image, patch_grid)

#         x1 = self.inc(x)
#         x2 = self.down1(x1) # Out:128
#         x3 = self.down2(x2) # Out:256
#         x4 = self.down3(x3) # Out:512
#         x5 = self.down4(x4) # Out:512

#         x = self.up1(x5, x4)    # In: 512,512   Out: 256
#         x = self.up2(x, x3)     # In: 256,256   Out: 128
#         x = self.up3(x, x2)     # In: 128,128   Out: 64
#         x = self.up4(x, x1)     # In: 64,64     Out: 64

#         logits = self.outc(x)
#         return logits
    

# class DoubleConv(nn.Module):
#     """(convolution => [BN] => ReLU) * 2"""


#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.double_conv = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x):
#         return self.double_conv(x)


# class Down(nn.Module):
#     """Downscaling with maxpool then double conv"""


#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.maxpool_conv = nn.Sequential(
#             nn.MaxPool2d(2),
#             DoubleConv(in_channels, out_channels)
#         )

#     def forward(self, x):
#         return self.maxpool_conv(x)


# class Up(nn.Module):
#     """Upscaling then double conv"""


#     def __init__(self, in_channels, out_channels, bilinear=True):
#         super().__init__()

#         # if bilinear, use the normal convolutions to reduce the number of channels
#         if bilinear:
#             self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         else:
#             self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

#         self.conv = DoubleConv(in_channels, out_channels)

#     def forward(self, x1, x2):
#         x1 = self.up(x1)
#         diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
#         diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

#         x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
#                         diffY // 2, diffY - diffY // 2])

#         x = torch.cat([x2, x1], dim=1)
#         return self.conv(x)


# class OutConv(nn.Module):


#     def __init__(self, in_channels, out_channels):
#         super(OutConv, self).__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

#     def forward(self, x):
#         return self.conv(x)