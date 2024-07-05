from awesome.model.cnn_net import concat_input
from torch.nn import nn
import torch
import torch.nn.functional as F

class DenseNet(nn.Module):
    ''' Densenet-Network'''


    def __init__(self, in_chn, out_chn, in_type):
        ''' Initializes the Module Densenet.

        Args:
            in_chn: 
                number of input channel.
            out_chn: 
                number of output channel.
            in_type: 
                can become: 'rgb', 'xy' or 'rgbxy'. Decides if the network uses 
                the plain image data, the plain feature or both concatenated as input.
        '''
        super(DenseNet, self).__init__()

        self.in_chn = in_chn
        self.out_chn = out_chn
        self.in_type = in_type

        kernel_size = 7
        kernel_size_small = 3
        padding = kernel_size//2

        self.conv0 = nn.Conv2d(in_chn, 16, kernel_size, padding=padding)
        self.conv1 = nn.Conv2d(16*1 + in_chn, 16, kernel_size_small, padding=1)
        self.conv2 = nn.Conv2d(16*2 + in_chn, 16, kernel_size_small, padding=1)
        self.conv3 = nn.Conv2d(16*3 + in_chn, 16, kernel_size_small, padding=1)
        self.conv4 = nn.Conv2d(16*4 + in_chn, out_chn, kernel_size_small, padding=1)

    def forward(self, patch_image, patch_grid):
        patch_in = concat_input(self.in_type, patch_image, patch_grid)
        x = (patch_in)
        x = torch.cat((F.relu(self.conv0(x)), x), dim=1)
        x = torch.cat((F.relu(self.conv1(x)), x), dim=1)
        x = torch.cat((F.relu(self.conv2(x)), x), dim=1)
        x = torch.cat((F.relu(self.conv3(x)), x), dim=1)
        x = self.conv4(x)
        return x