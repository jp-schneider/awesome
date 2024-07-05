from typing import Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from awesome.util.batcherize import batcherize


def conv_relu(width, kernel_size):
    return nn.Sequential(
        nn.Conv2d(width, width, kernel_size=kernel_size, padding=kernel_size//2),
        nn.ReLU()
    )



def concat_input(in_type: Literal['rgb', 'xy', 'rgbxy'], patch_image: torch.Tensor, patch_grid: torch.Tensor) -> torch.Tensor:
    ''' Returns the defined feature (eventually concatenated).'''

    if in_type == 'rgb':
        patch_in = patch_image
    elif in_type == 'xy':
        patch_in = patch_grid
    elif in_type == 'rgbxy':
        patch_in = torch.cat((patch_image, patch_grid.float()), dim=1)
    else:
        raise ValueError(f'in_type must be one of: rgb, xy, rgbxy but was: {in_type}')
    return patch_in

class CNNNet(nn.Module):
    ''' CNN-Network

    Convolutional Neural Network, with variable width and depth.
    '''


    def __init__(self, 
                 in_chn: int = None, 
                 out_chn: int = None, 
                 kernel_size: int = None, 
                 width: int = None, 
                 depth: int = None, 
                 in_type: Literal['rgb', 'xy', 'rgbxy'] = None,
                 input: Literal['rgb', 'xy', 'rgbxy'] = None,
                 decoding: bool = False):
        ''' Initializes the Module CNN_Net.

        Args:
            in_chn: 
                number of input channel.
            out_chn: 
                number of output channel.
            kernel_size: 
                convolutional kernel size.
            width: 
                width of convolutional layer.
            depth: 
                depth of convolutional layer.
            in_type: 
                can become: 'rgb', 'xy' or 'rgbxy'. Decides if the network uses 
                the plain image data, the plain feature or both concatenated as input.
        '''
        if decoding:
            return
        super().__init__()

        self.in_chn = in_chn
        self.out_chn = out_chn
        self.in_type = in_type

        assert (kernel_size % 2) == 1

        conv_blocks = [conv_relu(width, kernel_size) 
                       for i in range(depth)]

        self.model = nn.Sequential(
            nn.Conv2d(in_chn, width, kernel_size=kernel_size, padding=kernel_size//2),
            nn.LeakyReLU(),
            *conv_blocks,
            nn.Conv2d(width, out_chn, kernel_size=kernel_size, padding=kernel_size//2)
            )

    

    @batcherize(keep=True)
    def forward(self, image, grid, *args, **kwargs):
        ''' Forward Path of the Module CNN_Net.

        Args:
            image: 
                the rgb input image.
            grid: 
                the spacial or semantic features.
        '''
        patch_in = concat_input(self.in_type, image, grid)
        x = self.model(patch_in)
        return x  

