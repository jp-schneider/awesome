from awesome.model.cnn_net import concat_input
import torch.nn as nn

def linear_relu(width):
    return nn.Sequential(
        nn.Linear(width, width),
        nn.ReLU()
    )

class FCNet(nn.Module):
    ''' FC-Network

    Fully Connected Neural Network, with variable width and depth.
    '''


    def __init__(self, 
        in_chn: int = None, 
        out_chn: int = None, 
        width: int = None, 
        depth: int = None, 
        in_type: str =None,
                 decoding: bool = False):
        ''' Initializes the Module FC_Net.

        Args:
            in_chn: 
                number of input channel.
            out_chn: 
                number of output channel.
            width: 
                width of Linear layer.
            depth: 
                depth of Linear layer.
            in_type: 
                can become: 'rgb', 'xy' or 'rgbxy'. Decides if the network uses 
                the plain image data, the plain feature or both concatenated as input.
        '''

        super().__init__()

        if decoding:
            return
        self.in_chn = in_chn
        self.out_chn = out_chn
        self.in_type = in_type

        linear_blocks = [linear_relu(width) for i in range(depth)]
        
        self.model = nn.Sequential(
            nn.Linear(in_chn, width),
            nn.ReLU(),
            *linear_blocks,
            nn.Linear(width, out_chn),
            )

    def forward(self, image, grid, *args, **kwargs):
        patch_in = concat_input(self.in_type, image, grid)
        x = self.model(patch_in)
        return x  