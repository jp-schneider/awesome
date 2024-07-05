import torch
import torch.nn as nn

from awesome.util.batcherize import batcherize

class ForwardModule(nn.Module):
    """This module does nothing, but forwards the input to the output."""

    def __init__(self, *args, **kwargs):
        super().__init__()



    @batcherize(keep=True)
    def forward(self, fwd_input: torch.Tensor, *args, **kwargs):
        return fwd_input  