import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



class Net(nn.Module):
    def __init__(self,n_hidden: int = 130, **kwargs):
        # call constructor from superclass
        super().__init__()
        
        # define network layers
        self.W0 = nn.Linear(5, n_hidden)
        self.W1 = nn.Linear(n_hidden, n_hidden)
        self.W2 = nn.Linear(n_hidden, 1)
        
        
    def forward(self, x: torch.Tensor, **kwargs):
        # define forward pass
        x_input = x
        x = F.relu(self.W0(x))
        x = F.relu(self.W1(x))
        x = self.W2(x)
        return x
