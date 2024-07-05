from typing import Optional
import torch
import torch.nn as nn
from awesome.util.torch import TensorUtil
from awesome.util.pixelize import pixelize

class PixelizeNet(nn.Module):

    def __init__(self, network: nn.Module, max_batch_size: Optional[int] = None) -> None:
        super().__init__()
        self.network = network
        self.max_batch_size = max_batch_size # Hack to address https://github.com/pytorch/pytorch/issues/79191 on some networks cause of error in solve_triangular

    @pixelize()
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # Limit batch size to avoid CUDA Error
        if self.max_batch_size is None:
            return self.network(x, *args, **kwargs)
        else:
            num_batches, _ = divmod(x.shape[0], self.max_batch_size)
            _x = []
            for _i in range(num_batches + 1):
                _sub_x = x[_i*self.max_batch_size: min((_i+1)*self.max_batch_size, x.shape[0])]
                _x.append(self.network(_sub_x))
            x = torch.cat(_x, dim=0)
            return x
    
    @pixelize()
    def inverse(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if self.max_batch_size is None:
            return self.network.inverse(x, *args, **kwargs)
        else:
            raise NotImplementedError("Inverse with max_batch_size not implemented.")

    def reset_parameters(self) -> None:
        return TensorUtil.reset_parameters(self.network)