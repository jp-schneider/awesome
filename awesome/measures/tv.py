from typing import Any, Optional
import torch
import torch.nn as nn


class TV(nn.Module):
    ''' Total Variation.'''


    def __init__(self):
        super().__init__()


    def forward(self, 
                x: Optional[torch.Tensor], 
                _input: Optional[Any] = None, 
                **kwargs) -> torch.Tensor:
        """Return the total variation of the input tensor.

        Parameters
        ----------
        x : Optional[torch.Tensor]
            Input tensor the compute the total variation of.
        image : Optional[torch.Tensor], optional
            Reference input image. Should be of shape (3 x H x W), by default None

        Returns
        -------
        torch.Tensor
            The total variation of the input tensor.
        """
        # TODO This Copied loss is not correct => Does not perform a classical TV
        # TV
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()

        # Weight
        loss_weight = 1
        image = None
        if _input is not None and len(_input) > 0 and isinstance(_input[-1], dict):
            image = _input[-1].get("clean_image", None)

        if image is not None:
            gamma = 5
            g_img = torch.mean(image, dim=1)
            h_tv_img = torch.pow((g_img[:,1:,:]-g_img[:,:-1,:]),2).sum()
            w_tv_img = torch.pow((g_img[:,:,1:]-g_img[:,:,:-1]),2).sum()
            deriv_img = (torch.abs(h_tv_img/count_h) + torch.abs(w_tv_img/count_w))/batch_size
            loss_weight = torch.exp(-gamma * deriv_img)/2

        return loss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]
