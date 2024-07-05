from dataclasses import dataclass, field
from typing import List, Literal, Optional, Set, Union
from matplotlib.figure import Figure

import torch
from awesome.mixin.fast_repr_mixin import FastReprMixin

from awesome.util.series_convertible_mixin import SeriesConvertibleMixin
from awesome.util.matplotlib import saveable

@dataclass(repr=False)
class DataSample(SeriesConvertibleMixin, FastReprMixin):
    """Interface class for a data sample which is used in the awesome dataset. Subclasses or correspondant items should be returned in the data loader per dataset."""

    label: torch.Tensor = field(default=None)
    """The ground truth label"""

    weak_label: Optional[torch.Tensor] = field(default=None)
    """Weak label like scribbles or points"""

    image: torch.Tensor = field(default=None)
    """The image data"""

    name: str = field(default=None)
    """Some name or index of the sample"""

    use_memory_cache: bool = field(default=False)
    """If the sample should be cached in memory, images loaded only once for faster access"""

    has_label: bool = field(default=True)
    """If the sample has a label (ground truth) or is a weak segmentation only."""

    WEAK_LABEL_NONECLASS: float = 2.
    """None class for weak labels, e.g. for scribbles or points. This is used to indicate that the pixel is not labeled."""

    WEAK_LABEL_FOREGROUND: Union[int, List[int]] = 0.
    """Foreground class(es) for weak labels, e.g. for scribbles or points. This is used to indicate that the pixel is labeled as foreground."""

    WEAK_LABEL_BACKGROUND: int = 1.
    """Background class for weak labels, e.g. for scribbles or points. This is used to indicate that the pixel is labeled as background."""

    @property
    def feat_name(self) -> Optional[str]:
        """Name of the extracted sematic feature, if the sample has a feature."""
        return self.name
    
    @feat_name.setter
    def feat_name(self, value: Optional[str]):
        pass

    @classmethod
    def ignore_on_repr(cls) -> Set[str]:
        ret = super().ignore_on_repr()
        ret.add('use_memory_cache')
        ret.add('label')
        ret.add('image')
        ret.add('weak_label')
        return ret

    @property
    def mask(self) -> Optional[torch.Tensor]:
        """Alias for weak_label"""
        return self.weak_label
    
    @property
    def clean_image(self) -> Optional[torch.Tensor]:
        """Alias for image"""
        return self.image

    def __getitem__(self, key: str):
        """Get an attribute by key."""
        if key == 'label':
            return self.label
        elif key == 'weak_label':
            return self.weak_label
        elif key == 'image':
            return self.image
        elif key == 'mask':
            return self.mask
        elif key == 'clean_image':
            return self.clean_image
        elif key == 'name':
            return self.name
        elif key == 'feat_name':
            return self.feat_name
        else:
            raise KeyError(f"Key {key} not found")

    @saveable()
    def plot(self, 
             mode: Literal['weak_label', 'label'] = 'weak_label',
             **kwargs) -> Figure:
        """Plot the image with the weak label or the label.

        Parameters
        ----------
        mode : Literal[&#39;weak_label&#39;, &#39;label&#39;], optional
            Which mask to plot weak_label (like scribbles) or label / ground truth, by default 'weak_label'

        Returns
        -------
        Figure
            Matplot figure
        """
        from awesome.run.functions import plot_mask
        if 'scale' not in kwargs:
            kwargs['scale'] = 3
        if 'tight' not in kwargs:
            kwargs['tight'] = True
        mask = None
        if mode == 'weak_label':
            mask = self.weak_label
        elif mode == 'label':
            mask = self.label
        return plot_mask(self.image, mask, **kwargs)