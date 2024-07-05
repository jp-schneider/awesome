# Class for functions
import logging
import matplotlib as mpl
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.colors import LightSource, LinearSegmentedColormap, to_hex, to_rgb
from matplotlib.image import AxesImage
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
# File for useful functions when using matplotlib
import math
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
from awesome.dataset.sisbosi_dataset import SISBOSIDataset
from awesome.dataset.convexity_segmentation_dataset import ConvexitySegmentationDataset, OutputMode
from awesome.dataset.prior_dataset import PriorManager
from awesome.error.missing_ground_truth_error import MissingGroundTruthError
from awesome.model.fc_net import FCNet
from awesome.model.wrapper_module import WrapperModule
from awesome.transforms.min_max import minmax
from awesome.util.channelize import channelize
from awesome.util.temporary_property import TemporaryProperty
from mpl_toolkits.axes_grid1 import make_axes_locatable
from awesome.dataset.awesome_dataset import AwesomeDataset
from awesome.util.torch import VEC_TYPE, TensorUtil
try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
except (ModuleNotFoundError, ImportError):
    plt = None
    Figure = None
    pass
import os
from functools import partial, wraps
from awesome.util.path_tools import numerated_file_name
import numpy as np
import torch
import cv2
from awesome.util.matplotlib import saveable
from dataclasses import dataclass, field
from matplotlib.colors import ListedColormap, to_rgba

try:
    from awesome.run.crf import dense_crf, CRFOptions
except (ModuleNotFoundError, ImportError) as err:
    dense_crf = None
    CRFOptions = None
    logging.warning(
        "Error importing CRF module. CRF will not be available.")


def should_use_logarithm(x: np.ndarray, magnitudes: int = 2, allow_zero: bool = True) -> bool:
    """Checks if the data should be plotted with logarithmic scale.
    Result is calculated based on orders of magnitude of the data.

    Parameters
    ----------
    x : np.ndarray
        Data to be plotted

    magnitudes : int, optional
        Number of magnitudes the data should span to be plotted with logarithmic scale, by default 2

    allow_zero : bool, optional
        If zero values should be allowed, by default True

    Returns
    -------
    bool
        If the data should be plotted with logarithmic scale.
    """
    if not allow_zero:
        if np.any(x <= 0):
            return False
    return np.max(x) / np.min(x) > math.pow(10, magnitudes)


def register_alpha_map(base_name: str = 'binary', renew: bool = False) -> str:
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib import colormaps
    name = f'alpha_{base_name}'

    try:
        plt.get_cmap(name)
    except ValueError as err:
        pass
    else:
        if not renew:
            return name  # Already exists
        else:
            from matplotlib import cm
            colormaps.unregister(name)

    # get colormap
    ncolors = 256

    base_map = plt.get_cmap(base_name)
    N = base_map.N
    color_array = base_map(range(N))

    # change alpha values
    color_array[:, -1] = np.linspace(0, 1.0, ncolors)

    # create a colormap object
    map_object = LinearSegmentedColormap.from_list(
        name=name, colors=color_array)

    # register this new colormap with matplotlib
    colormaps.register(cmap=map_object)
    return name


def create_alpha_colormap(
        name: str,
        color: np.ndarray,
        ncolors: int = 256
) -> LinearSegmentedColormap:
    """Creates a linear alpha colormap with matplotlib.
    Colormap has static RGB values and linear alpha values from 0 to 1.
    Meaning 0 is transparent and 1 is opaque.

    Parameters
    ----------
    name : str
        Name of the new colormap.
    color : np.ndarray
        Colorvalues of the colormap. Shape is (3, ).
    ncolors : int, optional
        Number of colors in the map, by default 256

    Returns
    -------
    LinearSegmentedColormap
        The created colormap.
    """

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    color_array = np.zeros((ncolors, 4))
    color_array[:, :3] = color

    # change alpha values
    color_array[:, -1] = np.linspace(0, 1.0, ncolors)

    # create a colormap object
    map_object = LinearSegmentedColormap.from_list(
        name=name, colors=color_array)

    return map_object


def register_alpha_colormap(color: np.ndarray,
                            name: str,
                            renew: bool = False,
                            ncolors: int = 256) -> str:
    """Registers a linear alpha colormap with matplotlib.
    Colormap has static RGB values and linear alpha values from 0 to 1.
    Meaning 0 is transparent and 1 is opaque.

    Parameters
    ----------
    color : np.ndarray
        Colorvalues of the colormap. Shape is (3, ).
    name : str
        Name of the new colormap.
    renew : bool, optional
        If the map should be recreated when it exists, by default False
    ncolors : int, optional
        Number of colors in the map, by default 256

    Returns
    -------
    str
        Name of the colormap
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    try:
        plt.get_cmap(name)
    except ValueError as err:
        pass
    else:
        if not renew:
            return name  # Already exists
        else:
            from matplotlib import cm
            cm._colormaps.unregister(name)

    map_object = create_alpha_colormap(name, color, ncolors)
    # register this new colormap with matplotlib
    plt.register_cmap(cmap=map_object)
    return name


register_alpha_map('binary')
register_alpha_map('Greens')
register_alpha_map('Reds')
register_alpha_map('Blues')


def transparent_added_listed_colormap(base_cmap: str = "tab10") -> ListedColormap:
    """Creates a transparent version of a given colormap, where the first color is transparent. Increasing the number of colors by one.

    Parameters
    ----------
    base_cmap : str, optional
        Base color map, by default "tab10"

    Returns
    -------
    ListedColormap
        The created colormap with the first color being transparent.
    """
    tab_10_cmap = plt.get_cmap(base_cmap)
    colors = [to_rgba(x) for x in tab_10_cmap.colors]
    colors = [(0, 0, 0, 0)] + colors
    cmap = ListedColormap(colors=colors, name=f"transparent_{base_cmap}")
    return cmap


@saveable()
def plot_image_scribbles(image: VEC_TYPE,
                         inference_result: VEC_TYPE,
                         foreground_mask: VEC_TYPE = None,
                         background_mask: VEC_TYPE = None,
                         prior_result: Optional[VEC_TYPE] = None,
                         boxes: VEC_TYPE = None,
                         labels: VEC_TYPE = None,
                         scores: VEC_TYPE = None,
                         size: int = 5,
                         tight: bool = False,
                         title: str = None,
                         background_value: int = 1,
                         ax: Optional[Axes] = None,
                         legend: bool = True,
                         legend_label_mapping: Optional[Dict[str, str]] = None,
                         **kwargs) -> Figure:
    import matplotlib.patches as mpatches

    foreground_mask = foreground_mask.detach().cpu().numpy() if isinstance(
        foreground_mask, torch.Tensor) else foreground_mask
    background_mask = background_mask.detach().cpu().numpy() if isinstance(
        background_mask, torch.Tensor) else background_mask
    inference_result = inference_result.detach().cpu().numpy() if isinstance(
        inference_result, torch.Tensor) else inference_result
    if prior_result is not None:
        prior_result = prior_result.detach().cpu().numpy() if isinstance(
            prior_result, torch.Tensor) else prior_result

    image = image.detach().cpu().permute(1, 2, 0).numpy(
    ) if isinstance(image, torch.Tensor) else image

    if foreground_mask is None:
        foreground_mask = np.zeros(image.shape[:2])
    if background_mask is None:
        background_mask = np.zeros(image.shape[:2])

    fig = None
    if ax is None:
        fig, ax = get_mpl_figure(
            1, 1, tight=tight, size=size, ratio_or_img=image)
    else:
        fig = ax.figure

    if legend_label_mapping is None:
        legend_label_mapping = dict()

    # ax.imshow(image)

    # ax.imshow(inference_result == 0., cmap='alpha_binary', alpha=0.8, label='')

    mask = inference_result
    if prior_result is not None:
        mask = np.stack([mask, prior_result], axis=-1)
    else:
        if len(mask.shape) == 2:
            if isinstance(mask, torch.Tensor):
                mask = mask.unsqueeze(0)
            if isinstance(mask, np.ndarray):
                mask = mask[..., None]
    _colors = list()

    scale = 3 if labels is not None else 1

    if scale != 1.:
        foreground_mask = cv2.resize(
            foreground_mask, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        background_mask = cv2.resize(
            background_mask, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

    plot_mask_labels(image=image, mask=mask,
                     boxes=boxes, labels=labels,
                     scores=scores,
                     ax=ax,
                     _colors=_colors,
                     scale=scale,
                     background_value=background_value)

    fgh = ax.imshow(foreground_mask, cmap='alpha_Greens', alpha=1)
    bgh = ax.imshow(background_mask, cmap='alpha_Reds', alpha=1)

    cmap_labels = dict()

    if (foreground_mask > 0).any():
        cmap_labels['Scribble Foreground'] = 'alpha_Greens'
    if (background_mask > 0).any():
        cmap_labels['Scribble Background'] = 'alpha_Reds'

    label_colors = {}

    label_colors['Segmentation'] = _colors[0]
    if prior_result is not None:
        label_colors['Prior'] = _colors[1]

    for k, v in cmap_labels.items():
        base_map = plt.get_cmap(v)
        label_colors[k] = base_map(range(base_map.N))[-1]
        pass

    if legend:
        patches = [mpatches.Patch(color=v, label=legend_label_mapping.get(
            k, k)) for k, v in label_colors.items()]
        # plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
        ax.legend(handles=patches)

    ax.axis('off')
    # plt.legend()
    if title is not None:
        fig.suptitle(title)
    return fig


def meshgrid_of_img(img: VEC_TYPE) -> Tuple[np.ndarray, np.ndarray]:
    img = img.detach().cpu().permute(1, 2, 0).numpy(
    ) if isinstance(img, torch.Tensor) else img
    x = np.arange(0, img.shape[1])
    y = np.arange(0, img.shape[0])
    xx, yy = np.meshgrid(x, y)
    return xx, yy


def preserve_legend(ax: Axes, patches: List[Patch], **kwargs):
    if ax.get_legend() is not None:
        lgd = ax.get_legend()
        handles = list(lgd.legend_handles)
        labels = [x.get_label() for x in lgd.legend_handles]
        handles.extend(patches)
        labels.extend([p.get_label() for p in patches])
        ax.legend(handles=handles, labels=labels, **kwargs)
    else:
        ax.legend(handles=patches, **kwargs)


@saveable()
def plot_mask(image: VEC_TYPE,
              mask: VEC_TYPE,
              size: int = 5,
              title: str = None,
              tight: bool = False,
              background_value: int = 0,
              ignore_class: Optional[int] = None,
              _colors: Optional[List[str]] = None,
              color: str = "#5999cb",
              contour_linewidths: float = 2,
              object_mode: Literal['mask_value_is_object',
                                   'mask_channel_is_object'] = 'mask_channel_is_object',
              ax: Optional[Axes] = None,
              darkening_background: float = 0.7,
              labels: Optional[List[str]] = None,
              lined_contours: bool = True,
              filled_contours: bool = False,
              axes_description: bool = False,
              image_cmap: Optional[Any] = None,
              **kwargs) -> Figure:  # type: ignore
    import matplotlib.patches as mpatches

    if isinstance(mask, torch.Tensor):
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)

    mask = mask.detach().cpu().permute(1, 2, 0).numpy(
    ) if isinstance(mask, torch.Tensor) else mask
    image = image.detach().cpu().permute(1, 2, 0).numpy(
    ) if isinstance(image, torch.Tensor) else image
    mask = mask.squeeze()
    if len(mask.shape) == 2:
        mask = mask[..., None]
    # Check if mask contains multiple classes

    if ignore_class is not None:
        fill = np.zeros_like(mask)
        fill.fill(background_value)
        mask = np.where(mask == ignore_class, fill, mask)

    channel_mask = None

    if object_mode == 'mask_value_is_object':
        vals = np.unique(mask)
        multi_class = len(vals) > 2
        background_mask = np.where(mask != background_value, np.zeros_like(mask),
                                   np.ones_like(mask))  # True if not background
        channel_mask = np.zeros(mask.shape + (len(vals) - 1,))

        _valid_classes = [x for x in vals if x != background_value and (
            ignore_class is None or x != ignore_class)]
        for i, c in enumerate(_valid_classes):
            # True if not class i, 0 means fg for every channel
            channel_mask[..., i] = ~(mask == c)
    elif object_mode == 'mask_channel_is_object':
        channel_mask = mask
        multi_class = mask.shape[2] > 1
        any_fg_mask = np.clip(np.sum(np.where(
            mask != background_value, 1, 0), axis=-1), 0, 1)  # True if not background
        background_mask = np.logical_not(any_fg_mask).astype(float)

    cmap_name = 'Blues'

    fig = None
    if ax is None:
        if tight:
            sizes = np.shape(mask)
            fig = plt.figure(figsize=(size, size))
            dpi = 300
            fig.set_size_inches(
                size * (sizes[1] / dpi), size * (sizes[0] / dpi), forward=False)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
        else:
            fig, ax = plt.subplots(1, 1, figsize=(size, size))
    else:
        fig = ax.figure

    if image is not None:
        ax.imshow(image, cmap=image_cmap)

    cmap = plt.get_cmap("alpha_binary")

    alpha = 0.8
    # color = cmap(int(alpha * cmap.N))

    cmap = "tab10" if channel_mask.shape[-1] <= 10 else "tab20"
    if isinstance(color, (list, tuple)) and multi_class:
        colors = color
    else:
        colors = [color] if not multi_class else plt.get_cmap(
            cmap)(range(channel_mask.shape[-1]))

    m_inv = np.ones(mask.shape[:-1])

    patches = []
    for i in range(channel_mask.shape[-1]):
        m = channel_mask[..., i]
        label = labels[i] if labels is not None else None
        if lined_contours:
            ax.contour(
                m_inv - m, levels=[0.5], colors=[colors[i]], linewidths=contour_linewidths)
        if filled_contours:
            _color = to_rgba(colors[i][:])
            c_img = np.zeros((*m.shape, 4))
            c_img[:, :, :] = _color
            c_img[:, :, -1] = c_img[:, :, -1] * m
            ax.imshow(c_img)
        if label is not None:
            patches.append(mpatches.Patch(color=colors[i], label=label))

    ax.imshow(background_mask, cmap='alpha_binary',
              alpha=darkening_background, label='')

    if not tight:
        ax.axis('off')
    # plt.legend()
    if title is not None:
        fig.suptitle(title)

    if _colors is not None:
        _colors.clear()
        _colors.extend(colors)

    if patches is not None and len(patches) > 0:
        preserve_legend(ax, patches)

    origin_marker_color = kwargs.get('origin_marker_color', None)
    origin_marker_opposite_color = kwargs.get(
        'origin_marker_opposite_color', None)

    if origin_marker_color is not None or origin_marker_opposite_color is not None:
        from matplotlib.colors import get_named_colors_mapping
        # Create markers with imshow
        transparent_nav = np.zeros((*image.shape[:2], 4))
        cmap = get_named_colors_mapping()

        def make_circle(r):
            y, x = np.ogrid[-r:r, -r:r]
            return x**2 + y**2 <= r**2

        origin_marker_size = kwargs.get('origin_marker_size', 24)
        # int(round(math.sqrt((origin_marker_size) / np.pi)))
        marker_radius = origin_marker_size

        if origin_marker_color is not None:
            if isinstance(origin_marker_color, str):
                origin_marker_color = cmap[origin_marker_color]
            origin_marker_color = tuple(to_rgb(origin_marker_color))
            origin_marker_color += (1,)
            # Coloring the origin with 10 pixels
            transparent_nav[:2 * marker_radius, :2 *
                            marker_radius][make_circle(marker_radius)] = origin_marker_color

        if origin_marker_opposite_color is not None:
            if isinstance(origin_marker_opposite_color, str):
                origin_marker_opposite_color = cmap[origin_marker_opposite_color]
            origin_marker_opposite_color = tuple(
                to_rgb(origin_marker_opposite_color))
            origin_marker_opposite_color += (1,)
            # Coloring the origin with 10 pixels
            transparent_nav[-2 * marker_radius:, -2 * marker_radius:][make_circle(
                marker_radius)] = origin_marker_opposite_color

        ax.imshow(transparent_nav)

    if axes_description:
        ax.set_axis_on()
        ax.set_xlabel("Coordinates [x]")
        ax.set_ylabel("Coordinates [y]")
    return fig


def image_subsample(img: torch.Tensor,
                    factor: int = 6,
                    mode: Literal["grid_sample", "slicing"] = "grid_sample",
                    grid_sample_mode: str = "bilinear",
                    ) -> torch.Tensor:
    """Downsamples an image by a factor.


    Parameters
    ----------
    img : torch.Tensor
        Tensor image
    factor : int, optional
        The factor to downsample, by default 6
    mode : Literal[&quot;grid_sample&quot;, &quot;slicing&quot;], optional
        The mode to donwsample, either slicing or grid_sample, by default "slicing"
    grid_sample_mode : str, optional
        The mode to use for grid_sample, by default "bilinear"

    Returns
    -------
    torch.Tensor
        The downsampled image

    """
    import torch.nn.functional as F
    if mode == "grid_sample":
        x = torch.arange(-1, 1, factor * 2 / img.shape[-2])
        y = torch.arange(-1, 1, factor * 2 / img.shape[-1])
        xx, yy = torch.meshgrid(x, y)
        flowgrid = torch.stack((yy, xx), dim=-1).float()[None, ...]
        return F.grid_sample(img[None, ...], flowgrid, align_corners=True, mode=grid_sample_mode)[0, ...]
    elif mode == "slicing":
        return img[..., ::factor, ::factor]
    else:
        raise ValueError("Invalid mode")


def subsample_mask(x: torch.Tensor, subsample: int = 25, also_last: bool = False) -> torch.Tensor:
    """Create a subsample mask for a given input. The mask will be True at every subsample point.
    Only works inputs of shape (C, H, W) whereby C should be 2.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor
    subsample : int, optional
        The subsample point distance, by default 25
    also_last : bool, optional
        If also the last row / column should be included, by default False

    Returns
    -------
    torch.Tensor
        Result mask of shape (H, W)
    """
    image_shape = x.shape[-2:]
    ones_grid = torch.ones(x[0].shape)
    subsampled_grid = torch.zeros(x[0].shape)
    coords = (torch.argwhere(ones_grid) % subsample) == 0

    if also_last:
        # Add also last in any dimension
        is_last = torch.zeros((2, *image_shape), dtype=bool)
        is_last[0, -1] = True
        is_last[1, :, -1] = True
        is_last_coord_form = is_last.permute(
            1, 2, 0).reshape((is_last[0].numel(), 2))
        coords = coords | is_last_coord_form

    coords_mask = coords.all(dim=-1).reshape((image_shape))
    subsampled_grid[coords_mask] = 1
    return subsampled_grid.bool()


@saveable()
def plot_grid(grid: torch.Tensor,
              mask: torch.Tensor,
              ax: Optional[Axes] = None,
              tight: Optional[bool] = False,
              size: Optional[float] = 5,
              color: str = "b",
              dense: bool = True,
              origin: Literal['lower', 'upper'] = "upper",
              linewidth: float = 1,
              outer_line_colors: Optional[list] = None,
              grid_outer_linewidth: float = 1,
              **kwargs
              ) -> Figure:
    """Plots a grid with a subsampling mask.

    Parameters
    ----------
    grid : torch.Tensor
        Grid coordinates of shape (2, H, W)
    mask : torch.Tensor
        A subsampling mask of shape (H, W)
    ax : Optional[Axes], optional
        Optional axis to plot, by default None
    tight : Optional[bool], optional
        If plot should be tight, by default False
    size : Optional[float], optional
        The size of the plot in inches, by default 5
    color : str, optional
        The color for the grid lines, by default "b"
    dense : bool, optional
        If rows / cols should be densly plotted or only each subsampled point, by default True
    origin : Literal[&#39;lower&#39;, &#39;upper&#39;], optional
        Where the origin is, by default "upper"
    linewidth : float, optional
        Linewidth of the grid, by default 1
    outer_line_colors : Optional[list], optional
        Dedicated colors for Top, right, bottom, left. If not specified will be the normal color, by default None
    grid_outer_linewidth : float, optional
        Linewidth of output lines, by default 1

    Returns
    -------
    Figure
        Matplotlib figure
    """

    if ax is None:
        fig, ax = get_mpl_figure(
            1, 1, tight=tight, size=size, ratio_or_img=grid)
    else:
        fig = ax.figure

    dots = torch.argwhere(mask)

    row_idx = torch.unique(dots[:, 0])
    col_idx = torch.unique(dots[:, 1])

    rows = grid[:, row_idx]
    cols = grid[:, :, col_idx]

    # Order top, right, bottom, left
    if outer_line_colors is None:
        outer_line_colors = [color] * 4

    max_z_order = max([_.zorder for _ in ax.get_children()]) + 1

    range_rows = range(rows.shape[1])
    if kwargs.get('grid_rows_slicing', None) is not None:
        range_rows = list(range_rows)[kwargs.get('grid_rows_slicing')]

    use_row_coloring = kwargs.get('use_row_coloring', False)

    for idx in range_rows:
        row = rows[:, idx]
        if dense:
            x = row[0]
            y = row[1]
        else:
            x = row[0, col_idx]
            y = row[1, col_idx]
        _c = color
        _w = linewidth
        _z = max_z_order + idx - 1

        if idx == 0:
            # Top
            _c = outer_line_colors[0]
            _w = grid_outer_linewidth
            _z = max_z_order + rows.shape[1] - 1
        elif idx == rows.shape[1] - 1:
            # bottom
            _c = outer_line_colors[2]
            _w = grid_outer_linewidth

        if use_row_coloring:
            _c = None  # Ignore specified color and color individual rows

        ax.plot(x, y, color=_c, linewidth=_w, zorder=_z, label=f"Row {idx}")

    max_z_order = max([_.zorder for _ in ax.get_children()]) + 1
    range_cols = range(cols.shape[2])
    if kwargs.get('grid_cols_slicing', None) is not None:
        range_cols = list(range_cols)[kwargs.get('grid_cols_slicing')]

    use_col_coloring = kwargs.get('use_col_coloring', False)

    for idx in range_cols:
        col = cols[:, :, idx]
        if dense:
            x = col[0]
            y = col[1]
        else:
            x = col[0, row_idx]
            y = col[1, row_idx]
        _c = color
        _w = linewidth
        _z = max_z_order + idx - 1
        if idx == 0:
            # Left
            _c = outer_line_colors[3]
            _w = grid_outer_linewidth
            _z = max_z_order + cols.shape[2] - 1
        elif idx == cols.shape[2] - 1:
            # Right
            _c = outer_line_colors[1]
            _w = grid_outer_linewidth

        if use_col_coloring:
            _c = None

        ax.plot(x, y, color=_c, linewidth=_w, zorder=_z, label=f"Col {idx}")

    # Plot corner marker for origin
    origin_marker_color = kwargs.get('origin_marker_color', None)
    origin_marker_opposite_color = kwargs.get(
        'origin_marker_opposite_color', None)

    if origin_marker_color is not None or origin_marker_opposite_color is not None:
        from matplotlib.colors import get_named_colors_mapping, to_rgb
        cmap = get_named_colors_mapping()
        origin_marker_size = kwargs.get('origin_marker_size', None)
        if origin_marker_size is not None:
            origin_marker_size = (
                np.pi * (origin_marker_size * (fig.dpi / 72)) ** 2)
        if origin_marker_color is not None:
            if isinstance(origin_marker_color, str):
                origin_marker_color = cmap[origin_marker_color]
            origin_marker_color = tuple(to_rgb(origin_marker_color))
            ax.scatter(grid[0, 0, 0], grid[1, 0, 0], marker='o',
                       color=origin_marker_color, s=origin_marker_size)
        if origin_marker_opposite_color is not None:
            if isinstance(origin_marker_opposite_color, str):
                origin_marker_opposite_color = cmap[origin_marker_opposite_color]
            origin_marker_opposite_color = tuple(
                to_rgb(origin_marker_opposite_color))
            ax.scatter(grid[0, -1, -1], grid[1, -1, -1], marker='o',
                       color=origin_marker_opposite_color, s=origin_marker_size)

    if origin == "upper":
        ax.invert_yaxis()

    if kwargs.get('grid_legend', False):
        ax.legend()

    return fig


def plot_match(img: torch.Tensor,
               output: torch.Tensor,
               target: torch.Tensor,
               subsample: int = 25,
               subsample_grid_dots: bool = False,
               grid: torch.Tensor = None,
               ax: Optional[Axes] = None,
               **kwargs) -> Figure:
    """Plots an image with the output and target mask
    and eventually a subsample grid.

    Parameters
    ----------
    img : torch.Tensor
        Image tensor
    output : torch.Tensor
        Output mask
    target : torch.Tensor
        Target mask
    subsample : int, optional
        Subsampling if grid dots should be plotted, by default 25
    subsample_grid_dots : bool, optional
        If grid dots should be plotted, by default False
    grid : torch.Tensor, optional
        Custom grid to plot, by default None
    ax : Optional[Axes], optional
        Axis to plot to, by default None

    Returns
    -------
    Figure
        Matplotlib figure
    """
    from awesome.model.path_connected_net import PathConnectedNet

    image_shape = img.shape[-2:]

    add = []
    if target is not None:
        add.append(target)
    if output is not None:
        add.append(output)

    if subsample_grid_dots:
        if grid is None:
            grid = PathConnectedNet.create_coordinate_grid(image_shape)
        subsampled_grid = torch.zeros_like(grid[0])
        coords_mask = subsample_mask(grid, subsample=subsample)
        subsampled_grid[coords_mask] = 1
        add.append(subsampled_grid.float()[None, ...])

    stack_plot = torch.cat(add, dim=0)
    fig = plot_mask(img, stack_plot, ax=ax, **kwargs)

    return fig


def get_mpl_figure(
        rows: int = 1,
        cols: int = 1,
        size: float = 5,
        ratio_or_img: Union[float, np.ndarray] = 1.0,
        tight: bool = False,
        subplot_kw: Optional[Dict[str, Any]] = None,
        ax_mode: Literal["1d", "2d"] = "1d",
) -> Tuple[Figure, Union[Axes, List[Axes]]]:
    """Create a eventually tight matplotlib figure with axes.

    Parameters
    ----------
    rows : int, optional
        Number of rows for the figure, by default 1
    cols : int, optional
        Nombuer of columns, by default 1
    size : float, optional
        Size of the axes in inches, by default 5
    ratio_or_img : float | np.ndarray, optional
        Ratio of Y w.r.t X can also be an Image / np.ndarray which will compute it from the axis, by default 1.0
    tight : bool, optional
        If the figure should be tight => No axis spacing and borders, by default False
    subplot_kw : Optional[Dict[str, Any]], optional
        Optional kwargs for the subplots, by default None
        Only used if tight is False

    Returns
    -------
    Tuple[Figure, Axes | List[Axes]]
        Figure and axes.
    """
    if isinstance(ratio_or_img, torch.Tensor):
        ratio_or_img = ratio_or_img.detach().cpu()
        if len(ratio_or_img.shape) == 4:
            ratio_or_img = ratio_or_img[0]
        if len(ratio_or_img.shape) == 3:
            ratio_or_img = ratio_or_img.permute(1, 2, 0)
        ratio_or_img = ratio_or_img.numpy()

    if isinstance(ratio_or_img, np.ndarray):
        if len(ratio_or_img.shape) == 4:
            ratio_or_img = ratio_or_img[0]
        elif len(ratio_or_img.shape) == 2:
            ratio_or_img = ratio_or_img.shape[-2] / ratio_or_img.shape[-1]
        elif len(ratio_or_img.shape) == 3:
            ratio_or_img = ratio_or_img.shape[-3] / ratio_or_img.shape[-2]

    ratio_x = 1
    ratio_y = ratio_or_img
    dpi = 300
    axes = []
    if tight:
        fig = plt.figure()
        fig.set_size_inches(
            size * ratio_x * cols,
            size * ratio_y * rows,
            forward=False)
        # (left, bottom, width, height)
        rel_width = 1 / cols
        rel_height = 1 / rows
        for i in range(rows * cols):
            col, row = divmod(i, rows)
            ax = plt.Axes(fig, [col * rel_width, row *
                          rel_height, rel_width, rel_height])
            ax.set_axis_off()
            fig.add_axes(ax)
            axes.append(ax)
    else:
        fig, ax = plt.subplots(rows, cols, figsize=(size * ratio_x * cols,
                                                    size * ratio_y * rows), subplot_kw=subplot_kw)
        axes.append(ax)

    if ax_mode == "2d" and tight:
        axes = np.reshape(np.array(axes), (rows, cols), order="F")[::-1]
    elif ax_mode == "2d" and not tight:
        axes = np.reshape(np.array(axes), (rows, cols), order="C")  # [::-1]
    elif ax_mode == "1d" and not tight:
        axes = np.reshape(np.array(axes), (rows * cols), order="C")

    if len(axes) == 1:
        return fig, axes[0]
    return fig, axes


@saveable()
def plot_output_grid(img: torch.Tensor,
                     output: torch.Tensor,
                     target: torch.Tensor,
                     grid: torch.Tensor,
                     grid_color: str = "g",
                     grid_scale_mode: Literal["like_image",
                                              "original"] = "original",
                     size: float = 5,
                     subsample: int = 25,
                     subsample_also_last: bool = True,
                     grid_linewidth: float = 1,
                     show_nav_frame: bool = True,
                     image_linewidth: float = 5,
                     tight: bool = False,
                     **kwargs) -> Figure:
    """Plots an image with the output and target mask, and a subsample grid.

    Parameters
    ----------
    img : torch.Tensor
        Image tensor
    output : torch.Tensor
        Output mask
    target : torch.Tensor
        Target mask
    grid : torch.Tensor
        Grid coordinates
    grid_scale_mode : Literal[&quot;like_image&quot;, &quot;original&quot;], optional
        Grid scale mode. If like_image then it will scale to image dimensions via minmax, if original it will keep its scaling
        , by default "original"
    size : float, optional
        Size in inches of the figure, by default 5
    subsample : int, optional
        Subsampling distance for the grid, by default 25
    subsample_also_last : bool, optional
        If also the last row col of the subsampled grid should be plotted, by default True
    grid_linewidth : float, optional
        Linewidth of the grid, by default 1
    show_nav_frame : bool, optional
        If a navigation frame should be plotted indicating transformations, by default True
    image_linewidth : float, optional
        Linewith of the markers in the imagefor nav frame, by default 5
    tight : bool, optional
        If plot should be tight, by default False

    Returns
    -------
    Figure
        Matplotlib figure

    """
    from matplotlib.colors import get_named_colors_mapping, to_rgba
    image_shape = grid.shape[-2:]

    frame_colors = ["tab:red", "tab:purple", "tab:cyan", "tab:olive"]
    cmap = get_named_colors_mapping()
    for idx, color in enumerate(frame_colors):
        if isinstance(color, str):
            frame_colors[idx] = to_rgba(cmap[color])

    dnorm_grid_pt = None
    if grid_scale_mode == "like_image":
        # Scale the grid so it will fit in the image
        dnorm_grid_pt = minmax(grid, torch.tensor(grid.numpy().min(axis=(-1, -2)))[:, None, None],
                               torch.tensor(grid.numpy().max(
                                   axis=(-1, -2)))[:, None, None], 0,
                               (torch.tensor(image_shape[::-1]) - 1)[:, None, None])
    elif grid_scale_mode == "original":
        # Like it where given but scaled from [0, 1] to image shape
        # We need to flip the image shape as grid comes in x,y and image dim is usually y,x
        dnorm_grid_pt = minmax(grid, torch.tensor(0), torch.tensor(
            1), 0, (torch.tensor(image_shape[::-1]) - 1)[:, None, None])
    else:
        raise ValueError(f"Unknown grid_scale_mode: {grid_scale_mode}")

    x_min = min(dnorm_grid_pt[0].min().item(), 0)
    x_max = max(dnorm_grid_pt[0].max().item(), img.shape[-1])
    y_min = min(dnorm_grid_pt[1].min().item(), 0)
    y_max = max(dnorm_grid_pt[1].max().item(), img.shape[-2])

    y_diff = y_max - y_min
    x_diff = x_max - x_min

    fig, ax = get_mpl_figure(1, 1, tight=tight, size=size,
                             ratio_or_img=(y_diff / x_diff))

    fig = plot_match(img, output, target, size=size, tight=tight,
                     subsample=subsample, subsample_grid_dots=False, ax=ax, **kwargs)

    if show_nav_frame:
        frame_nav = np.zeros((*image_shape, 4))
        # Order top, right, bottom, left
        frame_nav[:image_linewidth, :, :] = frame_colors[0]
        frame_nav[-image_linewidth:, :, :] = frame_colors[2]
        frame_nav[:, :image_linewidth, :] = frame_colors[3]
        frame_nav[:, -image_linewidth:, :] = frame_colors[1]
        ax.imshow(frame_nav)

    # dnorm_grid_pt = torch.clamp(torch.tensor(dnorm_grid_pt), min=torch.tensor([[[0]], [[0]]]), max=torch.tensor([[[image_shape[0] - 1]], [[image_shape[1] - 1]]])).numpy()

    msk = subsample_mask(dnorm_grid_pt, subsample=subsample,
                         also_last=subsample_also_last)
    fig = plot_grid(dnorm_grid_pt, msk, ax=ax, color=grid_color, origin="lower",
                    outer_line_colors=frame_colors, linewidth=grid_linewidth, **kwargs)

    ax.set_axis_on()
    # ax.axis("equal")
    ax.set_xlabel("Coordinates [x]")
    ax.set_ylabel("Coordinates [y]")

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_max, y_min)
    return fig


def purge_mask_with_no_overlap(masks: torch.Tensor, overlap_mask: torch.Tensor):
    if len(overlap_mask.shape) == 2:
        overlap_mask = overlap_mask.unsqueeze(0)
    masks = masks.bool()
    overlap_mask = overlap_mask.bool()
    overlap_mask = overlap_mask.expand_as(masks)
    overlap = masks & overlap_mask
    mask_sizes = overlap.sum(dim=(-2, -1))
    select_idx = torch.argwhere(mask_sizes > 0)

    mask_sizes_wo_zero = mask_sizes[select_idx[:, 0]]
    select_idx_sorted = select_idx[mask_sizes_wo_zero.argsort(descending=True)]
    return masks[select_idx_sorted[:, 0]]


def extract_automatic_masks_sam(image: VEC_TYPE, checkpoint_path: str) -> torch.Tensor:
    image = image.detach().cpu().permute(1, 2, 0).numpy(
    ) if not isinstance(image, np.ndarray) else image
    if image.dtype != np.uint8:
        image = (image * 255.).astype(np.uint8)

    from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator

    sam = sam_model_registry["default"](checkpoint=checkpoint_path)
    sam.eval()
    try:
        sam.to(torch.device("cuda"))
        mask_generator = SamAutomaticMaskGenerator(sam)
        with torch.no_grad():
            masks = mask_generator.generate(image)
    finally:
        sam.to(torch.device("cpu"))
        del sam
    seg_masks = [masks[i].get("segmentation") for i in range(len(masks))]
    stacked_masks = torch.from_numpy(
        np.stack(seg_masks, axis=-1)).permute(2, 0, 1).float()
    return stacked_masks


def get_cleaned_up_sam_masks(image: torch.Tensor, result_prior: torch.Tensor,
                             component_pixel_area_threshold: int = 30,
                             sam_checkpoint_path: Optional[str] = None) -> torch.Tensor:
    import cv2 as cv
    if sam_checkpoint_path is None:
        sam_checkpoint_path = "./data/checkpoints/sam/sam_vit_h_4b8939.pth"
    masks_sam = extract_automatic_masks_sam(image, sam_checkpoint_path)

    filtered_masks = purge_mask_with_no_overlap(masks_sam, result_prior)
    value_mask = channel_masks_to_value_mask(filtered_masks)
    dedublicate = torch.tensor(
        value_mask_to_channel_masks(value_mask)[0]).permute(2, 0, 1)
    # Ceck if there are areas which are not covered
    missing = dedublicate.sum(dim=(0, )) < 1

    # For all masks find the connected components.
    # If some connected component is smaller than some threshold, add it to the missing mask
    still_exists = torch.ones(dedublicate.shape[0], dtype=torch.bool)
    for i, msk in enumerate(dedublicate):
        n_components, labels, stats, centeroids = cv.connectedComponentsWithStats(
            (msk * 255).numpy().astype(np.uint8), connectivity=4)
        purged = []
        for c in range(n_components):
            purge = False
            c_mask = torch.from_numpy(labels == c)
            if stats[c, cv.CC_STAT_AREA] < component_pixel_area_threshold:
                purge = True
            if not (msk.bool() & c_mask).any():
                purge = True

            if purge:
                to_set = (msk.bool() & c_mask).bool()
                missing = (missing | to_set).bool()
                dedublicate[i][to_set] = False
                purged.append(purge)

        if len(purged) == n_components:
            still_exists[i] = False

    # Now we have a mask for all the missing pixels
    # Add it to the dedublicate mask
    dedublicate = torch.cat(
        [dedublicate[still_exists], missing[None, ...]], dim=0)
    return dedublicate


@saveable()
def plot_mask_multi_channel(
        image: VEC_TYPE,
        mask: VEC_TYPE,
        size: int = 5,
        title: str = None,
        tight: bool = False,
        background_value: int = 0,
        _colors: Optional[List[str]] = None,
        ax: Optional[Axes] = None,
        contour_linewidths: float = 2,
        darkening_background: float = 0.7,
        **kwargs) -> Figure:

    import matplotlib.patches as mpatches

    # Mask input should be C x H X W When tensor or H x W x C when np.ndarray
    mask = mask.detach().cpu().permute(1, 2, 0).numpy(
    ) if not isinstance(mask, np.ndarray) else mask
    image = image.detach().cpu().permute(1, 2, 0).numpy(
    ) if not isinstance(image, np.ndarray) else image

    # Check if mask contains multiple classes

    multi_class = (len(mask.shape) == 3) and len(mask[-1]) > 2

    cmap_name = 'Blues'

    fig = None

    if ax is None:
        fig, ax = get_mpl_figure(
            rows=1, cols=1, size=size, tight=tight, ratio_or_img=image)
    else:
        fig = ax.figure

    ax.imshow(image)

    alpha_map = register_alpha_map('binary')

    cmap = plt.get_cmap(alpha_map)

    color = "#5999cb"

    cmap = "tab10" if mask.shape[-1] <= 10 else "tab20"
    colors = [color] if not multi_class else plt.get_cmap(
        cmap)(range(mask.shape[-1]))

    m_inv = np.ones(mask.shape[:-1])
    non_background_mask = np.ones(mask.shape[:-1])

    for i in range(mask.shape[-1]):
        m = mask[..., i]
        non_background_mask = np.where(
            m != background_value, 0., non_background_mask)
        ax.contour(
            m_inv - m, levels=[0.5], colors=[colors[i]], linewidths=contour_linewidths)

    ax.imshow(non_background_mask, cmap='alpha_binary',
              alpha=darkening_background, label='')

    if not tight:
        ax.axis('off')

    if title is not None:
        fig.suptitle(title)

    if _colors is not None:
        _colors.clear()
        _colors.extend(colors)
    return fig


@saveable()
def plot_dense_image_mask(image: torch.Tensor,
                          mask: torch.Tensor,
                          object_ids: torch.Tensor = None,
                          tight: bool = True,
                          size: int = 5,
                          scale: float = 1.,
                          ax: Optional[Axes] = None,
                          cmap: Optional[str] = None,
                          legend: bool = True,
                          plot_image: bool = True,
                          background_value: int = 0,
                          ignore_value: Optional[int] = None,
                          **kwargs
                          ) -> Figure:
    """Plots image with the given masks in channel format (C x H x W)
    as dense masks, e.g. without contour lines.

    Parameters
    ----------
    image : torch.Tensor
        The image to plot the masks on.
    mask : torch.Tensor
        The masks where each channel represents a different object.
    object_ids : torch.Tensor, optional
        The object its which should be plotted, by default None
    tight : bool, optional
        If the plot should be tight, e.g. without border, by default True
    size : int, optional
        Size of the plot in inches, by default 5
    ax : Optional[Axes], optional
        Optional existing axes, by default None
    cmap : Optional[str], optional
        Optional cmap to use, by default None
        If None, tab10 or tab20 will be used depending on the number of objects.
    legend : bool, optional
        If a legend should be plotted, by default True
        The legend will show the object ids.
    plot_image : bool, optional
        If the image should be plotted, by default True
    background_value : int, optional
        The value of the background in the mask, by default 0
    Returns
    -------
    Figure
        The created figure.
    """
    from matplotlib.patches import Patch
    if isinstance(mask, torch.Tensor):
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)
    mask = mask.detach().cpu().permute(1, 2, 0).numpy(
    ) if isinstance(mask, torch.Tensor) else mask
    image = image.detach().cpu().permute(1, 2, 0).numpy(
    ) if isinstance(image, torch.Tensor) else image
    unique_object_ids = range(mask.shape[-1])
    if object_ids is not None:
        unique_object_ids = [x.item() for x in object_ids]
        assert len(unique_object_ids) == mask.shape[-1]

    if scale != 1.:
        image = interpolate_image(image, scale)
        mask = interpolate_image(mask, scale)

    if cmap is None:
        if len(unique_object_ids) <= 10:
            cmap = plt.get_cmap("tab10")
        else:
            cmap = plt.get_cmap("tab20")
    elif isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)

    fig = None
    if ax is None:
        if tight:
            sizes = np.shape(mask)
            fig = plt.figure(figsize=(size, size))
            dpi = 300
            fig.set_size_inches(
                size * (sizes[1] / dpi), size * (sizes[0] / dpi), forward=False)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
        else:
            fig, ax = plt.subplots(1, 1, figsize=(size, size))
    else:
        fig = ax.figure

    if plot_image:
        ax.imshow(image)

    object_colors = dict()
    any_fg_mask = np.zeros(mask.shape[:-1], dtype=np.bool_)
    for i, object_id in enumerate(unique_object_ids):
        if mask[..., i].sum() == 0:
            # Empty channel
            continue
        color = np.array(cmap(i % cmap.N))
        color_mask = np.zeros(mask.shape[:-1] + (4,))
        color_mask[:, :, :] = color
        fg = (mask[..., i] != background_value)
        if ignore_value is not None:
            fg = np.logical_and(fg, mask[..., i] != ignore_value)
        any_fg_mask = np.logical_or(any_fg_mask, fg)
        color_mask[:, :, 3] = fg.astype(float)
        object_colors[object_id] = color
        ax.imshow(color_mask, alpha=1)

    any_fg_mask = np.clip(np.sum(np.where(
        mask != background_value, 1, 0), axis=-1), 0, 1)  # True if not background
    background_mask = np.logical_not(any_fg_mask).astype(float)

    ax.imshow(background_mask, cmap='alpha_binary', alpha=0.7, label='')

    if legend:
        patches = [Patch(color=v, label=str(k))
                   for k, v in object_colors.items()]
        ax.legend(handles=patches)
    return fig


def channel_masks_to_value_mask(masks: VEC_TYPE,
                                object_values: Optional[VEC_TYPE] = None,
                                handle_overlap: Literal['raise', 'ignore',
                                                        'warning', 'warning+exclude'] = 'warning',
                                base_value: Any = 0.
                                ) -> np.ndarray:
    """Converts a list of channel masks to a single mask with a new value per mask object.

    Parameters
    ----------
    masks : VEC_TYPE
        List of channel masks of shape C x H x W or H x W x C (resp. torch.Tensor or np.ndarray)

    object_values : Optional[VEC_TYPE], optional
        The object values to assign to the mask, by default 1, 2, 3, ...
        These values will be used within the mask to identify the object.
        Should be of shape (C, ) where C is the number of masks.

    Returns
    -------
    VEC_TYPE
        Single mask with multiple channels, where each number represents a different object.
        Shape is H x W
    """
    masks = masks.detach().cpu().permute((1, 2, 0)).numpy(
    ) if isinstance(masks, torch.Tensor) else masks
    object_values = object_values.detach().cpu().numpy() if isinstance(
        object_values, torch.Tensor) else object_values

    if object_values is None:
        object_values = np.arange(1, masks.shape[-1] + 1)
    else:
        if object_values.shape != (masks.shape[-1],):
            raise ValueError(
                f"Object values shape {object_values.shape} does not match number of masks {masks.shape[-1]}")
        if np.unique(object_values).shape != object_values.shape:
            raise ValueError(
                f"Object values must be unique, got {object_values}")

    mask = np.zeros(masks.shape[:-1], dtype=masks.dtype)
    mask.fill(base_value)
    for i in range(masks.shape[-1]):
        fill = masks[..., i] > 0

        if mask[fill].sum() != 0:
            # Overlap in classes.
            if handle_overlap == 'ignore':
                pass
            else:
                overlap_classes = ', '.join(
                    [str(x) for x in np.unique(mask[fill]).astype(int).tolist() if x != 0])
                if handle_overlap == 'raise':
                    raise ValueError(
                        f"Overlap in classes detected, class {object_values[i]} overlaps with class(es) {overlap_classes}")
                elif handle_overlap == 'warning':
                    logging.warning(
                        f"Overlap in classes detected, class {object_values[i]} overlaps with class(es) {overlap_classes}")
                elif handle_overlap == 'warning+exclude':
                    logging.warning(
                        f"Overlap in classes detected, class {object_values[i]} overlaps with class(es) {overlap_classes}, excluding it")
                    duplicate_class = (mask != 0) & fill
                    fill = fill & ~duplicate_class
                    mask[duplicate_class] = 0.
                    logging.warning(f"Excluded {duplicate_class.sum()} pixels")
                else:
                    raise ValueError(
                        f"Unknown overlap handling {handle_overlap}")
        mask = np.where(fill, object_values[i], mask)
    return mask


@saveable()
def plot_as_image(data: torch.Tensor,
                  size: float = 5,
                  variable_name: str = "Image",
                  cscale: Optional[Union[List[str], str]] = None,
                  ticks: bool = True,
                  title: Optional[str] = None,
                  colorbar: bool = False,
                  colorbar_tick_format: str = None,
                  axes: Optional[np.ndarray] = None) -> AxesImage:
    import itertools
    from matplotlib.axes import Subplot

    if data.device != torch.device("cpu"):
        data = data.detach().cpu()

    rows = 1
    cols = 1

    images = []
    _img_title = []
    cmap = []

    if 'complex' in str(data.dtype):
        cols = 2
        if data.shape[0] in [1, 3]:
            data = data.permute(1, 2, 0)
        _img_title.append(f"|{variable_name}|")
        images.append(torch.abs(data))
        cmap.append("gray")

        _img_title.append(f"angle({variable_name})")
        images.append(torch.angle(data))
        cmap.append("twilight")
    else:
        _img_title.append("Image")
        if data.shape[0] in [1, 3]:
            data = data.permute(1, 2, 0)
        images.append(data)
        cmap.append("gray")

    if axes is None:
        fig, axes = plt.subplots(
            rows, cols, figsize=(size * cols, size * rows))
    else:
        fig = plt.gcf()

    if isinstance(axes, Subplot):
        axes = [axes]

    for i, ax in enumerate(itertools.chain(axes)):
        _image = images[i]
        _title = _img_title[i]
        if cscale is not None:
            _cscale = cscale
            if isinstance(cscale, list):
                _cscale = cscale[i]
            if _cscale == 'auto':
                _cscale = 'log' if should_use_logarithm(_image) else None
            if _cscale is not None:
                if _cscale == 'log':
                    _image = np.log(_image)
                    _title = f"log({_title})"
        ax.imshow(_image, cmap=cmap[i])
        ax.set_title(_title)
        if colorbar:
            _cbar_format = None
            if colorbar_tick_format is not None:
                cft = ('{:' + colorbar_tick_format + '}')

                def _cbar_format(x, pos):
                    return cft.format(x)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(ax.get_images()[0], cax=cax,
                         format=_cbar_format, orientation='vertical')
        if not ticks:
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
    if title is not None:
        fig.suptitle(title)
    return fig


def value_mask_to_channel_masks(
    mask: VEC_TYPE,
    ignore_value: Optional[Union[int, List[int]]] = None,
    background_value: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Converts a value mask, where objects are identified as different values, to a channel mask, where each channel represents a different object.

    Parameters
    ----------
    mask : VEC_TYPE
        The mask as a value mask, e.g. where each value represents a different object.
        Should be of shape C x H x W or H x W x C (resp. torch.Tensor or np.ndarray)
    ignore_value : Optional[Union[int, List[int]]], optional
        Values to ignore when creating the mask, by default None
    background_value : int, optional
        Value which is treated as the background value, by default 0

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        1. The channel mask of shape H x W x C
        2. The object values of shape (C, ) (as they where in mask) corresponding to the channel mask index
    """
    if isinstance(mask, torch.Tensor) and len(mask.shape) == 2:
        mask = mask[None, ...]
    mask = mask.detach().cpu().permute((1, 2, 0)).numpy(
    ) if isinstance(mask, torch.Tensor) else mask
    mask = mask.squeeze()
    if len(mask.shape) > 2:
        raise ValueError(f"Value-Mask should be 2D, got {mask.shape}")
    invalid_values = set([background_value])
    if ignore_value is not None:
        if isinstance(ignore_value, int):
            invalid_values.add(ignore_value)
        else:
            invalid_values.update(ignore_value)
    vals = np.unique(mask)
    _valid_classes = np.stack([x for x in vals if x not in invalid_values])
    channel_mask = np.zeros(mask.shape + (len(_valid_classes),))
    for i, c in enumerate(_valid_classes):
        channel_mask[..., i] = (mask == c)
    return channel_mask, _valid_classes


def load_image(image_path: str) -> torch.Tensor:
    from PIL import Image
    img_pil = Image.open(image_path)
    img = np.array(img_pil, dtype='float')/255.0
    if len(img.shape) == 3:
        img = img[:, :, 0:3]
    else:
        img = img[:, :, None]
    return torch.from_numpy(img.transpose(2, 0, 1))


def load_mask_multi_channel(mask_path: str,
                            ignore_value: Optional[Union[int,
                                                         List[int]]] = None,
                            background_value: int = 0,
                            ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads a mask from the given path and converts it to a channel mask, where each channel represents a different object.

    Parameters
    ----------
    mask : VEC_TYPE
        The mask as a value mask, e.g. where each value represents a different object.
        Should be of shape C x H x W or H x W x C (resp. torch.Tensor or np.ndarray)
    ignore_value : Optional[Union[int, List[int]]], optional
        Values to ignore when creating the mask, by default None
    background_value : int, optional
        Value which is treated as the background value, by default 0

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        1. The channel mask of shape H x W x C
        2. The object values of shape (C, ) (as they where in mask) corresponding to the channel mask index
    """
    from PIL import Image
    img_pil = Image.open(mask_path)
    img = np.array(img_pil)
    if len(img.shape) == 3 and img.shape[-1] >= 1:
        raise ValueError("Mask should be single channel")
    return value_mask_to_channel_masks(img, ignore_value, background_value)


def load_mask_single_channel(mask_path: str) -> np.ndarray:
    from PIL import Image
    img_pil = Image.open(mask_path)
    img = np.array(img_pil)
    if len(img.shape) == 3 and img.shape[-1] >= 1:
        raise ValueError("Mask should be single channel")
    return img


def draw_text(img, text,
              font=cv2.FONT_HERSHEY_PLAIN,
              pos=(0, 0),
              font_scale=3,
              font_thickness=2,
              text_color=(0, 255, 0),
              text_color_bg=(0, 0, 0),
              padding=5
              ):

    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, pos, (x + text_w + 2 * padding, y +
                  text_h + 2 * padding), text_color_bg, -1)
    cv2.putText(img, text, (x + padding, y + padding + text_h + font_scale - 1),
                font, font_scale, text_color, font_thickness)

    return text_size


@channelize()
def interpolate_image(mask_or_img: np.ndarray, scale: int = 1) -> np.ndarray:
    if scale == 1:
        return mask_or_img
    scaled = cv2.resize(mask_or_img, (0, 0), fx=scale,
                        fy=scale, interpolation=cv2.INTER_NEAREST)
    return scaled


@saveable()
def plot_mask_labels(
        image: VEC_TYPE,
        mask: VEC_TYPE,
        boxes: VEC_TYPE = None,
        labels: VEC_TYPE = None,
        scores: VEC_TYPE = None,
        background_value: int = 0,
        ignore_class: Optional[int] = None,
        size: int = 5,
        scale: float = 1,
        title: str = None,
        tight: bool = False,
        _colors: Optional[List[str]] = None,
        ax: Optional[Axes] = None,
        box_linetype: int = cv2.LINE_AA,
        **kwargs) -> Figure:
    from matplotlib.colors import to_rgba
    image = image.detach().cpu().permute(1, 2, 0).numpy(
    ) if not isinstance(image, np.ndarray) else image
    # Mask input should be C x H X W When tensor or H x W x C when np.ndarray
    mask = mask.detach().cpu().numpy() if not isinstance(mask, np.ndarray) else mask
    if len(mask.shape) == 3:
        mask = mask.transpose((1, 2, 0))

    if scale != 1.:
        image = interpolate_image(image, scale)

        reshaped = False
        if len(mask.shape) == 2:
            mask = mask[..., None]
            reshaped = True
        mask = interpolate_image(mask, scale)
        if reshaped and len(mask.shape) == 3:
            mask = mask[..., 0]

    if _colors is None:
        _colors = list()
    fig = plot_mask(image, mask, size, title, tight,
                    background_value=background_value,
                    ignore_class=ignore_class,
                    _colors=_colors,
                    contour_linewidths=scale * 2,
                    ax=ax,
                    )
    ax = fig.axes[0]

    if boxes is not None:
        boxes = boxes.detach().cpu().numpy() if isinstance(boxes, torch.Tensor) else boxes
        labels = labels.detach().cpu().numpy() if isinstance(
            labels, torch.Tensor) else labels

        if scale != 1.:
            boxes = boxes * scale

        boxes_img = np.zeros(image.shape[:2] + (4, ))
        rect_th = 1 * scale
        text_th = 2
        text_size = 1

        for i in range(boxes.shape[0]):
            box = [int(round(x)) for x in boxes[i]]
            if i < len(_colors):
                color = to_rgba(_colors[i], 1)
            else:
                color = to_rgba((1, 1, 1), 1)
            cv2.rectangle(boxes_img, (box[0], box[1]), (box[2], box[3]),
                          color=color, thickness=rect_th, lineType=box_linetype)
            # cv2.putText(boxes_img, labels[i], (box[0], box[3] - 3), cv2.FONT_HERSHEY_SIMPLEX, text_size, color, thickness=text_th)

            text = str(labels[i])
            if scores is not None:
                text += f" {scores[i]:.2f}"

            draw_text(boxes_img, text=text, pos=(box[0], box[1]),
                      font=cv2.FONT_HERSHEY_SIMPLEX,
                      font_scale=text_size,
                      text_color=(0, 0, 0, 1),
                      text_color_bg=color,
                      font_thickness=text_th)
        ax.imshow(boxes_img, alpha=1., zorder=3)
    elif labels is not None:
        from matplotlib.patches import Patch
        labels = labels.detach().cpu().numpy() if isinstance(
            labels, torch.Tensor) else labels

        label_colors = zip(labels, _colors)

        patches = [Patch(color=v, label=k) for k, v in label_colors]
        ax.legend(handles=patches)
    return fig


@saveable()
def plot_as_image(data: VEC_TYPE,
                  size: float = 5,
                  variable_name: str = "Image",
                  cscale: Optional[Union[List[str], str]] = None,
                  ticks: bool = True,
                  title: Optional[str] = None,
                  colorbar: bool = False,
                  colorbar_tick_format: str = None,
                  value_legend: bool = False,
                  cmap: Optional[str] = None,
                  axes: Optional[np.ndarray] = None,
                  interpolation: Optional[str] = None,
                  tight: bool = False,
                  ) -> AxesImage:
    import itertools
    from matplotlib.axes import Subplot
    import matplotlib as mpl

    if isinstance(data, torch.Tensor):
        data = data.detach()
        if data.device != torch.device("cpu"):
            data = data.detach().cpu()
        if data.shape[0] in [1, 3]:
            data = data.permute(1, 2, 0)
        data = data.numpy()

    rows = 1
    cols = 1

    images = []
    _img_title = []
    cmaps = []

    if 'complex' in str(data.dtype):
        cols = 2
        _img_title.append(f"|{variable_name}|")
        images.append(torch.abs(data))
        cmaps.append("gray")

        _img_title.append(f"angle({variable_name})")
        images.append(torch.angle(data))
        cmaps.append("twilight")
    else:
        _img_title.append(variable_name)
        images.append(data)
        if cmaps is None:
            cmaps.append("gray")
        else:
            cmaps.append(cmap)

    if axes is None:
        fig, axes = get_mpl_figure(
            rows=rows, cols=cols, size=size, tight=tight, ratio_or_img=images[0])
    else:
        fig = plt.gcf()

    if isinstance(axes, Subplot):
        axes = [axes]

    for i, ax in enumerate(itertools.chain(axes)):
        _image = images[i]
        _title = _img_title[i]

        color_mapping = None

        vmin = _image.min()
        vmax = _image.max()
        _cmap = cmaps[i]
        if isinstance(_cmap, str):
            _cmap = plt.get_cmap(_cmap)

        if cscale is not None:
            _cscale = cscale
            if isinstance(cscale, list):
                _cscale = cscale[i]
            if _cscale == 'auto':
                _cscale = 'log' if should_use_logarithm(
                    _image.numpy()) else None
            if _cscale is not None:
                if _cscale == 'log':
                    _image = np.log(_image)
                    _title = f"log({_title})"
            if _cscale == "count":
                color_mapping = dict()
                for j, value in enumerate(np.unique(_image)):
                    color_mapping[j] = value
                    _image = np.where(_image == value, j, _image)

        if isinstance(_cmap, ListedColormap):
            vmax = len(_cmap.colors) - 1
            vmin = 0

        ax.imshow(_image, vmin=vmin, vmax=vmax,
                  cmap=_cmap, interpolation=interpolation)

        if not tight:
            ax.set_title(_title)
        if colorbar:
            _cbar_format = None
            if colorbar_tick_format is not None:
                cft = ('{:' + colorbar_tick_format + '}')

                def _cbar_format(x, pos):
                    return cft.format(x)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(ax.get_images()[0], cax=cax,
                         format=_cbar_format, orientation='vertical')

        if not ticks:
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])

        if value_legend:
            unique_vals = np.unique(_image)
            patches = []
            if isinstance(_cmap, str):
                _cmap = plt.get_cmap(_cmap)

            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
            for i, value in enumerate(unique_vals):
                c = _cmap(norm(value))
                if color_mapping is not None:
                    value = color_mapping[value]
                patches.append(Patch(color=c, label=f"{value:n}"))
            ax.legend(handles=patches)

    if title is not None:
        fig.suptitle(title)
    return fig


# fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
def colorFader(start, end, mix=0):
    start = np.array(mpl.colors.to_rgb(start))
    end = np.array(mpl.colors.to_rgb(end))
    return mpl.colors.to_hex((1-mix)*start + mix*end)


def gradient_end_transparent_map(start_color, end_color, name: str, alpha: float = 1) -> str:
    start_color = to_hex(start_color)

    end_color = to_hex(end_color)

    # start_color = Color(start_color)
    # get colormap
    ncolors = 256
    # colors = list([x.get_rgb() for x in start_color.range_to(Color(end_color), ncolors)])
    colors = [to_rgb(colorFader(start_color, end_color, x / (ncolors-1)))
              for x in range(ncolors)]
    color_array = np.array(colors)

    rgba_array = np.ones(color_array.shape[:1] + (4,))
    rgba_array[..., :3] = color_array

    # color_array[..., -1] = 1.

    # First and last should be fully transparent
    rgba_array[0, -1] = 0
    rgba_array[1, :3] = rgba_array[0, :3]

    rgba_array[-1, -1] = 0

    rgba_array[-2, :3] = rgba_array[-1, :3]

    rgba_array[:, -1] = rgba_array[:, -1] * alpha

    # create a colormap object
    map_object = LinearSegmentedColormap.from_list(
        colors=rgba_array, name=name)
    return map_object


def color_from_cmap(arr, cmap, ):
    _arr = np.where(np.isnan(arr), 0., arr)
    _non_nan_arr = minmax(_arr, _arr.min(), _arr.max(), 0, 1)
    colors = cmap(range(cmap.N))
    _color_arr = np.round(_non_nan_arr * (len(colors) - 1)).astype(int)
    # _color_arr = np.clip(_color_arr, 0, len(colors)-1)
    _colors_in_arr = colors[_color_arr]
    _colors_in_arr[np.isnan(arr), ...] = 0.
    return _colors_in_arr


def figure_to_numpy(fig: Figure, dpi: int = 300, transparent: bool = True) -> np.ndarray:
    """Converts a matplotlib figure to a numpy array.

    Parameters
    ----------
    fig : Figure
        The figure to convert

    dpi : int, optional
        Dots per inch, by default 72

    Returns
    -------
    np.ndarray
        The figure as a numpy array
    """
    import io
    arr = None
    if fig.dpi != dpi:
        fig.set_dpi(dpi)
    with io.BytesIO() as io_buf:
        fig.savefig(io_buf, format='raw', transparent=transparent, dpi=fig.dpi)
        io_buf.seek(0)
        arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                         newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
    return arr


@saveable()
def plot_surface_logits(
        image: torch.Tensor,
        logits: torch.Tensor,
        foreground_scribble_mask: torch.Tensor,
        background_scribble_mask: torch.Tensor,
        ax: Optional[Axes] = None,
        color_fg: Optional[Any] = None,
        color_bg: Optional[Any] = None,
        color_seg: Optional[Any] = None,
        size: float = 5,
        tight: bool = False,
        elevation: float = 30.,
        azimuth: float = 270.,
        zoom: float = 1,
        image_subsampling: Optional[int] = None,
        surface_log: bool = False,
        surface_log_eps: float = 1e-1,
):
    fig = None

    # Number of pixels in x and y direction
    img_rcount = 50
    img_ccount = 50

    if image_subsampling is not None:
        img_rcount = int(image.shape[-2] / image_subsampling)
        img_ccount = int(image.shape[-1] / image_subsampling)

    if ax is None:
        fig, ax = get_mpl_figure(
            rows=1, cols=1, size=size, tight=tight, subplot_kw=dict(projection='3d'))
    else:
        fig = ax.get_figure()

    if color_fg is None:
        color_fg = plt.get_cmap("tab10").colors[2]
    if color_bg is None:
        color_bg = plt.get_cmap("tab10").colors[3]
    if color_seg is None:
        color_seg = plt.get_cmap("tab10").colors[0]

    x = np.arange(image.shape[-1], 0, -1)
    y = np.arange(image.shape[-2], 0, -1)

    # Normalize
    x = x / x.max()
    y = y / y.max() * image.shape[-2] / image.shape[-1]

    xx, yy = np.meshgrid(x, y)

    rgb_img = image.permute(1, 2, 0).numpy()
    ax.plot_surface(xx, yy, np.zeros_like(xx), facecolors=rgb_img,
                    rcount=img_rcount, ccount=img_ccount)

    prior = logits.squeeze()
    prior = np.where((logits != 0.) & (logits != 1.), prior, np.nan)[0]

    cmap = gradient_end_transparent_map(color_seg,
                                        color_bg,
                                        "Segmenation Grad",
                                        alpha=0.2)

    scribble_fg = foreground_scribble_mask.squeeze().numpy()
    colors_fg_scribble = np.zeros(xx.shape + (4,))
    colors_fg_scribble[..., 0:3] = color_fg
    colors_fg_scribble[..., 3] = scribble_fg
    ax.plot_surface(xx, yy, np.zeros_like(xx), facecolors=colors_fg_scribble,
                    rcount=img_rcount, ccount=img_ccount, shade=True)

    scribble_bg = background_scribble_mask.squeeze().numpy()
    colors_bg_scribble = np.zeros(xx.shape + (4,))
    colors_bg_scribble[..., 0:3] = color_bg
    colors_bg_scribble[..., 3] = scribble_bg * 1
    ax.plot_surface(xx, yy, np.zeros_like(xx), facecolors=colors_bg_scribble,
                    rcount=img_rcount, ccount=img_ccount, shade=True)

    x = np.arange(prior.shape[-1], 0, -1)
    y = np.arange(prior.shape[-2], 0, -1)
    x = x / x.max()
    y = y / y.max() * (prior.shape[-2] / prior.shape[-1])
    xx, yy = np.meshgrid(x, y)

    rcount, ccount = prior.shape

    norm_prior = minmax(prior, prior.min(), prior.max(), 0, 1)

    color_prior = prior

    if surface_log:
        should_nan = prior < 0.0
        color_prior = np.log10(minmax(color_prior, color_prior.min(
        ), color_prior.max(), 0 + surface_log_eps, 1 + surface_log_eps))
        color_prior[should_nan] = np.nan
    else:
        is_zero = (prior < 0.0)
        color_prior[is_zero] = np.nan

    surf = ax.plot_surface(xx, yy,
                           norm_prior,
                           rcount=rcount,  # rcount // 4,
                           ccount=ccount,  # ccount // 4,
                           facecolors=color_from_cmap(color_prior, cmap),
                           shade=False,
                           # alpha=0.3
                           )

    surf.set_facecolor((0, 0, 0, 0))

    # ax.set_xlabel("x")
    # ax.set_ylabel("y")

    # plt.tick_params(left = False, right = False , labelleft = False ,
    #            labelbottom = False, bottom = False)

    x_left, x_right = ax.get_xlim()
    y_low, y_high = ax.get_ylim()

    x_pos = [i._x for i in ax.get_xticklabels() if i._x <= prior.shape[1]]
    x_lab = [round((i / prior.shape[1]), 2) for i in x_pos]
    x_lab = ["" for x in x_lab]
    ax.set_xticks(x_pos, x_lab)

    y_pos = [i._y for i in ax.get_yticklabels() if i._y <= prior.shape[0]]
    y_lab = [round((i / prior.shape[0]), 2) for i in y_pos]
    y_lab = ["" for x in y_lab]
    ax.set_yticks(y_pos, y_lab)

    z_pos = [i._y for i in ax.get_zticklabels()]
    # z_lab = [round((i / prior.shape[0]), 2) for i in z_pos]
    z_lab = ["" for x in z_pos]
    ax.set_zticks(z_pos, z_lab)

    ax.set_box_aspect(
        aspect=((x_right-x_left)/(y_low-y_high), 1, 1), zoom=zoom)
    ax.scatter(xx, yy, np.zeros_like(xx), color="black", zorder=-1, alpha=0.0)
    # ax.grid("off")
    ax.axis("off")
    ax.view_init(elev=elevation, azim=azimuth)

    return fig


def prepare_input_eval(dataloader: Any, model: Any = None, index: int = 0):
    patch_args = dict()
    _input = None
    img = None
    gt = None
    prior = None
    fg, bg = None, None  # Foreground and background scribbles
    target = None
    if isinstance(dataloader, ConvexitySegmentationDataset):
        patch_args['output_mode'] = OutputMode.IMAGE
        patch_args['return_prior'] = True

    if isinstance(dataloader, SISBOSIDataset):
        patch_args['mode'] = "sample"

    if isinstance(dataloader, AwesomeDataset):
        patch_args['mode'] = "sample"

    with TemporaryProperty(dataloader, **patch_args), torch.no_grad():
        if isinstance(dataloader, ConvexitySegmentationDataset):
            prior_state, ((img, fg, bg), gt) = dataloader[index]
            _input = ConvexitySegmentationDataset.get_important_pixels(
                img, None, None)[None, ...]  # Adding batch dimension
        elif isinstance(dataloader, SISBOSIDataset):
            prior_state, data = dataloader[index]
            _input = (data['rgb'], data['xy'], data['xy_clean'])
            # Adding batch dimension to input
            img = data['image']
            gt = data['gt']
            mask = data['mask']
            fg = torch.zeros_like(mask)
            fg[mask == 0.] = 1.
            bg = torch.zeros_like(mask)
            bg[mask == 1.] = 1.
        elif isinstance(dataloader, AwesomeDataset):
            _item = dataloader[index]
            if dataloader.has_prior:
                prior_state = _item[0]
                sample = _item[1]
            else:
                prior_state = None
                sample = _item
            with TemporaryProperty(dataloader, mode="model_input", return_prior=False):
                _input, target = dataloader[index]
            ret = dataloader.get_dimension_based_data(sample)
            ground_truth = ret['labels']
            img = sample['image']
            if ground_truth is not None:
                # No ground truth, so we can't evaluate
                gt = dataloader.encode_target(
                    ground_truth, index, sample=sample)
                # fg = torch.clip(gt.sum(dim=0), min=0, max=1).float()
                # bg = torch.logical_not(fg).float()
            if dataloader.supervision_mode == "full":
                fg = torch.zeros_like(img[0])
                bg = torch.zeros_like(img[0])
            elif dataloader.supervision_mode == "weakly":
                fg = torch.zeros_like(img[0])
                bg = torch.zeros_like(img[0])
            wk = sample["scribble"]
            fg[wk == 0] = 1.
            bg[wk == 1.] = 1.
        else:
            raise ValueError(f"Unknown dataset type: {type(dataloader)}")
    return img, gt, _input, target, fg, bg, prior_state


def get_result(
        model: torch.nn.Module,
        dataloader: Any,
        index: int,
        model_gets_targets: bool,
        raise_on_missing_ground_truth: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = next(model.parameters())[1].device

    image, ground_truth, _input, targets, fg, bg, prior_state = prepare_input_eval(
        dataloader, model, index)
    if ground_truth is None and raise_on_missing_ground_truth:
        raise MissingGroundTruthError(
            "No ground truth available, can't evaluate")
    was_training = model.training
    with torch.no_grad():
        res = None
        model.train(False)
        # Adding batch dimension as this would also do the batch composition
        _input_d = TensorUtil.apply_deep(
            _input, lambda x: x[None, ...].to(device=device))
        with PriorManager(model, prior_state, getattr(dataloader, "__prior_cache__", None), training=False) as prior_manager:
            model_kwargs = {}
            if model_gets_targets:
                targets = TensorUtil.apply_deep(
                    targets, lambda x: x[None, ...].to(device=device))
                model_kwargs['targets'] = targets

            if isinstance(_input_d, (tuple, list)):
                res = model(*_input_d, **model_kwargs)
            else:
                res = model(_input_d, **model_kwargs)
            res = TensorUtil.apply_deep(res, lambda x: x.detach().cpu())
    # If dataloader has image_channel format
    if hasattr(dataloader, "image_channel_format"):
        # Out put as
        fmt = dataloader.image_channel_format
        if fmt == "bgr":
            image = image[[2, 1, 0], ...]
    model.train(was_training)
    return res, ground_truth, image, fg, bg


def get_prior_result(model: torch.nn.Module,
                     input: torch.Tensor,
                     batch_size: int = 1,
                     device: Optional[torch.device] = None) -> torch.Tensor:
    from torch.utils.data import TensorDataset, DataLoader
    from tqdm.auto import tqdm
    dataset = TensorDataset(input)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    was_training = model.training
    old_device = next(model.parameters())[1].device
    out = []
    try:
        model.train(False)
        model.to(device=device)
        bar = tqdm(total=len(dataloader), desc="Producing Prior images")
        with torch.no_grad():
            res = None
            for i, x in enumerate(dataloader):
                x_d = x[0].to(device=device)
                res = model(x_d).detach().cpu()
                out.append(res)
                bar.update()
            bar.close()
    finally:
        model.train(was_training)
        model.to(device=old_device)
    return torch.cat(out, dim=0)


def create_grid_verticies(grid: np.ndarray, z_loc: float):
    indices = np.argwhere(grid.T >= 0)
    repeated = indices[:, None, :].repeat(4, axis=1)
    repeated[:, 1, 1] += 1  # Added one to x
    repeated[:, 2, :] += 1  # Added one to x and y
    repeated[:, 3, 0] += 1  # Added one to y
    # Add constant z
    zz = np.zeros_like(repeated[:, :, 0])[..., None]
    zz.fill(z_loc)
    vertices_img = np.concatenate([repeated, zz], axis=-1)
    return vertices_img


@saveable()
def plot_3d_tubes(logits,
                  images,
                  subsample_factor: int = 6,
                  subsample_image_mode: str = "grid_sample",
                  grid_sample_mode: str = "bilinear",
                  size: float = 5,
                  tube_facecolor: Optional[Any] = None,
                  top_image_alpha: float = 0,
                  z_ticks_max: Optional[int] = None,
                  z_ticks_min: Optional[int] = None,
                  z_ticks_step: Optional[int] = None,
                  ):
    import math
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from matplotlib.colors import to_rgba
    from skimage import measure

    elev = 40
    azim = 90

    priors_no_sig = logits

    vol = priors_no_sig[:, 0, ::subsample_factor,
                        ::subsample_factor].cpu().numpy()

    # Use marching cubes to obtain the surface mesh of these ellipsoids
    verts, faces, normals, values = measure.marching_cubes(vol.T, 0)

    # Display resulting triangular mesh using Matplotlib. This can also be done
    # with mayavi (see skimage.measure.marching_cubes docstring).
    fig, ax = get_mpl_figure(1, 1, size=size, subplot_kw=dict(projection='3d'))

    image_index = len(images) - 1
    rgb_img = image_subsample(images[image_index], factor=subsample_factor,
                              mode=subsample_image_mode, grid_sample_mode=grid_sample_mode).permute(1, 2, 0).numpy()

    x = np.arange(0, rgb_img.shape[-2], 1)
    y = np.arange(0, rgb_img.shape[-3], 1)
    xx, yy = np.meshgrid(x, y)

    if tube_facecolor is None:
        tube_facecolor = plt.get_cmap("tab10")(1)

    ls = LightSource(azdeg=azim, altdeg=elev)

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], lightsource=ls,
                            shade=True, facecolors=tube_facecolor, edgecolor=None, linewidth=0, antialiased=False)
    ax.add_collection3d(mesh)

    fcol = rgb_img.reshape((math.prod(rgb_img.shape[:-1]), 3), order="F")
    secmesh = Poly3DCollection(create_grid_verticies(xx, image_index),
                               lightsource=ls,
                               shade=False,
                               facecolors=fcol, edgecolor=None, linewidth=0, antialiased=True)
    ax.add_collection3d(secmesh)

    front_index = 0
    front_img = image_subsample(images[front_index], factor=subsample_factor,
                                mode=subsample_image_mode, grid_sample_mode=grid_sample_mode).permute(1, 2, 0).numpy()

    fcol_front = front_img.reshape(
        (math.prod(front_img.shape[:-1]), 3), order="F")
    # Stack alpha
    alpha_fcol = np.zeros_like(fcol_front[:, 0])[..., None]
    alpha_fcol.fill(top_image_alpha)
    fcol_front = np.concatenate([fcol_front, alpha_fcol], axis=-1)

    # Set alpha to 0 for all pixels that are not in the foreground
    subsample_df_prior = priors_no_sig[front_index, :,
                                       ::subsample_factor, ::subsample_factor][0].numpy()
    fg = (subsample_df_prior <= 0).reshape(
        (math.prod(subsample_df_prior.shape), 1), order="F")
    fcol_front[fg[:, 0], 3] = 1

    thirdmesh = Poly3DCollection(create_grid_verticies(xx, front_index), lightsource=ls,
                                 shade=False, facecolors=fcol_front, edgecolor=None, linewidth=0, antialiased=False)
    ax.add_collection3d(thirdmesh)

    ax.set_xlim(0, vol.shape[-1])
    ax.set_ylim(0, vol.shape[-2])
    ax.set_zlim(0, vol.shape[-3])

    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    ax.set_zlabel('Time [t]')

    ax.view_init(elev=elev, azim=azim, roll=0)

    ax.invert_xaxis()
    ax.invert_zaxis()

    ax.grid(False)
    ax.xaxis.line.set_lw(0.)
    ax.set_xticks([])

    ax.yaxis.line.set_lw(0.)
    ax.set_yticks([])

    if z_ticks_step is None:
        z_ticks_step = 5
    if z_ticks_max is None:
        z_ticks_max = len(images)
    if z_ticks_min is None:
        z_ticks_min = 0

    t = np.arange(0, len(logits), 1)
    l = np.arange(z_ticks_min, z_ticks_max, z_ticks_step)

    dt = minmax(l, z_ticks_min, l.max(), t.min(), t.max())

    ax.set_zticks(dt, labels=l.astype(int))

    ax.set_aspect('equalxy')

    return fig


def save_result_mask(mask: torch.Tensor, path: str, invert: bool = True) -> None:
    """Saves the given mask to the given path.
    Mask should be of shape C x H x W or H x W (resp. torch.Tensor or np.ndarray)
    Each channel should represent a different object.
    These channels will be encoded into a color mask where each color represents a different object.

    Should have only binary values. If values are floats, an error is raised.

    Parameters
    ----------
    mask : torch.Tensor
        Channel mask to save.
    path : str
        Path to save the mask to.
    invert : bool, optional
        If the mask should be inverted, by default True
        Will flip 0 and 1, so 0 is background and 1 is foreground.
        E.g. If it foreground was 0 before, it will be 1 now.
    """
    import cv2
    if len(mask.shape) == 2:
        mask = mask[None, ...]
    mask = mask.detach().cpu().permute(1, 2, 0).numpy()
    # Invert mask
    if invert:
        mask = np.logical_not(mask)  # Now 0 is background and 1 is foreground

    combined_mask = np.zeros(mask.shape[:2], dtype=np.uint8)

    colors = list(range(1, mask.shape[-1] + 1))

    for obj_idx in range(mask.shape[-1]):
        obj_mask = mask[..., obj_idx]
        # Check for full occlusion
        existing = combined_mask[obj_mask]

        # Objects that are full occluded when they where overriden
        occluded_objs = set(np.unique(existing)) - \
            set(np.unique(combined_mask))
        # remove object mask where is full occlusion
        for o in occluded_objs:
            obj_mask[existing == o] = 0.
        # Now the obj_mask should have wholes where other objects are which are fully occluded
        combined_mask[obj_mask] = colors[obj_idx]

    cv2.imwrite(path, combined_mask)


def save_unary_mask(mask: torch.Tensor, path: str) -> None:
    """Saves the given unary mask to the given path.
    Supports floats when tif is used as file format.

    Parameters
    ----------
    mask : torch.Tensor
        Mask with potential unaries.
    path : str
        Path to save to.
    """
    import cv2
    if len(mask.shape) == 2:
        mask = mask[None, ...]
    if len(mask.shape) != 3:
        raise ValueError("Mask should be 2D or 3D")
    if mask.shape[0] != 1:
        raise ValueError("Mask should have only one channel => Multi channel masks are not supported when \
                         saving unaries.")
    mask = mask.detach().cpu().permute(1, 2, 0).numpy()
    cv2.imwrite(path, mask)


def save_mask(mask: VEC_TYPE, path: str) -> None:
    """Saves the given (value) based mask to the given path.

    Parameters
    ----------
    mask : VEC_TYPE
        Mask to save, should be of shape C x H x W or H x W (resp. torch.Tensor or np.ndarray)
        Can have vales in range 0 - 255. If values are floats, an error is raised.
    path : str
        Path to save the mask to.

    Raises
    ------
    ValueError
        If the mask contains floats or values outside the range 0 - 255.
    """
    import cv2
    if isinstance(mask, torch.Tensor):
        if len(mask.shape) == 2:
            mask = mask[None, ...]
        mask = mask.detach().cpu().permute(1, 2, 0).numpy()
    # Check if all values are ints
    if not np.all((mask.astype(int) == mask)):
        raise ValueError("Mask must be integer values")
    # Check if all values are in range 0 - 255
    if not np.all((mask >= 0) & (mask <= 255)):
        raise ValueError("Mask must be in range 0 - 255")
    # Cast to uint8
    mask = mask.astype(np.uint8)
    cv2.imwrite(path, mask)


@dataclass
class PredictionResult:

    # region Inputs and other data

    ground_truth: torch.Tensor
    # endregion

    # region output
    raw_model_output: Any

    # endregion


def split_model_result(res: Any, model, dataloader, image, compute_crf: bool = False):
    ret = dict()
    res_pred, res_prior = model.split_model_output(res, additional_data=ret)[0]

    ret["segmentation_raw"] = res_pred
    ret["prior_raw"] = res_prior

    # Optional boxes and labels from instance segmentation
    boxes = ret.get("boxes", None)
    labels = ret.get("labels", None)
    if boxes is not None and len(boxes) > 0:
        boxes = boxes[0]  # Single image inference, select first
    if labels is not None and len(labels) > 0:
        labels = labels[0]
        if hasattr(dataloader, "decode_classes"):
            labels = dataloader.decode_classes(labels)
    if boxes is not None:
        ret["boxes"] = boxes
    if labels is not None:
        ret["labels"] = labels

    if res_pred is None or res_pred.numel() == 0:
        res_pred = torch.ones(image.shape[1:])

    res_pred = dataloader.decode_encoding(res_pred)

    if res_pred.shape[-2:] != image.shape[1:]:
        res_pred = res_pred.reshape(res_pred.shape[:-2] + image.shape[1:])

    res_pred = res_pred.squeeze()
    if len(res_pred.shape) == 2:
        res_pred = res_pred[None, ...]

    num_objs = res_pred.shape[0]
    if res_prior is not None:
        res_prior = dataloader.decode_encoding(res_prior)

        if res_prior.shape != res_pred.shape:
            res_prior = res_prior.reshape(res_pred.shape)

        res_prior = res_prior.squeeze()
        if len(res_prior.shape) == 2:
            res_prior = res_prior[None, ...]

    ret["segmentation"] = res_pred
    ret["prior"] = res_prior

    if compute_crf:
        crf = dense_crf(image, ret.get("segmentation_raw"),
                        is_softmax_unaries=model.use_segmentation_sigmoid)
        # Convert crf to torch
        crf = torch.from_numpy(crf).permute(2, 0, 1).float()
        ret["segmentation_crf_raw"] = crf
        ret["segmentation_crf"] = dataloader.decode_encoding(crf)

    return ret


def save_result(model: torch.nn.Module,
                model_gets_targets: bool,
                dataloader: Any,
                base_dir: str,
                index: int,
                step: int = -1,
                output_folder: str = "output",
                result_folder: str = "result",
                save_raw: bool = False,
                compute_crf: bool = False,
                ) -> None:

    model = model
    os.makedirs(os.path.join(base_dir, output_folder), exist_ok=True)
    os.makedirs(os.path.join(base_dir, result_folder), exist_ok=True)
    if save_raw:
        os.makedirs(os.path.join(
            base_dir, result_folder, "raw"), exist_ok=True)
    with plt.ioff():

        # prior_state, ((img, fg, bg), gt) = dataloader[index]
        # Functions treats 1 as BG and 0 as FG

        res, ground_truth, img, fg, bg = get_result(
            model, dataloader, index, model_gets_targets=model_gets_targets)
        res = split_model_result(
            res, model, dataloader, img, compute_crf=compute_crf)
        res_prior = res.get("prior", None)
        res_pred = res["segmentation"]
        boxes = res.get("boxes", None)
        labels = res.get("labels", None)

        obj_str = ""

        mask_path = os.path.join(
            base_dir, result_folder, f"mask_{index}{obj_str}_ep_{step}.png")
        save_result_mask(res_pred, mask_path)
        if save_raw:
            raw_path = os.path.join(
                base_dir, result_folder, "raw", f"unary_{index}{obj_str}_ep_{step}.tif")
            save_unary_mask(res.get("segmentation_raw"), raw_path)

        if compute_crf:
            crf = res.get("segmentation_crf", None)
            mask_path = os.path.join(
                base_dir, result_folder, f"mask_crf_{index}{obj_str}_ep_{step}.png")
            save_result_mask(crf, mask_path)

            if save_raw:
                raw_path = os.path.join(
                    base_dir, result_folder, "raw", f"unary_crf_{index}{obj_str}_ep_{step}.tif")
                save_unary_mask(res.get("segmentation_crf_raw"), raw_path)

        if res_prior is not None:
            # prior = res_prior[obj_idx]

            mask_path = os.path.join(
                base_dir, result_folder, f"prior_mask_{index}{obj_str}_ep_{step}.png")
            save_result_mask(res_prior, mask_path)
            if save_raw:
                save_unary_mask(res.get("prior_raw"), os.path.join(base_dir,
                                                                   result_folder,
                                                                   "raw",
                                                                   f"prior_unary_{index}{obj_str}_ep_{step}.tif"))

            hull_path = os.path.join(
                base_dir, output_folder, f"prior_img_{index}{obj_str}_ep_{step}.png")
            fig = plot_image_scribbles(img, res_pred, fg, bg, res_prior, save=True, size=5,
                                       path=hull_path, tight_layout=True, title="Epoch: " + str(step),
                                       legend=False)
            plt.close(fig)
        else:
            path = os.path.join(base_dir, output_folder,
                                f"img_{index}{obj_str}_ep_{step}.png")
            fig = plot_image_scribbles(image=img,
                                       inference_result=res_pred,
                                       foreground_mask=fg,
                                       background_mask=bg,
                                       prior_result=None,
                                       boxes=boxes,
                                       labels=labels,
                                       save=True,
                                       size=5,
                                       path=path,
                                       tight_layout=True,
                                       title="Epoch: " + str(step),
                                       legend=False)
            plt.close(fig)


def count_parameters(model: torch.nn.Module) -> List[Dict[str, Any]]:
    """Counts the number of parameters in the given model.

    Parameters
    ----------
    model : torch.nn.Module
        Pytorch module to count the parameters of.

    Returns
    -------
    List[Dict[str, Any]]
        List of dictionaries containing the name, number of learnable parameters and id of the parameter.
        Name is the "path" to the parameter in the model.
    """
    param_list = []
    total_params = 0
    for i, (name, parameter) in enumerate(model.named_parameters()):
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        total_params += params
        param_list.append(dict(name=name, learnable_params=params, id=i))
    param_list.append(
        dict(name="total", learnable_params=total_params, id=len(param_list)))
    return param_list
