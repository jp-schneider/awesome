
# Class for functions
# File for useful functions when using matplotlib
from typing import Any, Callable, List, Optional, Union
try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    import matplotlib as mpl
except (ModuleNotFoundError, ImportError):
    plt = None
    Figure = None
    pass
import os
from functools import wraps
from awesome.util.path_tools import numerated_file_name, open_in_default_program

def saveable(
    default_ext: Union[str, List[str]] = "png",
    default_output_dir: Optional[str] = "./temp",
    default_transparent: bool = False,
    default_dpi: int = 300,
    default_override: bool = False,
    default_tight_layout: bool = False,
):
    """Declares a matplotlib figure producing function as saveable so the functions
    figure output can directly saved to disk by setting the save=True as kwarg.

    Supported params:

    kwargs
    ---------

    save: bool, optional
       Triggers saving of the output, Default False.
    
    open: bool, optional
        Opens the saved figure in the default program, Default False.

    path: str, optional
        Path where the figure should be saved. Can be a path to a directory, or just a filename.
        If it is a filename, the figure will be saved in a default folder.
        Default default_output_dir + uuid4()
    
    ext: str, optional
        File extension of the path. If path already contains an extension, this is ignored.
        Otherwise it can be a str or a list to save the figure in different formats, like ["pdf", "png"]
        Default see: default_ext
        
    transparent: bool, optional
        If the generated plot should be with transparent background. Default see: default_transparent

    dpi: int, optional
        The dpi when saving the figure.
        Default see: default_dpi

    override: bool optional
        If the function should override existing file with the same name.
        Default see: default_override

    tight_layout: bool, optional
        If the function should call tight_layout on the figure before saving.
        Default see: default_tight_layout
    
    set_interactive_mode: bool, optional
        If the function should set the interactive mode of matplotlib before calling the function.
        Default None
        Meaning it will not change the interactive mode. The interactive mode will be restored after the function call.

    auto_close: bool, optional
        If the function should close the figure after saving.
        Default False

    Parameters
    ----------
    default_ext : Union[str, List[str]], optional
        [List of] Extension[s] to save the figure, by default "png"
    default_output_dir : Optional[str], optional
        Output directory of figures when path did not contain directory information, 
        If the Environment Variable "PLOT_OUTPUT_DIR" is set, it will be used as destination.
        by default "./temp"
    default_transparent : bool, optional
        If the function should output the figures with transparent background, by default False
    default_dpi : int, optional
        If the function should by default override, by default 300
    default_override : bool, optional
        If the function should by default override, by default False
    default_tight_layout : bool, optional
        If the function should by default call tight_layout on the figure, by default False
    """
    from uuid import uuid4

    def decorator(function: Callable[[Any], Figure]) -> Callable[[Any], Figure]:
        @wraps(function)
        def wrapper(*args, **kwargs):
            nonlocal default_output_dir
            path = kwargs.pop("path", str(uuid4()))
            save = kwargs.pop("save", False)
            ext = kwargs.pop("ext", default_ext)
            transparent = kwargs.pop("transparent", default_transparent)
            dpi = kwargs.pop("dpi", default_dpi)
            override = kwargs.pop("override", default_override)
            tight_layout = kwargs.pop("tight_layout", default_tight_layout)
            open = kwargs.pop("open", False)
            set_interactive_mode = kwargs.pop("set_interactive_mode", None)
            close = kwargs.pop("auto_close", False)
            display = kwargs.pop("display", False)
            display_auto_close = kwargs.pop("display_auto_close", True)

            # Get interactive mode.
            is_interactive = mpl.is_interactive()

            if set_interactive_mode is not None:
                mpl.interactive(set_interactive_mode)
            try:
                out = function(*args, **kwargs)
            finally:
                mpl.interactive(is_interactive)

            if tight_layout:
                out.tight_layout()

            paths = []
            if save or open:
                if any([(x in path) for x in ["/", "\\", os.sep]]):
                    # Treat path as abspath
                    path = os.path.abspath(path)
                else:
                    default_output_dir = os.environ.get("PLOT_OUTPUT_DIR", default_output_dir)
                    path = os.path.join(os.path.abspath(default_output_dir), path)
                # Check if path has extension
                _, has_ext = os.path.splitext(path)
                if len(has_ext) == 0:
                    if isinstance(ext, str):
                        ext = [ext]
                    for e in ext:
                        paths.append(path + "." + e)
                else:
                    paths = [path]

                # Create parent dirs
                dirs = set([os.path.dirname(p) for p in paths])
                for d in dirs:
                    os.makedirs(d, exist_ok=True)

                for p in paths:
                    if not override:
                        p = numerated_file_name(p)
                    out.savefig(p, transparent=transparent, dpi=dpi)
            if open:
                try:
                    open_in_default_program(paths[0])
                except Exception as err:
                    pass
            if display:
                from IPython.display import display
                display(out)
                if display_auto_close:
                    plt.close(out)
            if close and not (display and display_auto_close):
                plt.close(out)
            return out
        return wrapper
    return decorator