import os
import re
import shutil
import subprocess
from typing import List, Literal
from tqdm.auto import tqdm

def relpath(_from: str, _to: str, is_from_file: bool = True, is_to_file: bool = True) -> None:
    """Creates a relative path from _from to _to."""
    _to_dir = os.path.dirname(_to) if is_to_file else _to
    _from_dir = os.path.dirname(_from) if is_from_file else _from
    path = os.path.relpath(_to_dir, _from_dir)
    if is_to_file:
        return os.path.join(path, os.path.basename(_to))
    else:
        return path

def format_os_independent(path: str) -> str:
    """Formats a path to be os independent."""
    return path.replace("\\", "/")


def open_folder(path: str) -> None:
    """Opens the given path in the systems file explorer.

    Parameters
    ----------
    path : str
        Path f open in explorer.
    """
    from sys import platform
    path = os.path.abspath(os.path.normpath(path))
    if os.path.exists(path):
        if platform == "linux" or platform == "linux2":
            # linux
            raise NotImplementedError()
        elif platform == "darwin":
            # OS X
            raise NotImplementedError()
        elif platform == "win32":
            # Windows...
            subprocess.run(f"explorer {path}")


def open_in_default_program(path_to_file: str) -> None:
    """Opens the given file in the systems default program.

    Parameters
    ----------
    path_to_file : str
        Path to open in default program.
    """
    from sys import platform
    path = os.path.abspath(os.path.normpath(path_to_file))
    if os.path.exists(path):
        if platform == "linux" or platform == "linux2":
            # linux
            raise NotImplementedError()
        elif platform == "darwin":
            # OS X
            raise NotImplementedError()
        elif platform == "win32":
            # Windows...
            subprocess.run(f"powershell {path_to_file}")


def numerated_file_name(path: str, max_check: int = 1000) -> str:
    """Checks whether the given path exists, if so it will try 
    to evaluate a free path by appending a consecutive number.

    Parameters
    ----------
    path : str
        The path to check
    max_check : int, optional
        How much files should be checked until an error is raised, by default 1000

    Returns
    -------
    str
        A Path which is non existing.

    Raises
    ------
    ValueError
        If max_check is reached.
    """
    PATTERN = r" \((?P<number>[0-9]+)\)$"
    pattern = re.compile(PATTERN)
    i = 2
    for _ in range(max_check):
        if os.path.exists(path):
            directory = os.path.dirname(path)
            extension = os.path.splitext(path)[1]
            name = os.path.basename(path).replace(extension, '')
            match = pattern.match(name)
            if i == 2 and match is None:
                name += f' ({i})'
            else:
                if match:
                    number = match.group["number"]
                name = re.sub(pattern, repl=f' ({i})', string=name)
            path = os.path.join(directory, name + extension)
            i += 1
        else:
            return path
    raise ValueError(f"Could not find free path within max checks of: {max_check}!")



def filtered_copy(
        src: str, 
        dst: str, 
        copy_with_basename: bool = True,
        blacklist: List[str] = None,
        match_mode: Literal["path", "basename"] = "basename",
        progress: bool = False,
        relpath: str = None
        ) -> None:
    """Copies a directory (or) file from src to dst if it is not in the blacklist.
    
    Parameters
    ----------
    src : str
        Source path to copy from.
    dst : str
        Destination path to copy to.
    blacklist : List[str], optional
        List of files or directories to exclude, by default None
    progress : bool, optional
        Whether to show progress, by default False
    relpath : str, optional
        A relative path to the source directory, by default None
        Will be set automatically to show nested structure in progress bar.
    """
    black_list_patterns = [re.compile(bl) for bl in (blacklist if blacklist is not None else [])]
    
    if relpath is None:
        relpath = "./"

    def allow_copy(path: str):
        for pattern in black_list_patterns:
            item = os.path.basename(path) if match_mode == "basename" else path
            if pattern.fullmatch(item):
                return False
        return True

    def is_dir(path: str) -> bool:
        ext = os.path.splitext(path)[1]
        return os.path.isdir(path) or ext == ""


    if os.path.isfile(src):
        if allow_copy(src):
            if is_dir(dst):
                if not os.path.exists(dst):
                    os.makedirs(dst)
            else:
                if not os.path.exists(os.path.dirname(dst)):
                    os.makedirs(os.path.dirname(dst))
            shutil.copy2(src, dst)
    elif os.path.isdir(src) and is_dir(dst):
        if copy_with_basename:
            if os.path.basename(src) != os.path.basename(dst):
                dst = os.path.join(dst, os.path.basename(src))
        if allow_copy(src):
            it = os.listdir(src)
            if progress:
                relpath = os.path.join(relpath, os.path.basename(src))
                it = tqdm(it, desc="Path: {}".format(relpath), unit="file")
            for item in it:
                src_file = os.path.join(src, item)
                dst_file = os.path.join(dst, item)
                filtered_copy(src_file, dst_file, blacklist=blacklist, progress=progress, relpath=relpath)
    else:
        raise ValueError(f"Could not copy {src} to {dst}!")

    


def numerated_folder_name(path: str, max_check: int = 1000) -> str:
    """Checks whether the given folder path exists, if so it will try 
    to evaluate a free path by appending a consecutive number.

    Parameters
    ----------
    path : str
        The path to check
    max_check : int, optional
        How much files should be checked until an error is raised, by default 1000

    Returns
    -------
    str
        A Path which is non existing.

    Raises
    ------
    ValueError
        If max_check is reached.
    """
    PATTERN = r" \((?P<number>[0-9]+)\)$"
    pattern = re.compile(PATTERN)
    i = 2
    for _ in range(max_check):
        if os.path.exists(path):
            directory = os.path.dirname(path)
            name = os.path.basename(path)
            match = pattern.match(name)
            if i == 2 and match is None:
                name += f' ({i})'
            else:
                if match:
                    number = match.group["number"]
                name = re.sub(pattern, repl=f' ({i})', string=name)
            path = os.path.join(directory, name)
            i += 1
        else:
            return path
    raise ValueError(f"Could not find free path within max checks of: {max_check}!")



def get_project_root_path() -> str:
    """Gets the root path of the project.
    A project root path is defined as the directory where the 
    pyproject.toml file is located.

    Returns
    -------
    str
        The absolute root path of the project.
    """
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def get_package_root_path() -> str:
    """Gets the package path of the project.
    A package root path is defined as the directory where the 
    source code of the project is located.
    In this case it is the directory where the awesome package is located.

    Returns
    -------
    str
        The absolute package root path of the project.
    """
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))