import os
import toml


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


def get_package_info() -> dict:
    """Gets the package information of the project.
    The package information is defined as the package name, version and description.

    Returns
    -------
    dict
        The package information of the project.
    """
    try:
        project_root_path = get_project_root_path()
        pyproject_path = os.path.join(project_root_path, "pyproject.toml")
        with open(pyproject_path, "r") as file:
            pyproject = toml.load(file)
        package_info = pyproject["tool"]["poetry"]
        return package_info
    except FileNotFoundError as err:
        # Toml file not found, package could be installed. Just return the package info.
        return {
            "name": "awesome",
            "version": "0.1.0",
            "description": "Awesome package"
        }


def get_package_name() -> str:
    """Gets the package name of the project.
    The package name is defined as the name of the package.

    Returns
    -------
    str
        The package name of the project.
    """
    try:
        package_info = get_package_info()
        package_name = package_info["name"]
        return package_name
    except FileNotFoundError as err:
        # Toml file not found, package could be installed. Just return the package name.
        return "awesome"
