## Getting Started

In this document we will guide you through the process of setting up the environment to run any of the experiments in this repository or to run the experiments or notebooks.

> Note: *Advanced users, who are familar with poetry or want to use conda/pip with a requirements.txt files, can skip this section. Nonetheless, we recommend using poetry as this was used for development.*

### 1. Environment Setup

This project uses poetry to manage the dependencies. Therefore it's recommended to install poetry first. You can find the installation instructions [here](https://python-poetry.org/docs/). To install multiple python versions, you might also want to use [pyenv](https://github.com/pyenv/pyenv) or [pyenv-win](https://github.com/pyenv-win/pyenv-win) on Windows.

As many researchers are using conda, or pip with other virtual environments, we provide a `requirements.txt` file for the installation of the dependencies.

#### 1.A Poetry

Once poetry is installed, you can create a virtual environment and install the dependencies by running the following commands in the root directory of the repository:

```bash
poetry env use [Your-Python-Version]
```
Make sure to have a supported python version installed on your system. You can find the supported python versions in the `pyproject.toml` file.
If your default version in poetry is >3.9 a simple:

```bash
poetry shell
```

Should do the trick as well.

Make sure the environment is activated, by checking whether the python executable is the one from the virtual environment. You can check this by running:

```bash
which python
```

on Unix systems or:

```powershell
(Get-Command python).Path
```

on Windows systems (powershell).

Once the environment is activated, you can install the dependencies by running:

```bash
poetry install
```

This will install the basic dependencies for the project. If you want to install the development dependencies as well - **which you will need for executing the notebooks** - you can run:

```bash
poetry install --with dev
```

#### 1.B Pip / Conda

If you prefer to use pip or conda, you can install the dependencies from the requirements.txt file. To include development dependencies you should use the requirements.dev.txt file. As conda / pip might have issues with hashes and constraints in the file, we also providing a slim version of both of the files, which just includes the package names.

```bash
pip install -r [Requirements-Path]
```

These files were generated on Windows, so there might be some package conflicts on Unix systems. If you encounter any issues, please refer to the `pyproject.toml` file for the actual needed dependencies.

### 2. PyTorch Installation

In the dependency configuration, we assume a Cuda version of `11.8` and a fixed pytorch version. You may need/want to adjust these based on your needs.
To do so we refer to the official [pytorch installation guide](https://pytorch.org/get-started/locally/).


### 3. Done

If you want to execute our experiments, we recommend following the [reproduction guide](reproduction_guide.md) to download the data and models needed, which we automized in a setup script.
If you want to run our how-to notebooks, you can find them in the [notebooks](../notebooks/how_to/) folder.

We have currently explanatory notebooks for:
- [Convexity](../notebooks/how_to/convexity.ipynb)
- [Path-Connectedness](../notebooks/how_to/path-connectedness.ipynb)

To run the notebooks you dont need to run the reproducibility script, but you need to install the development dependencies.

### N. Pre-Commit [Developers]

This repository uses [pre-commit](https://pre-commit.com/) to ensure that the code is formatted correctly and environment files are up-to-date ([pre-commit-config](../.pre-commit-config.yaml)). If the environment was set up with the `dev` group, the `pre-commit` package should be installed already. To register the pre-commit hooks, run the following command in the root directory of the repository, with the virtual environment activated:

```bash
pre-commit install
```
