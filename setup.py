#!/usr/bin/env python
# For installing the Peach package
import torch
from os import path
from setuptools import find_packages, setup

torch_ver = [int(x) for x in torch.__version__.split(".")[:2]]
assert torch_ver >= [1, 7], "Requires PyTorch >= 1.7"

# Get version from Peach/__init__.py
def get_version():
    init_py_path = path.join(path.abspath(path.dirname(__file__)), "Peach", "__init__.py")
    init_py = open(init_py_path, "r").readlines()
    version_line = [l.strip() for l in init_py if l.startswith("__version__")][0]
    version = version_line.split("=")[-1].strip().strip("'\"")
    return version

# Install Peach package
setup(
    name="AutoMM",
    version=get_version(),
    author="MilkTea",
    packages=find_packages(exclude=("configs", "tests*")),
    # package_dir=PROJECTS,
    python_requires=">=3.6",
    install_requires=[
        # These dependencies are not pure-python.
        # In general, avoid adding more dependencies like them because they are not
        # guaranteed to be installable by `pip install` on all platforms.
        # To tell if a package is pure-python, go to https://pypi.org/project/{name}/#files
        "Pillow>=7.1",  # or use pillow-simd for better performance
        "matplotlib",  
        "termcolor>=1.1",
        "yacs>=0.1.6",
        "tabulate",
        "tqdm>4.29.0",
        "cloudpickle",
        "tensorboard",
        "thop",
        "torch-dct",
        "timm",
        "einops",
        "pyqtwebengine==5.12",
        # Lock version of fvcore/iopath because they may have breaking changes
        # NOTE: when updating fvcore/iopath version, make sure fvcore depends
        # on compatible version of iopath.
        "fvcore>=0.1.5,<0.1.6",  # required like this to make it pip installable
        "iopath>=0.1.7,<0.1.9",
        "omegaconf>=2.1",
        "hydra-core>=1.1",
        "black>=21.4b2",
        # If a new dependency is required at import time (in addition to runtime), it
        # probably needs to exist in docs/requirements.txt, or as a mock in docs/conf.py
    ],
    extras_require={
        # optional dependencies, required by some features
        "all": [
            "shapely",
            "pygments>=2.2",
        ],
        # dev dependencies.
        "dev": [
            "flake8==3.8.1",
            "isort==4.3.21",
            "flake8-bugbear",
            "flake8-comprehensions",
        ],
    },
)