from setuptools import setup, Extension, find_packages 
from glob import glob
from pybind11 import get_cmake_dir
import sys

__version__ = "0.0.3"

ext_modules = [
        Extension("_embedding",
        sorted(glob("hyperlib/embedding/src/*.cc")),
        include_dirs = ["hyperlib/embedding/include", "pybind11/include"],
        language = "c++",
        extra_compile_args = ["-std=c++11", "-stdlib=libc++"],
        ),
]

setup(
    name="hyperlib",
    packages = find_packages(),
    version=__version__,
    author="Nalex.ai",
    author_email="info@nalexai.com",
    description="A hyperbolic deep learning library",
    ext_modules=ext_modules,
)
