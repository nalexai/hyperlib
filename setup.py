from setuptools import find_packages
from distutils.core import setup, Extension
import pybind11
from pybind11.setup_helpers import Pybind11Extension, build_ext
from glob import glob
import sys

__version__ = "0.0.6"
with open("README.md") as f:
    readme = f.read()

ext_modules = [
    Pybind11Extension("__hyperlib_embedding", sorted(glob("hyperlib/embedding/src/*.cc")),
        include_dirs = ["hyperlib/embedding/include", pybind11.get_include()],
        cxx_std=11,
    )
]

setup(
    name="hyperlib",
    version=__version__,
    packages = find_packages(),
    setup_requires = ["pip>=21"],
    install_requires = [
        "numpy>=1.19.3",
        "tensorflow>=2.0.0",
        "scipy>=1.7.0",
        "mpmath",
        "networkx",
        "pybind11>=2.7.0",
        "gmpy2",
    ],
    ext_modules=ext_modules,
    author="Nalex.ai",
    author_email="info@nalexai.com",
    description="Library that contains implementations of machine learning components in the hyperbolic space",
    long_description=readme,
    long_description_content_type="text/markdown",
    project_urls = {
        "Source" : "https://github.com/nalexai/hyperlib",
        "Issues" : "https://github.com/nalexai/hyperlib/issues"
    },
    license_files = "LICENSE",
    url = "https://github.com/nalexai/hyperlib",
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
