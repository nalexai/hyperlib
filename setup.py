from setuptools import find_packages
from distutils.core import setup, Extension
import pybind11
from glob import glob
import sys

__version__ = "0.0.3"
with open("README.md") as f:
    readme = f.read()

ext_modules = [
        Extension("_embedding",
            sorted(glob("hyperlib/embedding/src/*.cc")),
            include_dirs = ["hyperlib/embedding/include", pybind11.get_include()],
            language = "c++",
            extra_compile_args = ["-std=c++11"]
        )
]


setup(
    name="hyperlib",
    version=__version__,
    packages = find_packages(),
    setup_requires = ["pip>=21"],
    install_requires = ["numpy>=1.19.3", 
						"tensorflow>=2.0.0",
						"scipy>=1.7.0", 
                        "mpmath"],
    ext_modules=ext_modules,
    author="Nalex.ai",
    author_email="info@nalexai.com",
    description="Library that contains implementations of machine learning components in the hyperbolic space",
    long_description=readme,
    project_urls = {
                    "Source" : "https://github.com/nalexai/hyperlib",
                    "Issues" : "https://github.com/nalexai/hyperlib/issues"
                    },
    license_files = "LICENSE",
    url = "https://github.com/nalexai/hyperlib",
    classifiers = [ 
                    "Programming Language :: Python :: 3",
                    "Liscense :: OSI Approved :: MIT License",
                    "Operating System :: OS Independent"
                    ], 
)
