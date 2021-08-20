import pybind11
from setuptools import find_packages, setup
from setuptools.extension import Extension
from glob import glob
import sys

__version__ = "0.0.3"

ext_modules = [
        Extension("_embedding",
            sorted(glob("hyperlib/embedding/src/*.cc")),
            include_dirs = ["hyperlib/embedding/include", pybind11.get_include()],
            language = "c++",
            extra_compile_args = ["-std=c++11"]
        ),
]

setup(
    name="hyperlib",
    packages = find_packages(),
    setup_requires = ["pip>=21"],
    install_requires = ["numpy>=1.19.3", 
						#"tensorflow>=2.0.0",
						"scipy", 
                        "mpmath"],
    version=__version__,
    author="Nalex.ai",
    author_email="info@nalexai.com",
    description="A hyperbolic deep learning library",
    ext_modules=ext_modules,
    license_files = "LICENSE",
    zip_safe = False
)
