from setuptools import setup, find_packages
import os

VERSION = "0.0.1"
DESCRIPTION = "Atikul Islam Sajib"
LONG_DESCRIPTION_FILE = "README.md"

# Automatically read the long description from the README file
long_description = ""
if os.path.exists(LONG_DESCRIPTION_FILE):
    with open(LONG_DESCRIPTION_FILE, "r", encoding="utf-8") as fh:
        long_description = fh.read()

# Setting up
setup(
    name="ml_helpers",
    version=VERSION,
    author="Atikul Islam Sajib",
    author_email="atikulislamsajib137@gmail.com",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(),
    install_requires=[
        "graphviz",
        "numpy",
        "scikit-learn",
        "matplotlib",
        "imodels",
        "pmlb",
    ],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ],
)
