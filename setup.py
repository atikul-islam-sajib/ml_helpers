from setuptools import setup, find_packages
import codecs
import os

VERSION = "0.0.1"
DESCRIPTION = "Atikul Islam Sajib"
LONG_DESCRIPTION = "A package to use Custom RF - classifier and regressor"

# Setting up
setup(
    name="ml_helpers",
    version=VERSION,
    author="Atikul Islam Sajib",
    author_email="atikulislamsajib137@gmail.com",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=[],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ],
)
