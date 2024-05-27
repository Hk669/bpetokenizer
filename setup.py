import os
from setuptools import find_packages, setup

here = os.path.abspath(os.path.dirname(__file__))

# Get the code version
version = {}
with open(os.path.join(here, "bpetokenizer/version.py")) as f:
    exec(f.read(), version)
__version__ = version["__version__"]


with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()


setup(
    name="bpetokenizer",
    version=__version__,
    description="Byte Pair Encoding Tokenizer with special tokens and regex pattern",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Hk669/bpetokenizer",
    author="Hrushikesh Dokala",
    author_email="hrushi669@gmail.com",
    license="MIT",
    package_dir={"bpetokenizer": "bpetokenizer"},
    packages=find_packages(where="bpetokenizer"),
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    install_requires=[],
    extras_require={
        "dev": ["pytest", "twine"],
    },
    python_requires=">=3.9,<3.13",
)