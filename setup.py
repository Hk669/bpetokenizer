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
    description="A Byte Pair Encoding (BPE) tokenizer, which algorithmically follows along the GPT tokenizer(tiktoken), allows you to train your own tokenizer. The tokenizer is capable of handling special tokens and uses a customizable regex pattern for tokenization(includes the gpt4 regex pattern). supports `save` and `load` tokenizers in the `json` and `file` format. The `bpetokenizer` also supports [pretrained](bpetokenizer/pretrained/) tokenizers.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Hk669/bpetokenizer",
    author="Hrushikesh Dokala",
    author_email="hrushi669@gmail.com",
    license="MIT",
    packages=find_packages(include=["bpetokenizer"]),
    package_data={
        'bpetokenizer': ['pretrained/wi17k_base/wi17k_base.json'],
    },
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    install_requires=["regex"],
    extras_require={
        "dev": ["pytest", "twine"],
    },
    python_requires=">=3.9,<3.13",
)