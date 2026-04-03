from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="titans-torch",
    version="0.1.2",
    author="Bui Quang Dat",
    author_email="buiquangdat1458@gmail.com",
    description="PyTorch implementation of Titans: Learning to Memorize at Test Time",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/buiquangdat1710/titans-torch",
    project_urls={
        "Bug Tracker": "https://github.com/buiquangdat1710/titans-torch/issues",
        "Source Code": "https://github.com/buiquangdat1710/titans-torch",
    },
    packages=find_packages(exclude=["tests*", "examples*"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "icecream>=2.1.0",
        ],
    },
)