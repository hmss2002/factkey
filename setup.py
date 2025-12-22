#!/usr/bin/env python3
"""
Setup script for FactKey package.
"""

from setuptools import setup, find_packages

setup(
    name="factkey",
    version="0.1.0",
    description="Anchor-Cycle Method for Mitigating Reversal Curse in LLMs",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/factkey",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.45.0",
        "datasets>=2.20.0",
        "accelerate>=0.33.0",
        "peft>=0.12.0",
        "tqdm>=4.66.0",
        "pyyaml>=6.0",
        "matplotlib>=3.8.0",
        "seaborn>=0.13.0",
        "pandas>=2.1.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.12.0",
            "isort>=5.13.0",
            "mypy>=1.8.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
