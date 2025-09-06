#!/usr/bin/env python3
"""
Setup script for RL Token Compression project.
"""

from setuptools import find_packages, setup

setup(
    name="rl-token-compression",
    version="0.1.0",
    description="RL-based token compression for efficient LLM inference",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[],  # Requirements are in requirements.txt
)
