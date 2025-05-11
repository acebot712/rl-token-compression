from setuptools import setup, find_packages

setup(
    name="rl-token-compression",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "stable-baselines3>=2.0.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "tqdm>=4.65.0",
        "datasets>=2.12.0",
        "wandb>=0.15.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.2.0"
    ],
) 