from setuptools import setup, find_packages

setup(
    name="imitation_learning_lerobot",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "lerobot",
        "numpy",
        "pathlib",
    ],
    python_requires=">=3.8",
)
