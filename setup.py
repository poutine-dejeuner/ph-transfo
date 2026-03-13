from setuptools import find_packages, setup

setup(
    name="ph-transfo",
    version="0.1.0",
    description=(
        "Graph Transformer with Persistence Homology Betti number featurization"
    ),
    packages=find_packages(exclude=["tests*"]),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.20.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov",
        ],
    },
)
