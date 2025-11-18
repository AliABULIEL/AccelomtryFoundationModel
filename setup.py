from setuptools import setup, find_packages

setup(
    name="ttm-accelerometry",
    version="0.1.0",
    description="Production-grade TTM foundation model for UK Biobank accelerometry analysis",
    author="TTM Accelerometry Team",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "granite-tsfm==0.2.18",
        "transformers>=4.35.0",
        "torch>=2.0.0",
        "accelerate>=0.24.0",
        "peft>=0.7.0",
        "actipy>=3.0.0",
        "h5py>=3.10.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.11.0",
        "scikit-learn>=1.3.0",
    ],
)
