from setuptools import setup, find_packages

setup(
    name="calibrate",
    version="0.1",
    packages=find_packages(),
    python_requries=">=3.8",
    install_requires=[
        "torch",
        "torchvision>=0.8.2",
        "ipdb==0.13.9",
        "albumentations==1.1.0",
        "opencv-python==4.5.1.48",
        "hydra-core==1.2.0",
        "flake8==4.0.1",
        "terminaltables==3.1.10",
        "matplotlib==3.5.1",
        "plotly==5.7.0",
        "pandas==1.4.2"
    ],
)
