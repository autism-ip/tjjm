"""
 * [INPUT]: 依赖 setuptools 的包发现与安装配置
 * [OUTPUT]: 对外提供 lung-diffusion-anomaly 的 Python 包元数据与运行时依赖
 * [POS]: 项目根配置，定义 src/ 包布局和可安装依赖边界
 * [PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from setuptools import setup, find_packages

setup(
    name="lung-diffusion-anomaly",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "monai>=1.3.0",
        "pytorch-lightning>=2.0.0",
        "hydra-core>=1.3.0",
        "nibabel>=5.0.0",
        "scipy>=1.10.0",
        "scikit-image>=0.20.0",
        "scikit-learn>=1.3.0",
        "SimpleITK>=2.3.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "tensorboard>=2.13.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
    ],
)
