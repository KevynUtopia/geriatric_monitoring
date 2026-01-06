#!/usr/bin/env python3
"""
Setup script for YOLO11 Keypoint Detection with Self-Training.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    requirements = requirements_path.read_text().strip().split("\n")
    requirements = [req.strip() for req in requirements if req.strip() and not req.startswith("#")]

setup(
    name="yolo11-keypoint-selftraining",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="YOLO11 Keypoint Detection with Self-Training capabilities",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/yolo11-keypoint-selftraining",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "isort>=5.0.0",
            "mypy>=0.950",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "myst-parser>=0.17.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "yolo11-keypoint-train=train:main",
            "yolo11-keypoint-eval=evaluate:main",
            "yolo11-keypoint-setup=data_setup:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.md", "*.txt"],
    },
    zip_safe=False,
    keywords="yolo11 keypoint detection pose estimation self-training pytorch computer-vision",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/yolo11-keypoint-selftraining/issues",
        "Source": "https://github.com/yourusername/yolo11-keypoint-selftraining",
        "Documentation": "https://github.com/yourusername/yolo11-keypoint-selftraining/blob/main/README.md",
    },
) 