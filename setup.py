from setuptools import find_packages, setup

# Metadata
name = "combined_package"
version = "0.1.0"
description = "Combined package for RAFT Benchmarks baselines and SetFit"

# Requirements
install_requires = [
    "scipy",
    "scikit-learn==0.24.2",
    "datasets>=2.3.0",
    "transformers",
    "openai",
    "sacred",
    "python-dotenv",
    "cachetools",
    "torch",
    "sentence-transformers>=2.2.1",
    "evaluate>=0.3.0",
    "ipykernel",
    "tiktoken"
]

# Extras
extras_require = {
    "optuna": ["optuna"],
    "quality": ["black", "flake8", "isort", "tabulate"],
    "tests": ["pytest", "pytest-cov", "onnxruntime", "onnx", "skl2onnx", "hummingbird-ml<0.4.9", "openvino==2022.3.0"],
    "onnx": ["onnxruntime", "onnx", "skl2onnx"],
    "openvino": ["hummingbird-ml<0.4.9", "openvino==2022.3.0"],
    "docs": ["hf-doc-builder>=0.3.0"],
    "dev": ["optuna", "black", "flake8", "isort", "tabulate", "pytest", "pytest-cov", "onnxruntime", "onnx", "skl2onnx", "hummingbird-ml<0.4.9", "openvino==2022.3.0", "hf-doc-builder>=0.3.0"],
    "compat_tests": ["pandas<2"]
}

# Combine the setup configurations
setup(
    name=name,
    version=version,
    description=description,
    python_requires=">=3.7.0",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=install_requires,
    extras_require=extras_require,
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="nlp, machine learning, fewshot learning, transformers, benchmarks, baselines",
    zip_safe=False,
)
