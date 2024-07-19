# Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
usage: docs_builder.py [-h] --src SRC --dest DEST

Create and package documentation for the repository

optional arguments:
  -h, --help   show this help message and exit
  --src SRC    the source directory to read the source for the docs from
  --dest DEST  the destination directory to put the built docs
"""

import os
from typing import Dict, List, Tuple

from setuptools import find_packages, setup

# default variables to be overwritten by the version.py file
is_release = None
is_dev = None
version = "unknown"
version_base = version

# load and overwrite version and release info from sparseml package
exec(open(os.path.join("src", "sparsezoo", "version.py")).read())
print(f"loaded version {version} from src/sparsezoo/version.py")

if is_release:
    _PACKAGE_NAME = "sparsezoo"
elif is_dev:
    _PACKAGE_NAME = "sparsezoo-dev"
else:
    _PACKAGE_NAME = "sparsezoo-nightly"

_deps = [
    "numpy>=1.17.0,<2.0",
    "onnx>=1.5.0,<1.15.0",
    "pyyaml>=5.1.0",
    "requests>=2.0.0",
    "tqdm>=4.0.0",
    "pydantic~=2.0",
    "click>=7.1.2,!=8.0.0",  # latest version < 8.0 + blocked version with reported bug
    "protobuf>=3.12.2",
    "pandas>1.3",
    "py-machineid>=0.3.0",
    "geocoder>=1.38.0",
    "onnxruntime>=1.0.0",
]
_notebook_deps = ["ipywidgets>=7.0.0", "jupyter>=1.0.0"]

_dev_deps = [
    "beautifulsoup4==4.9.3",
    "black==22.12.0",
    "flake8>=3.8.3",
    "isort>=5.7.0",
    "pytest>=6.0.0",
    "wheel>=0.36.2",
    "matplotlib>=3.0.0",
]

_docs_deps = [
    "m2r2~=0.2.7",
    "mistune==0.8.4",
    "myst-parser~=0.14.0",
    "rinohtype>=0.4.2",
    "sphinx>=3.4.0",
    "sphinx-copybutton>=0.3.0",
    "sphinx-markdown-tables>=0.0.15",
    "sphinx-multiversion==0.2.4",
    "sphinx-rtd-theme",
]


def _setup_packages() -> List:
    return find_packages(
        "src", include=["sparsezoo", "sparsezoo.*"], exclude=["*.__pycache__.*"]
    )


def _setup_install_requires() -> List:
    return _deps


def _setup_extras() -> Dict:
    return {"dev": _dev_deps, "nb": _notebook_deps, "docs": _docs_deps}


def _setup_entry_points() -> Dict:
    return {
        "console_scripts": [
            "sparsezoo=sparsezoo.main:main",
            "sparsezoo.download=sparsezoo.download_main:main",
            "sparsezoo.analyze=sparsezoo.analyze_cli:main",
            "sparsezoo.deployment_package=sparsezoo.deployment_package.cli:main",
        ]
    }


def _setup_long_description() -> Tuple[str, str]:
    return open("README.md", "r", encoding="utf-8").read(), "text/markdown"


setup(
    name=_PACKAGE_NAME,
    version=version,
    author="Neuralmagic, Inc.",
    author_email="support@neuralmagic.com",
    description=(
        "Neural network model repository for highly sparse and sparse-quantized "
        "models with matching sparsification recipes"
    ),
    long_description=_setup_long_description()[0],
    long_description_content_type=_setup_long_description()[1],
    keywords=(
        "inference, machine learning, neural network, deep learning model, models, "
        "computer vision, nlp, pretrained transfer learning, sparsity, pruning, "
        "quantization, sparse models, resnet, mobilenet, yolov3"
    ),
    license="Apache",
    url="https://github.com/neuralmagic/sparsezoo",
    package_dir={"": "src"},
    packages=_setup_packages(),
    include_package_data=True,
    install_requires=_setup_install_requires(),
    extras_require=_setup_extras(),
    entry_points=_setup_entry_points(),
    python_requires=">=3.8.0",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
