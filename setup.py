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

import sys
from datetime import date
from typing import Tuple, List, Dict
from setuptools import find_packages, setup

_PACKAGE_NAME = "sparsezoo"
_VERSION = "0.2.0"
_VERSION_MAJOR, _VERSION_MINOR, _VERSION_BUG = _VERSION.split(".")
_VERSION_MAJOR_MINOR = f"{_VERSION_MAJOR}.{_VERSION_MINOR}"
_NIGHTLY = "nightly" in sys.argv

if _NIGHTLY:
    _PACKAGE_NAME += "-nightly"
    _VERSION += "." + date.today().strftime("%Y%m%d")
    # remove nightly param so it does not break bdist_wheel
    sys.argv.remove("nightly")

_deps = [
    "numpy>=1.0.0",
    "onnx>=1.0.0,<1.8.0",
    "pyyaml>=5.1.0",
    "requests>=2.0.0",
    "tqdm>=4.0.0",
]
_notebook_deps = ["ipywidgets>=7.0.0", "jupyter>=1.0.0"]

_dev_deps = [
    "black>=20.8b1",
    "flake8>=3.8.3",
    "isort>=5.7.0",
    "onnxruntime>=1.0.0",
    "pytest>=6.0.0",
    "rinohtype>=0.4.2",
    "recommonmark>=0.7.0",
    "sphinx>=3.4.0",
    "sphinx-copybutton>=0.3.0",
    "sphinx-markdown-tables>=0.0.15",
    "wheel>=0.36.2",
    "pytest>=6.0.0",
    "sphinx-rtd-theme",
    "wheel>=0.36.2",
]


def _setup_packages() -> List:
    return find_packages(
        "src", include=["sparsezoo", "sparsezoo.*"], exclude=["*.__pycache__.*"]
    )


def _setup_install_requires() -> List:
    return _deps


def _setup_extras() -> Dict:
    return {"dev": _dev_deps, "nb": _notebook_deps}


def _setup_entry_points() -> Dict:
    return {"console_scripts": ["sparsezoo=sparsezoo.main:main"]}


def _setup_long_description() -> Tuple[str, str]:
    return open("README.md", "r", encoding="utf-8").read(), "text/markdown"


setup(
    name=_PACKAGE_NAME,
    version=_VERSION,
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
    python_requires=">=3.6.0",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "Programming Language :: Python :: 3",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
