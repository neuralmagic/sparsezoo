from typing import Tuple, List, Dict
from setuptools import find_packages, setup

_deps = [
    "requests>=2.0.0",
    "PyYaml>=5.1.0",
    "tqdm>=4.0.0",
    "numpy>=1.0.0",
    "onnx>=1.0.0",
]

_dev_deps = [
    "black>=20.8b1",
    "flake8>=3.8.3",
    "isort>=5.7.0",
    "rinohtype>=0.4.2",
    "sphinxcontrib-apidoc>=0.3.0",
    "wheel>=0.36.2",
    "pytest>=6.0.0",
]


def _setup_packages() -> List:
    return find_packages(
        "src", include=["sparsezoo", "sparsezoo.*"], exclude=["*.__pycache__.*"]
    )


def _setup_install_requires() -> List:
    return _deps


def _setup_extras() -> Dict:
    return {"dev": _dev_deps}


def _setup_entry_points() -> Dict:
    return {}


def _setup_long_description() -> Tuple[str, str]:
    return open("README.md", "r", encoding="utf-8").read(), "text/markdown"


setup(
    name="sparsezoo",
    version="0.1.0",
    author="Mark Kurtz, Kevin Rodriguez",
    author_email="support@neuralmagic.com",
    description="[TODO]",
    long_description=_setup_long_description()[0],
    long_description_content_type=_setup_long_description()[1],
    keywords="[TODO]",
    license="[TODO]",
    url="https://github.com/neuralmagic/sparsezoo",
    package_dir={"": "src"},
    packages=_setup_packages(),
    include_package_data=True,
    install_requires=_setup_install_requires(),
    extras_require=_setup_extras(),
    entry_points=_setup_entry_points(),
    python_requires=">=3.6.0",
    classifiers=["[TODO]"],
)
