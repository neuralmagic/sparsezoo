<!--
Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

<h1><img alt="tool icon" src="https://raw.githubusercontent.com/neuralmagic/sparsezoo/main/docs/source/icon-sparsezoo.png" />&nbsp;&nbsp;SparseZoo</h1>

<h3>Neural network model repository for highly sparse and sparse-quantized models with matching sparsification recipes</h3>

<p>
    <a href="https://docs.neuralmagic.com/sparsezoo">
        <img alt="Documentation" src="https://img.shields.io/badge/documentation-darkred?&style=for-the-badge&logo=read-the-docs" height=25>
    </a>
    <a href="https://join.slack.com/t/discuss-neuralmagic/shared_invite/zt-q1a1cnvo-YBoICSIw3L1dmQpjBeDurQ/">
        <img src="https://img.shields.io/badge/slack-purple?style=for-the-badge&logo=slack" height=25>
    </a>
    <a href="https://discuss.neuralmagic.com/">
        <img src="https://img.shields.io/badge/support%20forums-navy?style=for-the-badge&logo=discourse" height=25>
    </a>
    <a href="https://github.com/neuralmagic/sparsezoo/actions/workflows/test-check.yaml">
        <img alt="Main" src="https://img.shields.io/github/workflow/status/neuralmagic/sparsezoo/Test%20Checks/main?label=build&style=for-the-badge" height=25>
    </a>
    <a href="https://github.com/neuralmagic/sparsezoo/releases">
        <img alt="GitHub release" src="https://img.shields.io/github/release/neuralmagic/sparsezoo.svg?style=for-the-badge" height=25>
    </a>
    <a href="https://github.com/neuralmagic/sparsezoo/blob/main/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/neuralmagic/sparsezoo.svg?color=lightgray&style=for-the-badge" height=25>
    </a>
    <a href="https://github.com/neuralmagic/sparsezoo/blob/main/CODE_OF_CONDUCT.md">
        <img alt="Contributor Covenant" src="https://img.shields.io/badge/Contributor%20Covenant-v2.0%20adopted-ff69b4.svg?color=yellow&style=for-the-badge" height=25>
    </a>
    <a href="https://www.youtube.com/channel/UCo8dO_WMGYbWCRnj_Dxr4EA">
        <img src="https://img.shields.io/badge/-YouTube-red?&style=for-the-badge&logo=youtube&logoColor=white" height=25>
    </a>
     <a href="https://medium.com/limitlessai">
        <img src="https://img.shields.io/badge/medium-%2312100E.svg?&style=for-the-badge&logo=medium&logoColor=white" height=25>
    </a>
    <a href="https://twitter.com/neuralmagic">
        <img src="https://img.shields.io/twitter/follow/neuralmagic?color=darkgreen&label=Follow&style=social" height=25>
    </a>
</p>

## Overview

[SparseZoo is a constantly-growing repository](https://sparsezoo.neuralmagic.com) of sparsified (pruned and pruned-quantized) models with matching sparsification recipes for neural networks. 
It simplifies and accelerates your time-to-value in building performant deep learning models with a collection of inference-optimized models and recipes to prototype from. 
Read more about sparsification [here](https://docs.neuralmagic.com/main/source/getstarted.html#sparsification).

Available via API and hosted in the cloud, the SparseZoo contains both baseline and models sparsified to different degrees of inference performance vs. baseline loss recovery. 
Recipe-driven approaches built around sparsification algorithms allow you to take the models as given, transfer-learn from the models onto private datasets, or transfer the recipes to your architectures.

The [GitHub repository](https://github.com/neuralmagic/sparsezoo) contains the Python API code to handle the connection and authentication to the cloud.

<img alt="SparseZoo Flow" src="https://docs.neuralmagic.com/docs/source/infographics/sparsezoo.png" width="960px" />

## Highlights

- [Model Stub Architecture Overview](https://github.com/neuralmagic/sparsezoo/blob/main/docs/source/models.md)
- [Available Model Recipes](https://github.com/neuralmagic/sparsezoo/blob/main/docs/source/recipes.md)
- [sparsezoo.neuralmagic.com](https://sparsezoo.neuralmagic.com)

## Installation

This repository is tested on Python 3.6+, and Linux/Debian systems.
It is recommended to install in a [virtual environment](https://docs.python.org/3/library/venv.html) to keep your system in order.

Install with pip using:

```bash
pip install sparsezoo
```

## Quick Tour

### Python APIS

The Python APIs respect this format enabling you to search and download models. Some code examples are given below.

#### Searching the Zoo

```python
from sparsezoo import Zoo

models = Zoo.search_models(domain="cv", sub_domain="classification")
print(models)
```

#### Common Models

```python
from sparsezoo.models.classification import resnet_50

model = resnet_50()
model.download()

print(model.onnx_file.downloaded_path())
```

#### Searching Optimized Versions

```python
from sparsezoo import Zoo
from sparsezoo.models.classification import resnet_50

search_model = resnet_50()
sparse_models = Zoo.search_sparse_models(search_model)

print(sparse_models)
```

### Console Scripts

In addition to the Python APIs, a console script entry point is installed with the package `sparsezoo`.
This enables easy interaction straight from your console/terminal.

#### Downloading

Download command help

```shell script
sparsezoo.download -h
```

<br>Download ResNet-50 Model

```shell script
sparsezoo.download zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/base-none
```

<br>Download pruned and quantized ResNet-50 Model

```shell script
sparsezoo.download zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/pruned_quant-moderate
```

#### Searching

Search command help

```shell script
sparsezoo search -h
```

<br>Searching for all classification models in the computer vision domain

```shell script
sparsezoo search --domain cv --sub-domain classification
```

<br>Searching for all ResNet-50 models

```shell script
sparsezoo search --domain cv --sub-domain classification \
    --architecture resnet_v1 --sub-architecture 50
```

For a more in-depth read, check out [SparseZoo documentation](https://docs.neuralmagic.com/sparsezoo/).

## Resources

### Learning More

- Documentation: [SparseML](https://docs.neuralmagic.com/sparseml/), [SparseZoo](https://docs.neuralmagic.com/sparsezoo/), [Sparsify](https://docs.neuralmagic.com/sparsify/), [DeepSparse](https://docs.neuralmagic.com/deepsparse/)
- Neural Magic: [Blog](https://www.neuralmagic.com/blog/), [Resources](https://www.neuralmagic.com/resources/)

### Release History

Official builds are hosted on PyPI

- stable: [sparsezoo](https://pypi.org/project/sparsezoo/)
- nightly (dev): [sparsezoo-nightly](https://pypi.org/project/sparsezoo-nightly/)

Additionally, more information can be found via [GitHub Releases.](https://github.com/neuralmagic/sparsezoo/releases)

### License

The project is licensed under the [Apache License Version 2.0](https://github.com/neuralmagic/sparsezoo/blob/main/LICENSE).

## Community

### Contribute

We appreciate contributions to the code, examples, integrations, and documentation as well as bug reports and feature requests! [Learn how here](https://github.com/neuralmagic/sparsezoo/blob/main/CONTRIBUTING.md).

### Join

For user help or questions about SparseZoo, sign up or log in: **Deep Sparse Community** [Discourse Forum](https://discuss.neuralmagic.com/) and/or [Slack](https://join.slack.com/t/discuss-neuralmagic/shared_invite/zt-q1a1cnvo-YBoICSIw3L1dmQpjBeDurQ). 
We are growing the community member by member and happy to see you there.

You can get the latest news, webinar and event invites, research papers, and other ML Performance tidbits by [subscribing](https://neuralmagic.com/subscribe/) to the Neural Magic community.

For more general questions about Neural Magic, please fill out this [form](http://neuralmagic.com/contact/).
