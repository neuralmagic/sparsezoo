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

SparseZoo is a constantly-growing repository of sparsified (pruned and pruned-quantized) models with matching sparsification recipes for neural networks. 
It simplifies and accelerates your time-to-value in building performant deep learning models with a collection of inference-optimized models and recipes to prototype from. 
Read more about sparsification [here](https://docs.neuralmagic.com/main/source/getstarted.html#sparsification).

Available via API and hosted in the cloud, the SparseZoo contains both baseline and models sparsified to different degrees of inference performance vs. baseline loss recovery. 
Recipe-driven approaches built around sparsification algorithms allow you to take the models as given, transfer-learn from the models onto private datasets, or transfer the recipes to your architectures.

The [GitHub repository](https://github.com/neuralmagic/sparsezoo) contains the Python API code to handle the connection and authentication to the cloud.

<img alt="SparseZoo Flow" src="https://docs.neuralmagic.com/docs/source/infographics/sparsezoo.png" width="960px" />

## Highlights

- [Available Models Listing](https://github.com/neuralmagic/sparsezoo/blob/main/docs/source/models.md)
- [Available Recipes Listing](https://github.com/neuralmagic/sparsezoo/blob/main/docs/source/recipes.md)

## Installation

This repository is tested on Python 3.6+, and Linux/Debian systems.
It is recommended to install in a [virtual environment](https://docs.python.org/3/library/venv.html) to keep your system in order.

Install with pip using:

```bash
pip install sparsezoo
```

## Quick Tour

Each model in the SparseZoo has a specific stub that identifies it. The stubs are made up of the following structure:

`DOMAIN/SUB_DOMAIN/ARCHITECTURE{-SUB_ARCHITECTURE}/FRAMEWORK/REPO/DATASET{-TRAINING_SCHEME}/SPARSE_NAME-SPARSE_CATEGORY-{SPARSE_TARGET}`

The properties within each model stub are defined as the following:

| Model Property   | Definition                                                                                     | Examples                                                                           |
|:----------------:|:----------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------:|
| DOMAIN           | The type of solution the model is architected and trained for                                  | cv, nlp                                                                            |
| SUB_DOMAIN       | The sub type of solution the model is architected and trained for                              | classification, segmentation                                                       |
| ARCHITECTURE     | The name of the guiding setup for the network's graph                                          | resnet_v1, mobilenet_v1                                                            |
| SUB_ARCHITECTURE | (optional) The scaled version of the architecture such as width or depth                       | 50, 101, 152                                                                       |
| FRAMEWORK        | The machine learning framework the model was defined and trained in                            | pytorch, tensorflow_v1                                                             |
| REPO             | The model repository the model and baseline weights originated from                            | sparseml, torchvision                                                              |
| DATASET          | The dataset the model was trained on                                                           | imagenet, cifar10                                                                  |
| TRAINING_SCHEME  | (optional) A description on how the model was trained                                          | augmented, lower_lr                                                                |
| SPARSE_NAME      | An overview of what was done to sparsify the model                                             | base, pruned, quant (quantized), pruned_quant, arch (architecture modified)        |
| SPARSE_CATEGORY  | Descriptor on the degree to which the model is sparsified as compared with the baseline metric | none, conservative (100% baseline), moderate (>= 99% baseline), aggressive (< 99%) |
| SPARSE_TARGET    | (optional) Descriptor for the target environment the model was sparsified for                  | disk, edge, deepsparse, gpu                                                        |

The contents of each model are made up of the following:

- model.md: The model card containing metadata, descriptions, and information for the model.
- model.onnx: The [ONNX](https://onnx.ai/) representation of the model's graph.
- model.onnx.tar.gz: A compressed format for the ONNX file. 
    Currently ONNX does not support sparse tensors and quantized sparse tensors well for compression.
- [FRAMEWORK]/model.[EXTENSION]: The native ML framework file(s) for the model in which it was originally trained.
    Such as PyTorch, Keras, TensorFlow V1
- recipes/original.[md|yaml]: The original sparsification recipe used to create the model.
- recipes/[NAME].[md|yaml]: Additional sparsification recipes that can be used with the model such as transfer learning.
- sample-originals: The original sample data without any preprocessing for use with the model.
- sample-inputs: The sample data after pre processing for use with the model.
- sample-outputs: The outputs after running the sample inputs through the model.
- sample-labels: The labels that classify the sample inputs.

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
Note, for some environments the console scripts cannot install properly.
If this happens for your system and the sparsezoo command is not available, 
`https://github.com/neuralmagic/sparsezoo/blob/main/scripts/sparsezoo.py` may be used in its place.

```shell script
sparsezoo -h
```

#### Searching

Search command help

```shell script
sparsezoo search -h
```

<br>Searching for all classification models in the computer vision domain

```shell script
sparsezoo search --domain cv --sub-domain classification \
    --architecture resnet_v1 --sub-architecture 50
```

<br>Searching for all ResNet-50 models

```shell script
sparsezoo search --domain cv --sub-domain classification
```

#### Downloading

Download command help

```shell script
sparsezoo download -h
```

<br>Download ResNet-50 Model

```shell script
sparsezoo download --domain cv --sub-domain classification \
    --architecture resnet_v1 --sub-architecture 50 \
    --framework pytorch --repo sparseml --dataset imagenet \
    --sparse-name base --sparse-category none
```

<br>Download pruned and quantized ResNet-50 Model

```shell script
sparsezoo download --domain cv --sub-domain classification \
    --architecture resnet_v1 --sub-architecture 50 \
    --framework pytorch --repo sparseml \
    --dataset imagenet --training-scheme augmented \
    --sparse-name pruned_quant --sparse-category aggressive
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
