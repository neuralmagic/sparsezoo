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

# ![icon for SparseZoo](https://raw.githubusercontent.com/neuralmagic/sparsezoo/main/docs/source/icon-sparsezoo.png) SparseZoo

### Neural network model repository for highly sparse and sparse-quantized models with matching sparsification recipes

<p>
    <a href="https://docs.neuralmagic.com/sparsezoo/index.html">
        <img alt="Documentation" src="https://img.shields.io/website/http/docs.neuralmagic.com/sparsezoo/index.html.svg?down_color=red&down_message=offline&up_message=online&style=for-the-badge" height=25>
    </a>
    <a href="https://github.com/neuralmagic/sparsezoo/actions/workflows/quality-check.yaml">
        <img alt="Quality Check" src="https://img.shields.io/github/workflow/status/neuralmagic/sparsezoo/Quality%20Checks/main?label=Quality%20Checks&style=for-the-badge" height=25>
    </a>
 </p>
<p>
    <a href="https://github.com/neuralmagic/sparsezoo/blob/main/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/neuralmagic/sparsezoo.svg?color=purple&style=for-the-badge" height=25>
    </a>
    <a href="https://github.com/neuralmagic/sparsezoo/releases">
        <img alt="GitHub release" src="https://img.shields.io/github/release/neuralmagic/sparsezoo.svg?style=for-the-badge" height=25>
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

SparseZoo is a constantly-growing repository of highly sparse and sparse-quantized models with matching sparsification recipes for neural networks. 
It simplifies and accelerates your time-to-value in building performant deep learning models with a collection of inference-optimized models and recipes to prototype from.

Available via API and hosted in the cloud, the SparseZoo contains both baseline and models optimized to different degrees of inference performance vs. baseline loss recovery. 
Recipe-driven approaches built around sparsification algorithms allow you to take the models as given, transfer-learn from the models onto private datasets, or transfer the recipes to your architectures.

This repository contains the Python API code to handle the connection and authentication to the cloud.


## Sparsification

Sparsification is the process of taking a trained deep learning model and removing redundant information from the overprecise and over-parameterized network resulting in a faster and smaller model.
Techniques for sparsification are all encompassing including everything from inducing sparsity using [pruning](https://neuralmagic.com/blog/pruning-overview/) and [quantization](https://arxiv.org/abs/1609.07061) to enabling naturally occurring sparsity using [activation sparsity](http://proceedings.mlr.press/v119/kurtz20a.html) or [winograd/FFT](https://arxiv.org/abs/1509.09308). 
When implemented correctly, these techniques result in significantly more performant and smaller models with limited to no effect on the baseline metrics.
For example, pruning plus quantization can give noticeable improvements in performance while recovering to nearly the same baseline accuracy.

The Deep Sparse product suite builds on top of sparsification enabling you to easily apply the techniques to your datasets and models using recipe-driven approaches.
Recipes encode the directions for how to sparsify a model into a simple, easily editable format.
- Download a sparsification recipe and sparsified model from the [SparseZoo](https://github.com/neuralmagic/sparsezoo).
- Alternatively, create a recipe for your model using [Sparsify](https://github.com/neuralmagic/sparsify).
- Apply your recipe with only a few lines of code using [SparseML](https://github.com/neuralmagic/sparseml).
- Finally, for GPU-level performance on CPUs, deploy your sparse-quantized model with the [DeepSparse Engine](https://github.com/neuralmagic/deepsparse).


**Full Deep Sparse product flow:**  

<img src="https://docs.neuralmagic.com/docs/source/sparsification/flow-overview.svg" width="960px">

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
optimized_models = Zoo.search_optimized_models(search_model)

print(optimized_models)
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
    --optim-name base --optim-category none
```

<br>Download pruned and quantized ResNet-50 Model

```shell script
sparsezoo download --domain cv --sub-domain classification \
    --architecture resnet_v1 --sub-architecture 50 \
    --framework pytorch --repo sparseml \
    --dataset imagenet --training-scheme augmented \
    --optim-name pruned_quant --optim-category aggressive
```

For a more in-depth read, check out [SparseZoo documentation](https://docs.neuralmagic.com/sparsezoo/).

## Installation

This repository is tested on Python 3.6+, and Linux/Debian systems.
It is recommended to install in a [virtual environment](https://docs.python.org/3/library/venv.html) to keep your system in order.

Install with pip using:

```bash
pip install sparsezoo
```

Then if you would like to explore any of the [scripts](https://github.com/neuralmagic/sparsezoo/blob/main/scripts/) or [notebooks](https://github.com/neuralmagic/sparsezoo/blob/main/notebooks/)
clone the repository and install any additional dependencies as required.

## Available Models and Recipes

A number of pre-trained baseline and sparsified models across domains and sub domains are available and constantly being added.
For an up to date list, please consult the [available models listing](https://github.com/neuralmagic/sparsezoo/blob/main/docs/source/models.md).

## Resources and Learning More

- [SparseZoo Documentation](https://docs.neuralmagic.com/sparsezoo/)
- [SparseML Documentation](https://docs.neuralmagic.com/sparseml/)
- [Sparsify Documentation](https://docs.neuralmagic.com/sparsify/)
- [DeepSparse Documentation](https://docs.neuralmagic.com/deepsparse/)
- Neural Magic [Blog](https://www.neuralmagic.com/blog/), [Resources](https://www.neuralmagic.com/resources/), [Website](https://www.neuralmagic.com/)

## Contributing

We appreciate contributions to the code, examples, and documentation as well as bug reports and feature requests! [Learn how here](https://github.com/neuralmagic/sparsezoo/blob/main/CONTRIBUTING.md).

## Join the Community

For user help or questions about SparseZoo, use our [GitHub Discussions](https://www.github.com/neuralmagic/sparsezoo/discussions/). Everyone is welcome!

You can get the latest news, webinar and event invites, research papers, and other ML Performance tidbits by [subscribing](https://neuralmagic.com/subscribe/) to the Neural Magic community.

For more general questions about Neural Magic, please email us at [learnmore@neuralmagic.com](mailto:learnmore@neuralmagic.com) or fill out this [form](http://neuralmagic.com/contact/).

## License

The project is licensed under the [Apache License Version 2.0](https://github.com/neuralmagic/sparsezoo/blob/main/LICENSE).

## Release History

Official builds are hosted on PyPi

- stable: [sparsezoo](https://pypi.org/project/sparsezoo/)
- nightly (dev): [sparsezoo-nightly](https://pypi.org/project/sparsezoo-nightly/)

Additionally, more information can be found via [GitHub Releases.](https://github.com/neuralmagic/sparsezoo/releases)
