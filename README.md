<!---
Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# ![icon for SparseZoo](docs/icon-sparsezoo.png) SparseZoo

### Neural network model repository for highly sparse models and sparsification recipes

<p>
    <a href="https://github.com/neuralmagic/comingsoon/blob/master/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/neuralmagic/comingsoon.svg?color=purple&style=for-the-badge" height=25>
    </a>
    <a href="https://docs.neuralmagic.com/sparsezoo/index.html">
        <img alt="Documentation" src="https://img.shields.io/website/http/neuralmagic.com/sparsezoo/index.html.svg?down_color=red&down_message=offline&up_message=online&style=for-the-badge" height=25>
    </a>
    <a href="https://github.com/neuralmagic/sparsezoo/releases">
        <img alt="GitHub release" src="https://img.shields.io/github/release/neuralmagic/sparsezoo.svg?style=for-the-badge" height=25>
    </a>
    <a href="https://github.com/neuralmagic.com/comingsoon/blob/master/CODE_OF_CONDUCT.md">
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

SparseZoo is a constantly-growing repository of optimized models. It simplifies and accelerates your time-to-value in building performant deep learning models with a collection of pre-trained, performance-optimized models to prototype from.

Available via API and hosted in the cloud, the SparseZoo contains both baseline and models optimized to different degrees of performance and recovery. With recipe-driven approaches, you can benchmark from the SparseZoo and transfer learn from your private data sets or other open source data sets.

This repository contains the Python API code to handle the connection and authentication to the cloud.

## Quick Tour and Documentation

[TODO ENGINEERING: EDIT THE CONTENT PLACEHOLDERS AS NEEDED]

Follow the quick tour below to get started.
For a more in-depth read, check out [SparseZoo documentation](https://docs.neuralmagic.com/sparsezoo/).

### Requirements

- This repository is tested on Python 3.6+, PyTorch base-ver+ and TensorFlow 1.x+
- Use Case: Computer Vision - Image Classification, Object Detection
- Model Architectures: Deep Learning Neural Network Architectures (e.g., CNNs, DNNs - refer to [SparseZoo](https://docs.neuralmagic.com/sparsezoo/) for examples)
- Instruction Set: CPUs with AVX2 or AVX-512 (best); (e.g., Intel Xeon Cascade Lake, Icelake, Skylake; AMD) and 2 FMAs. VNNI support required for sparse quantization.
- OS / Environment: Linux

### Installation

To install, run:

```bash
pip install sparsezoo
```

## Tutorials

[SparseZoo Tutorials](notebooks/) and [Use Cases](examples/) are provided for easily integrating and using the models in the SparseZoo. The APIs provided to interface with the SparseZoo are located in `neuralmagicML.utils`.

To retrieve all available models in the repo, you can use the `available_models` function. It returns a list of `RepoModel` objects.
Example code:

```python
from neuralmagicML.utils import available_models, RepoModel

models = available_models()  # type: List[RepoModel]
print(models)
```

## Available Models and Recipes

A number of pre-trained models are available in this API. Included are both baseline and recalibrated models for higher performance. These can optionally be used with the [DeepSparse Engine](https://github.com/neuralmagic/engine/). The types available for each model architecture are noted in the [SparseZoo model repository listing](docs/available-models.md).

## Resources and Learning More

- [SparseZoo Documentation](https://docs.neuralmagic.com/sparsezoo/), [Tutorials](notebooks/), [Use Cases](examples/)
- [DeepSparse Documentation](https://docs.neuralmagic.com/deepsparse/)
- [Neural Magic Blog](https://www.neuralmagic.com/blog/), [Resources](https://www.neuralmagic.com/resources/), [Website](https://www.neuralmagic.com/)

## Contributing

We appreciate contributions to the code, examples, and documentation as well as bug reports and feature requests! [Learn how here](CONTRIBUTING.md).

## Join the Community

For user help or questions about SparseML, use our [GitHub Discussions](https://www.github.com/neuralmagic/sparsezoo/discussions/). Everyone is welcome!

You can get the latest news, webinar and event invites, research papers, and other ML Performance tidbits by [subscribing](https://neuralmagic.com/subscribe/) to the Neural Magic community.

For more general questions about Neural Magic, please email us at [learnmore@neuralmagic.com](mailto:learnmore@neuralmagic.com) or fill out this [form](http://neuralmagic.com/contact/).

## License

The project is licensed under the [Apache License Version 2.0](LICENSE).

## Release History

[Track this project via GitHub Releases.](https://github.com/neuralmagic/sparsezoo/releases)
