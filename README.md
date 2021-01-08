<!---
Copyright 2021 Neuralmagic, Inc. All rights reserved.

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

# [Related Icon Here] SparseZoo

### Neural network model repository for pre-optimized models and optimization recipes

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

Simplify time to value and reduce skill burden to build performant deep learning models by having a collection of pre-trained, performance-optimized deep learning models to prototype from. The growing repository consists of popular image classification and object detection models and is constantly growing. They have been sparsified with the latest pruning techniques to deliver exceptional performance on CPUs, and accelerates the process of deploying those models in production. Neural Magic already did the hard work of building, pruning, and re-training the models for immediate use in production. We invite others to contribute to the SparseZoo!

Available via API and hosted in the cloud, there are a number of baseline and recalibrated models for higher performance. The SparseZoo repository contains the Python API code to handle the connection and authentication to the cloud.

Models in the SparseZoo can be transfer-learned with your data, and then recalibrated. This approach makes it easier for folks to use pre-optimized models with their own data, in their own environments. In addition to freely using these models, new models and recipes are very much welcomed!

## Quick Tour and Documentation

Follow the quick tour below to get started.
For a more in-depth read, check out [SparseZoo documentation](https://docs.neuralmagic.com/sparsezoo/).

### Installation and Requirements
[TODO ENGINEERING: installation instructions]

SparseZoo is OS-agnostic. Requires Python 3.6 or higher. 
```python
$ pip install sparsezoo
```
### Downloading and Usage
Tutorial notebooks and use cases are provided for easily integrating and using the models in the SparseZoo. 
Check the ??? [Tutorials section](INSERT PATH HERE) for more details. [TODO ENGINEERING: CLARIFY HOW TUTORIALS WILL BE ADDED IF ANY]
The APIs provided to interface with the SparseZoo are located in `neuralmagicML.utils`. 

To retrieve all available models in the repo, you can use the `available_models` function. 
It returns a list of `RepoModel` objects.
Example code:
```python
from neuralmagicML.utils import available_models, RepoModel

models = available_models()  # type: List[RepoModel]
print(models)
```

### Available Models and Recipes
A number of pre-trained models are available in this API. Included are both baseline and recalibrated models for higher performance. These can optionally be used with [Neural Magic Inference Engine](https://github.com/neuralmagic/engine/). The types available for each model architecture are noted in the [SparseZoo model repository listing](docs/available-models.md).

### Development Setup
[TODO ENGINEERING: dev instructions or point to CONTRIBUTING.md]

## Resources and Learning More
* [SparseZoo Documentation](https://docs.neuralmagic.com/sparsezoo/).
* [SparseZoo Use Cases](INSERT PATH HERE).
* [SparseZoo Examples] Coming soon in February 2021
* [Neural Magic Blog](https://www.neuralmagic.com/blog/)
* [Neural Magic](https://www.neuralmagic.com/)

[TODO ENGINEERING: table with links for deeper topics or other links that should be included above]

## Contributing

We appreciate contributions to the code, documentation and examples, documentation!

- Report issues and bugs directly in [this GitHub project](https://github.com/neuralmagic/sparsezoo/issues).
- Learn how to work with the SparseZoo source code, including building and testing SparseZoo models and recipes as well as contributing code changes to SparseZoo by reading our [Development and Contribution guidelines](CONTRIBUTING.md).

Give SparseZoo a shout out on social! Are you able write a blog post, do a lunch ’n learn, host a meetup, or simply share via your networks? Help us build the community, yay! Here’s some details to assist:
- item 1 [TODO MARKETING: NEED METHODS]
- item n

## Join the Community

For user help or questions about SparseZoo, please use our [GitHub Discussions](https://www.github.com/neuralmagic/sparsezoo/issues). Everyone is welcome!

You can get the latest news, webinar invites, and other ML Performance tidbits by [connecting with the Neural Magic community](https://www.neuralmagic.com/NEED_URL/).[TODO MARKETING: NEED METHOD]

For more general questions about Neural Magic please contact us this way [Method](URL). [TODO MARKETING: NEED METHOD]

[TODO MARKETING: Example screenshot here]

## <a name=“license”></a> License

The project is licensed under the [Apache License Version 2.0](LICENSE).

## <a name=“release”></a> Release History

[Track this project via GitHub Releases.](https://github.com/neuralmagic/sparsezoo/releases)
