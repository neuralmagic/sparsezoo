..
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

===================
SparseZoo |version|
===================

Neural network model repository for highly sparse and sparse-quantized models with matching sparsification recipes

.. raw:: html

    <div style="margin-bottom:16px;">
        <a href="https://github.com/neuralmagic/sparsezoo/blob/main/LICENSE">
            <img alt="GitHub" src="https://img.shields.io/github/license/neuralmagic/sparsezoo.svg?color=purple&style=for-the-badge" height=25 style="margin-bottom:4px;">
        </a>
        <a href="https://docs.neuralmagic.com/sparsezoo/index.html">
            <img alt="Documentation" src="https://img.shields.io/website/http/docs.neuralmagic.com/sparsezoo/index.html.svg?down_color=red&down_message=offline&up_message=online&style=for-the-badge" height=25 style="margin-bottom:4px;">
        </a>
        <a href="https://github.com/neuralmagic/sparsezoo/releases">
            <img alt="GitHub release" src="https://img.shields.io/github/release/neuralmagic/sparsezoo.svg?style=for-the-badge" height=25 style="margin-bottom:4px;">
        </a>
        <a href="https://github.com/neuralmagic/sparsezoo/blob/main/CODE_OF_CONDUCT.md">
            <img alt="Contributor Covenant" src="https://img.shields.io/badge/Contributor%20Covenant-v2.0%20adopted-ff69b4.svg?color=yellow&style=for-the-badge" height=25 style="margin-bottom:4px;">
        </a>
         <a href="https://www.youtube.com/channel/UCo8dO_WMGYbWCRnj_Dxr4EA">
            <img src="https://img.shields.io/badge/-YouTube-red?&style=for-the-badge&logo=youtube&logoColor=white" height=25 style="margin-bottom:4px;">
        </a>
         <a href="https://medium.com/limitlessai">
            <img src="https://img.shields.io/badge/medium-%2312100E.svg?&style=for-the-badge&logo=medium&logoColor=white" height=25 style="margin-bottom:4px;">
        </a>
        <a href="https://twitter.com/neuralmagic">
            <img src="https://img.shields.io/twitter/follow/neuralmagic?color=darkgreen&label=Follow&style=social" height=25 style="margin-bottom:4px;">
        </a>
     </div>

Overview
========

SparseZoo is a constantly-growing repository of highly sparse and sparse-quantized models with matching sparsification recipes for neural networks.
It simplifies and accelerates your time-to-value in building performant deep learning models with a collection of inference-optimized models and recipes to prototype from.

Available via API and hosted in the cloud, the SparseZoo contains both baseline and models optimized to different degrees of inference performance vs. baseline loss recovery.
Recipe-driven approaches built around sparsification algorithms allow you to take the models as given, transfer-learn from the models onto private datasets, or transfer the recipes to your architectures.

`This repository <https://github.com/neuralmagic/sparsezoo>`_ contains the Python API code to handle the connection and authentication to the cloud.

Sparsification
==============

Sparsification is the process of taking a trained deep learning model and removing redundant information from the overprecise and over-parameterized network resulting in a faster and smaller model.
Techniques for sparsification are all encompassing including everything from inducing sparsity using `pruning <https://neuralmagic.com/blog/pruning-overview/>`_ and `quantization <https://arxiv.org/abs/1609.07061>`_ to enabling naturally occurring sparsity using `activation sparsity <http://proceedings.mlr.press/v119/kurtz20a.html>`_ or `winograd/FFT <https://arxiv.org/abs/1509.09308>`_.
When implemented correctly, these techniques result in significantly more performant and smaller models with limited to no effect on the baseline metrics.
For example, pruning plus quantization can give noticeable improvements in performance while recovering to nearly the same baseline accuracy.

The Deep Sparse product suite builds on top of sparsification enabling you to easily apply the techniques to your datasets and models using recipe-driven approaches.
Recipes encode the directions for how to sparsify a model into a simple, easily editable format.

- Download a sparsification recipe and sparsified model from the `SparseZoo <https://github.com/neuralmagic/sparsezoo>`_.
- Alternatively, create a recipe for your model using `Sparsify <https://github.com/neuralmagic/sparsify>`_.
- Apply your recipe with only a few lines of code using `SparseML <https://github.com/neuralmagic/sparseml>`_.
- Finally, for GPU-level performance on CPUs, deploy your sparse-quantized model with the `DeepSparse Engine <https://github.com/neuralmagic/deepsparse>`_.


**Full Deep Sparse product flow:**  

.. raw:: html

    <img src="https://docs.neuralmagic.com/docs/source/sparsification/flow-overview.svg" width="960px">

Resources and Learning More
===========================

- `SparseML Documentation <https://docs.neuralmagic.com/sparseml>`_
- `Sparsify Documentation <https://docs.neuralmagic.com/sparsify>`_
- `DeepSparse Documentation <https://docs.neuralmagic.com/deepsparse>`_
- `Neural Magic Blog <https://www.neuralmagic.com/blog>`_,
  `Resources <https://www.neuralmagic.com/resources>`_,
  `Website <https://www.neuralmagic.com>`_

Release History
===============

Official builds are hosted on PyPI

- stable: `sparsezoo <https://pypi.org/project/sparsezoo>`_
- nightly (dev): `sparsezoo-nightly <https://pypi.org/project/sparsezoo-nightly>`_

Additionally, more information can be found via
`GitHub Releases <https://github.com/neuralmagic/sparsezoo/releases>`_.

.. toctree::
    :maxdepth: 3
    :caption: General

    source/quicktour
    source/installation
    source/models
    source/recipes

.. toctree::
    :maxdepth: 2
    :caption: API

    api/sparsezoo

.. toctree::
    :maxdepth: 3
    :caption: Connect Online

    Bugs, Feature Requests <https://github.com/neuralmagic/sparsezoo/issues>
    Support, General Q&A Forums <https://discuss.neuralmagic.com/>
    Deep Sparse Slack Community <https://discuss-neuralmagic.slack.com/>
    Neural Magic Docs <https://docs.neuralmagic.com>

