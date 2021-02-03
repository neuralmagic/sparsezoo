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

Neural network model repository for highly sparse models and optimization recipes

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
        <a href="https://github.com/neuralmagic.com/sparsezoo/blob/main/CODE_OF_CONDUCT.md">
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

SparseZoo is a constantly-growing repository of optimized models and optimization recipes for neural networks.
It simplifies and accelerates your time-to-value in building performant deep learning models with a
collection of inference-optimized models and recipes to prototype from.

Available via API and hosted in the cloud, the SparseZoo contains both baseline and models optimized
to different degrees of inference performance vs baseline loss recovery.
Optimizations on neural networks include approaches such as
`pruning <https://neuralmagic.com/blog/pruning-overview />`_ and `quantization <https://arxiv.org/abs/1609.07061 />`_
allowing for significantly faster models with limited to no effect on their baseline metrics such as accuracy.
Recipe-driven approaches built around these optimizations allow you to take the models as given,
transfer learn from the models onto private datasets, or transfer the recipes to your architectures.

This repository contains the Python API code to handle the connection and authentication to the cloud.

Related Products
================

- `DeepSparse <https://github.com/neuralmagic/deepsparse />`_:
  CPU inference engine that delivers unprecedented performance for sparse models
- `SparseML <https://github.com/neuralmagic/sparseml />`_:
  Libraries for state-of-the-art deep neural network optimization algorithms,
  enabling simple pipelines integration with a few lines of code
- `Sparsify <https://github.com/neuralmagic/sparsify />`_:
  Easy-to-use autoML interface to optimize deep neural networks for
  better inference performance and a smaller footprint

Resources and Learning More
===========================

- `SparseML Documentation <https://docs.neuralmagic.com/sparseml/ />`_
- `Sparsify Documentation <https://docs.neuralmagic.com/sparsify/ />`_
- `DeepSparse Documentation <https://docs.neuralmagic.com/deepsparse/ />`_
- `Neural Magic Blog <https://www.neuralmagic.com/blog/ />`_,
  `Resources <https://www.neuralmagic.com/resources/ />`_,
  `Website <https://www.neuralmagic.com/ />`_

Release History
===============

Official builds are hosted on PyPi
- stable: `sparsezoo <https://pypi.org/project/sparsezoo/ />`_
- nightly (dev): `sparsezoo-nightly <https://pypi.org/project/sparsezoo-nightly/ />`_

Additionally, more information can be found via
`GitHub Releases <https://github.com/neuralmagic/sparsezoo/releases />`_.

.. toctree::
    :maxdepth: 3
    :caption: General

    quicktour
    installation
    models
    recipes

.. toctree::
    :maxdepth: 2
    :caption: API

    api/sparsezoo

.. toctree::
    :maxdepth: 2
    :caption: Help and Support

    `Bugs, Feature Requests <https://github.com/neuralmagic/sparsezoo/discussions>`_ 
    `Support, General Q&A <https://github.com/neuralmagic/sparsezoo/issues>`_ 
   