# [Related Icon Here] SparseZoo

Neural network model repository for preoptimized models and optimization recipes

[TODO: ENGINEERING: Badges for build info, license, website, release, etc.]

## Overview

Simplify time to value and reduce skill burden to build performant deep learning models by having a collection of pre-trained, performance-optimized deep learning models to prototype from. The growing repository consists of popular image classification and object detection models and is constantly growing. They have been sparsified with the latest pruning techniques to deliver exceptional performance on CPUs, and accelerates the process of deploying those models in production. Neural Magic already did the hard work of building, pruning, and re-training the models for immediate use in production. We invite others to contribute to the SparseZoo!

Available via API and hosted in the cloud, there are a number of baseline and recalibrated models for higher performance on the Neural Magic Inference Engine.[JEANNIE: DO WE NEED TO CITE NMIE? CAN WE REMOVE THIS REFERENCE? 
The SparseZoo repo contains the Python API code to handle the connection and authentication to the cloud. The types available for each model architecture are detailed in the table below. 

Existing models seeded from the SparseZoo can be transfer-learned with your data, and then recalibrated. This approach makes it easier for folks to use pre-optimized models with their own data, in their own environments. In addition to freely using these models, new models and recipes are very much welcomed!


# [CHECK TABLE OF CONTENTS LAST; REMOVE THIS HEADER]
 - [Join the Community](#community)
 - [Getting Started and Documentation](#documentation)
 - [Use Cases and Examples](#usecases)
 - [Installation and Requirements](#install)
 - [Development Setup](#dev)
 - [Models and Recipes](#models)
 - [Downloading and Usage](#downloading)
 - [Resources and Learning More](#resources)
 - [Join the Community](#community)
 - [Contributing](#contribute)
 - [Citation](#citation)
 - [Release History](#release)

## <a name=“documentation”></a> Getting Started and Documentation

* Follow the [SparseZoo QuickStart](https://docs.neuralmagic.com/sparsezoo/quickstart) to get started in just a few minutes.
* Read through the [SparseZoo documentation](https://docs.neuralmagic.com/sparsezoo).
* Check out some [SparseZoo tutorials and a quick tour](https://docs.neuralmagic.com/sparsezoo/tutorials) to get familiar.

[TODO MARKETING: Example screenshot here]

## <a name=“usecases”></a> Use Cases and Examples

### Use Case 1 Name/Topic

_<content here>_

### Use Case N Name/Topic

_<content here>_

## <a name=“install”></a> Installation and Requirements
[TODO ENGINEERING: installation instructions]

## <a name=“dev”></a> Development Setup
[TODO ENGINEERING: dev instructions]

## <a name=“models”></a> Models and Recipes

### Available Models
A number of pre-trained models are available in this API.
Included are both baseline and recalibrated models for higher performance on the Neural Magic Inference Engine. 
The types available for each model architecture are noted in the table below.

Possible types are:
 - base - the baseline model (standard training process)
 - recal - a recalibrated model for better performance that achieves ~100% of baseline validation metrics
 - recal-perf - a recalibrated model for better performance that meets ~99% of baseline validation metrics


|  Architecture       | Dataset  | Available Types         | Frameworks                 | Validation Baseline Metric |
| ------------------- | -------- | ----------------------- | -------------------------- | -------------------------- |
| MnistNet            | MNIST    | base                    | ONNX, PyTorch, TensorFlow  | ~99% top1 accuracy         |
| EfficientNet-B0     | ImageNet | base, recal-perf        | ONNX, PyTorch              | 77.3% top1 accuracy        |
| EfficientNet-B4     | ImageNet | base, recal-perf        | ONNX, PyTorch              | 83.0% top1 accuracy        |
| InceptionV3         | ImageNet | base, recal, recal-perf | ONNX, PyTorch              | 77.45% top1 accuracy       |
| MobileNetV1         | ImageNet | base, recal, recal-perf | ONNX, PyTorch, TensorFlow  | 70.9% top1 accuracy        |
| MobileNetV2         | ImageNet | base                    | ONNX, PyTorch, TensorFlow  | 71.88% top1 accuracy       |
| ResNet-18           | ImageNet | base, recal             | ONNX, PyTorch, TensorFlow  | 69.8% top1 accuracy        |
| ResNet-34           | ImageNet | base, recal             | ONNX, PyTorch, TensorFlow  | 73.3% top1 accuracy        |
| ResNet-50           | ImageNet | base, recal, recal-perf | ONNX, PyTorch, TensorFlow  | 76.1% top1 accuracy        |
| ResNet-50 2xwidth   | ImageNet | base                    | ONNX, PyTorch              | 78.51% top1 accuracy       |
| ResNet-101          | ImageNet | base, recal-perf        | ONNX, PyTorch, TensorFlow  | 77.37% top1 accuracy       |
| ResNet-101 2xwidth  | ImageNet | base                    | ONNX, PyTorch              | 78.84% top1 accuracy       |
| ResNet-152          | ImageNet | base, recal-perf        | ONNX, PyTorch, TensorFlow  | 78.31% top1 accuracy       |
| VGG-11              | ImageNet | base, recal-perf        | ONNX, PyTorch, TensorFlow  | 69.02% top1 accuracy       |
| VGG-11bn            | ImageNet | base                    | ONNX, PyTorch, TensorFlow  | 70.38% top1 accuracy       |
| VGG-13              | ImageNet | base                    | ONNX, PyTorch, TensorFlow  | 69.93% top1 accuracy       |
| VGG-13bn            | ImageNet | base                    | ONNX, PyTorch, TensorFlow  | 71.55% top1 accuracy       |
| VGG-16              | ImageNet | base, recal, recal-perf | ONNX, PyTorch, TensorFlow  | 71.59% top1 accuracy       |
| VGG-16bn            | ImageNet | base                    | ONNX, PyTorch, TensorFlow  | 71.55% top1 accuracy       |
| VGG-19              | ImageNet | base, recal-perf        | ONNX, PyTorch, TensorFlow  | 72.38% top1 accuracy       |
| VGG-19bn            | ImageNet | base                    | ONNX, PyTorch, TensorFlow  | 74.24% top1 accuracy       |
| SSD-300-ResNet-50   | COCO     | base, recal-perf        | ONNX, PyTorch              | 42.7% mAP@0.5              |
| SSD-300-ResNet-50   | VOC      | base, recal-perf        | ONNX, PyTorch              | 52.2% mAP@0.5              |
| SSDLite-MobileNetV2 | COCO     | base                    | ONNX, PyTorch              | 35.7% mAP@0.5              |
| SSDLite-MobileNetV2 | VOC      | base                    | ONNX, PyTorch              | 43.5% mAP@0.5              |
| YOLOv3              | COCO     | base, recal-perf        | ONNX, PyTorch              | 68.6% mAP@0.5              |

## <a name=“downloading”></a> Downloading and Usage
Tutorial notebooks are provided for easily integrating and using the models in the SparseZoo. 
Check the [Tutorials section](#tutorials) for more details. [TODO ENGINEERING: CLARIFY HOW NOTEBOOKS WILL BE ADDED]
The APIs provided to interface with the SparseZoo are located in `neuralmagicML.utils`. 

To retrieve all available models in the repo, you can use the `available_models` function. 
It returns a list of `RepoModel` objects.
Example code:
```python
from neuralmagicML.utils import available_models, RepoModel

models = available_models()  # type: List[RepoModel]
print(models)
```

## <a name=“resources”></a> Resources and Learning More
[TODO: links out to other products or items of related interest to component]

* [SparseZoo Tutorials](https://docs.neuralmagic.com/sparsezoo/tutorials/)
* [SparseZoo Examples](https://github.com/neuralmagic/sparsezoo/examples/)
* [Neural Magic Twitter](https://twitter.com/neuralmagic)
* [Neural Magic Blog](https://www.neuralmagic.com/blog/)
* [Neural Magic YouTube](https://www.youtube.com/channel/UCo8dO_WMGYbWCRnj_Dxr4EA)
* [Neural Magic](https://www.neuralmagic.com/)

[TODO ENGINEERING: table with links for deeper topics]

## <a name="community"></a> Join the Community

For user help or questions about SparseZOO, please use our [GitHub Discussions](https://www.github.com/neuralmagic/sparsezoo/issues). Everyone is welcome!

You can get the latest news, webinar invites, and other ML Performance tidbits by [connecting with the Neural Magic community](https://www.neuralmagic.com/NEED_URL/).[TODO MARKETING: NEED METHOD]

For more general questions about Neural Magic please contact us this way [Method](URL). [TODO MARKETING: NEED METHOD]

## <a name=“contribute”></a> Contributing

We appreciate contributions to the code, documentation and examples, documentation!

- Report issues and bugs directly in [this GitHub project](https://github.com/neuralmagic/sparsezoo/issues).
- Learn how to work with the SparseZoo source code, including building and testing SparseZoo models and recipes as well as contributing code changes to SparseZoo by reading our [Development and Contribution guidelines](CONTRIBUTING.md).
- One good way to get started is by tackling a [TODO ENGINEERING: WHAT OR HOW CAN A USER DO OUT OF THE GATE?]

Give the SparseZoo a shout out on social! Are you able write a blog post, do a lunch ’n learn, host a meetup, or simply share via your networks? Help us build the community, yay! Here’s some details to assist:
- item 1 [TODO MARKETING: NEED METHODS]
- item n

## <a name=“license”></a> License

The project is licensed under the [Apache License Version 2.0](LICENSE).

## <a name=“citation”></a> Citation

[TODO ENGINEERING: list out any citations]

## <a name=“release”></a> Release History

[TODO ENGINEERING: release history]
