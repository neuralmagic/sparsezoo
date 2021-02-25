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

## Models

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

### Image Classification

|  Model Tag                                                                                 | Validation Baseline Metric |
| ------------------------------------------------------------------------------------------ | -------------------------- |
| cv/classification/efficientnet-b0/pytorch/sparseml/imagenet/base-none                      | 77.3% top1 accuracy        |
| cv/classification/efficientnet-b0/pytorch/sparseml/imagenet/arch-moderate                  | 76.5% top1 accuracy        |  
| cv/classification/efficientnet-b4/pytorch/sparseml/imagenet/base-none                      | 83.0% top1 accuracy        |  
| cv/classification/efficientnet-b4/pytorch/sparseml/imagenet/arch-moderate                  | 82.1% top1 accuracy        |  
| cv/classification/inception_v3/pytorch/sparseml/imagenet/base-none                         | 77.4% top1 accuracy        |  
| cv/classification/inception_v3/pytorch/sparseml/imagenet/pruned-conservative               | 77.4% top1 accuracy        |  
| cv/classification/inception_v3/pytorch/sparseml/imagenet/pruned-moderate                   | 76.6% top1 accuracy        |  
| cv/classification/mnistnet/pytorch/sparseml/mnist/base-none                                | 99.4% top1 accuracy        |  
| cv/classification/mobilenet_v1-1.0/pytorch/sparseml/imagenet/base-none                     | 70.9% top1 accuracy        |  
| cv/classification/mobilenet_v1-1.0/pytorch/sparseml/imagenet/pruned-conservative           | 70.9% top1 accuracy        |  
| cv/classification/mobilenet_v1-1.0/pytorch/sparseml/imagenet/pruned-moderate               | 70.1% top1 accuracy        |  
| cv/classification/mobilenet_v1-1.0/pytorch/sparseml/imagenet/pruned_quant-moderate         | 70.1% top1 accuracy        |  
| cv/classification/mobilenet_v2-1.0/pytorch/sparseml/imagenet/base-none                     | 71.9% top1 accuracy        |  
| cv/classification/resnet_v1-101/pytorch/sparseml/imagenet/base-none                        | 77.4% top1 accuracy        |  
| cv/classification/resnet_v1-101/pytorch/sparseml/imagenet/pruned-moderate                  | 76.6% top1 accuracy        |  
| cv/classification/resnet_v1-101/pytorch/torchvision/imagenet/base-none                     | 76.6% top1 accuracy        |  
| cv/classification/resnet_v1-101_2x/pytorch/sparseml/imagenet/base-none                     | 78.8% top1 accuracy        |  
| cv/classification/resnet_v1-101_2x/pytorch/torchvision/imagenet/base-none                  | 78.8% top1 accuracy        |  
| cv/classification/resnet_v1-152/pytorch/sparseml/imagenet/base-none                        | 78.3% top1 accuracy        |  
| cv/classification/resnet_v1-152/pytorch/sparseml/imagenet/pruned-moderate                  | 77.5% top1 accuracy        |  
| cv/classification/resnet_v1-152/pytorch/torchvision/imagenet/base-none                     | 77.5% top1 accuracy        |  
| cv/classification/resnet_v1-18/pytorch/sparseml/imagenet/base-none                         | 69.8% top1 accuracy        |  
| cv/classification/resnet_v1-18/pytorch/sparseml/imagenet/pruned-conservative               | 69.8% top1 accuracy        |  
| cv/classification/resnet_v1-18/pytorch/torchvision/imagenet/base-none                      | 69.8% top1 accuracy        |  
| cv/classification/resnet_v1-34/pytorch/sparseml/imagenet/base-none                         | 73.3% top1 accuracy        |  
| cv/classification/resnet_v1-34/pytorch/sparseml/imagenet/pruned-conservative               | 73.3% top1 accuracy        |  
| cv/classification/resnet_v1-34/pytorch/torchvision/imagenet/base-none                      | 73.3% top1 accuracy        |  
| cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/base-none                         | 76.1% top1 accuracy        |  
| cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/pruned-conservative               | 76.1% top1 accuracy        |  
| cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/pruned-moderate                   | 75.3% top1 accuracy        |  
| cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/pruned_quant-moderate             | 75.4% top1 accuracy        |  
| cv/classification/resnet_v1-50/pytorch/sparseml/imagenet-augmented/pruned_quant-aggressive | 76.1% top1 accuracy        |  
| cv/classification/resnet_v1-50/pytorch/sparseml/imagenette/base-none                       | 99.9% top1 accuracy        |  
| cv/classification/resnet_v1-50/pytorch/sparseml/imagenette/pruned-conservative             | 99.9% top1 accuracy        |  
| cv/classification/resnet_v1-50/pytorch/torchvision/imagenet/base-none                      | 99.9% top1 accuracy        |  
| cv/classification/resnet_v1-50/pytorch/torchvision/imagenette/pruned-conservative          | 99.9% top1 accuracy        |  
| cv/classification/resnet_v1-50_2x/pytorch/sparseml/imagenet/base-none                      | 78.1% top1 accuracy        |  
| cv/classification/resnet_v1-50_2x/pytorch/torchvision/imagenet/base-none                   | 78.1% top1 accuracy        |  
| cv/classification/vgg-11/pytorch/sparseml/imagenet/base-none                               | 69.0% top1 accuracy        |  
| cv/classification/vgg-11/pytorch/sparseml/imagenet/pruned-moderate                         | 68.3% top1 accuracy        |  
| cv/classification/vgg-11/pytorch/torchvision/imagenet/base-none                            | 68.3% top1 accuracy        |  
| cv/classification/vgg-11_bn/pytorch/sparseml/imagenet/base-none                            | 70.4% top1 accuracy        |  
| cv/classification/vgg-11_bn/pytorch/torchvision/imagenet/base-none                         | 70.4% top1 accuracy        |  
| cv/classification/vgg-13/pytorch/sparseml/imagenet/base-none                               | 69.9% top1 accuracy        |  
| cv/classification/vgg-13/pytorch/torchvision/imagenet/base-none                            | 69.9% top1 accuracy        |  
| cv/classification/vgg-13_bn/pytorch/sparseml/imagenet/base-none                            | 71.5% top1 accuracy        |  
| cv/classification/vgg-13_bn/pytorch/torchvision/imagenet/base-none                         | 71.5% top1 accuracy        |  
| cv/classification/vgg-16/pytorch/sparseml/imagenet/base-none                               | 71.6% top1 accuracy        |  
| cv/classification/vgg-16/pytorch/sparseml/imagenet/pruned-conservative                     | 71.6% top1 accuracy        |  
| cv/classification/vgg-16/pytorch/sparseml/imagenet/pruned-moderate                         | 70.8% top1 accuracy        |  
| cv/classification/vgg-16/pytorch/torchvision/imagenet/base-none                            | 70.8% top1 accuracy        |  
| cv/classification/vgg-16_bn/pytorch/sparseml/imagenet/base-none                            | 71.6% top1 accuracy        |  
| cv/classification/vgg-16_bn/pytorch/torchvision/imagenet/base-none                         | 71.6% top1 accuracy        |  
| cv/classification/vgg-19/pytorch/sparseml/imagenet/base-none                               | 72.4% top1 accuracy        |  
| cv/classification/vgg-19/pytorch/sparseml/imagenet/pruned-moderate                         | 71.7% top1 accuracy        |  
| cv/classification/vgg-19/pytorch/torchvision/imagenet/base-none                            | 71.7% top1 accuracy        |  
| cv/classification/vgg-19_bn/pytorch/sparseml/imagenet/base-none                            | 74.2% top1 accuracy        |  
| cv/classification/vgg-19_bn/pytorch/torchvision/imagenet/base-none                         | 74.2% top1 accuracy        | 

### Object Detection

|  Model Tag                                                                                 | Validation Baseline Metric |
| ------------------------------------------------------------------------------------------ | -------------------------- |
| cv/detection/ssd-resnet50_300/pytorch/sparseml/coco/base-none                              | 42.7 mAP@0.5               |
| cv/detection/ssd-resnet50_300/pytorch/sparseml/coco/pruned-moderate                        | 41.8 mAP@0.5               |
| cv/detection/ssd-resnet50_300/pytorch/sparseml/voc/base-none                               | 52.2 mAP@0.5               |
| cv/detection/ssd-resnet50_300/pytorch/sparseml/voc/pruned-moderate                         | 51.5 mAP@0.5               |
| cv/detection/yolo_v3-spp/pytorch/ultralytics/coco/base-none                                | 63.5 mAP@0.5               |
| cv/detection/yolo_v3-spp/pytorch/ultralytics/coco/pruned-aggressive                        | 62.1 mAP@0.5               |
