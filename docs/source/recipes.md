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

## Recipes

Each recipe in the SparseZoo is stored under the model created with it and has a specific stub that identifies it. 
The stubs are made up of the following structure:

`DOMAIN/SUB_DOMAIN/ARCHITECTURE{-SUB_ARCHITECTURE}/FRAMEWORK/REPO/DATASET{-TRAINING_SCHEME}/SPARSE_NAME-SPARSE_CATEGORY-{SPARSE_TARGET}?recipe-type=RECIPE_TYPE`

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
| RECIPE_TYPE      | A named descriptor for the recipe signifying what the recipe is for                            | original, transfer_learn                                                           |

### Image Classification

|  Model Tag                                                                                                        | Validation Baseline Metric |
| ----------------------------------------------------------------------------------------------------------------- | -------------------------- |
|  cv/classification/efficientnet-b0/pytorch/sparseml/imagenet/arch-moderate?recipe_type=original                   |  76.5% top1 accuracy       |   
|  cv/classification/efficientnet-b4/pytorch/sparseml/imagenet/arch-moderate?recipe_type=original                   |  82.1% top1 accuracy       |   
|  cv/classification/inception_v3/pytorch/sparseml/imagenet/pruned-conservative?recipe_type=original                |  77.4% top1 accuracy       |  
|  cv/classification/inception_v3/pytorch/sparseml/imagenet/pruned-moderate?recipe_type=original                    |  76.6% top1 accuracy       |  
|  cv/classification/mobilenet_v1-1.0/pytorch/sparseml/imagenet/base-none?recipe_type=original                      |  70.9% top1 accuracy       |  
|  cv/classification/mobilenet_v1-1.0/pytorch/sparseml/imagenet/pruned-conservative?recipe_type=original            |  70.9% top1 accuracy       |  
|  cv/classification/mobilenet_v1-1.0/pytorch/sparseml/imagenet/pruned-moderate?recipe_type=original                |  70.1% top1 accuracy       |  
|  cv/classification/mobilenet_v1-1.0/pytorch/sparseml/imagenet/pruned_quant-moderate?recipe_type=original          |  70.1% top1 accuracy       |  
|  cv/classification/mobilenet_v1-1.0/pytorch/sparseml/imagenet/pruned_quant-moderate?recipe_type=original          |  70.1% top1 accuracy       | 
|  cv/classification/resnet_v1-101/pytorch/sparseml/imagenet/pruned-moderate?recipe_type=original                   |  76.6% top1 accuracy       |
|  cv/classification/resnet_v1-152/pytorch/sparseml/imagenet/pruned-moderate?recipe_type=original                   |  77.5% top1 accuracy       |
|  cv/classification/resnet_v1-18/pytorch/sparseml/imagenet/pruned-conservative?recipe_type=original                |  69.8% top1 accuracy       |
|  cv/classification/resnet_v1-34/pytorch/sparseml/imagenet/pruned-conservative?recipe_type=original                |  73.3% top1 accuracy       |
|  cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/pruned-conservative?recipe_type=original                |  76.1% top1 accuracy       |  
|  cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/pruned-moderate?recipe_type=original                    |  75.3% top1 accuracy       |  
|  cv/classification/resnet_v1-50/pytorch/sparseml/imagenet-augmented/pruned_quant-aggressive?recipe_type=original  |  76.1% top1 accuracy       |  
|  cv/classification/resnet_v1-50/pytorch/sparseml/imagenette/pruned-conservative?recipe_type=original              |  99.9% top1 accuracy       |  
|  cv/classification/resnet_v1-50/pytorch/torchvision/imagenette/pruned-conservative?recipe_type=original           |  99.9% top1 accuracy       |  
|  cv/classification/vgg-11/pytorch/sparseml/imagenet/pruned-moderate?recipe_type=original                          |  68.3% top1 accuracy       |  
|  cv/classification/vgg-16/pytorch/sparseml/imagenet/pruned-conservative?recipe_type=original                      |  71.6% top1 accuracy       |  
|  cv/classification/vgg-16/pytorch/sparseml/imagenet/pruned-moderate?recipe_type=original                          |  70.8% top1 accuracy       |  
|  cv/classification/vgg-19/pytorch/sparseml/imagenet/pruned-moderate?recipe_type=original                          |  71.7% top1 accuracy       |

### Object Detection

|  Model Tag                                                                                                        | Validation Baseline Metric |
| ----------------------------------------------------------------------------------------------------------------- | -------------------------- |
| cv/detection/ssd-resnet50_300/pytorch/sparseml/coco/pruned-moderate?recipe_type=original                          | 41.8 mAP@0.5               |
| cv/detection/ssd-resnet50_300/pytorch/sparseml/voc/pruned-moderate?recipe_type=original                           | 51.5 mAP@0.5               |
| cv/detection/yolo_v3-spp/pytorch/ultralytics/coco/pruned-aggressive?recipe_type=original                          | 62.1 mAP@0.5               |
