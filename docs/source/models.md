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
 
- `training`: The directory containing checkpoints. Every checkpoint contains a set of files required to load 
   a model in the specific state (e.g. directly after pruning). The checkpoint stores the trained model in the 
   native ML framework in which it was originally trained such as PyTorch, Keras, Tensorflow V1.
- `deployment`: The directory containing all the files necessary for the deployment of the model within an inference
   pipeline. 
- `logs`: The directory containing the artifacts generated during the training flow that helps to track the 
   reproducibility and audibility. Optional directory.
- `model.onnx`: The [ONNX](https://onnx.ai/) representation of the model's graph.
- `onnx`: The directory to store different opset representations of the `model.onnx`. Optional directory.
- `model.md`: The model card containing metadata, descriptions, and information for the model.
- `benchmarks.yaml`: The information about the performance of the `model.onnx` on given hardware systems. Optional file.
- `metrics.yaml`: Reporting metrics such as accuracy for the model on the given datasets such as validation and training. Optional file.
- `recipes`: The directory containing the recipes - the original sparsification recipe (`recipe_original.md`) 
   or others(e.g. transfer learning recipe).
- `sample_originals`: The original sample data without any pre-processing for use with the model.
- `sample_inputs`: The sample data after pre-processing for use with the model.
- `sample_outputs`: The outputs after running the sample inputs through the model.
- `sample_labels`: The labels that classify the sample inputs.
- `sample_originals`: The unedited data that can be used as inputs to a training pipeline (images, text files, numpy arrays, etc).

