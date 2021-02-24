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
