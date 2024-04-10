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

<h1 style="display: flex; align-items: center;" >
     <img width="100" height="100" alt="tool icon" src="https://neuralmagic.com/wp-content/uploads/2024/03/icon_SparseZoo-003.svg" />
      <span>&nbsp;&nbsp;SparseZoo</span>
  </h1>

<h3>Neural network model repository for highly sparse and sparse-quantized models with matching sparsification recipes</h3>

<p>
    <a href="https://docs.neuralmagic.com/sparsezoo">
        <img alt="Documentation" src="https://img.shields.io/badge/documentation-darkred?&style=for-the-badge&logo=read-the-docs" height=25>
    </a>
    <a href="https://neuralmagic.com/community/">
        <img src="https://img.shields.io/badge/slack-purple?style=for-the-badge&logo=slack" height=25>
    </a>
    <a href="https://github.com/neuralmagic/sparsezoo/issues">
        <img src="https://img.shields.io/badge/support%20forums-navy?style=for-the-badge&logo=github" height=25>
    </a>
    <a href="https://github.com/neuralmagic/sparsezoo/actions/workflows/test-check.yaml">
        <img alt="Main" src="https://img.shields.io/github/workflow/status/neuralmagic/sparsezoo/Test%20Checks/main?label=build&style=for-the-badge" height=25>
    </a>
    <a href="https://github.com/neuralmagic/sparsezoo/releases">
        <img alt="GitHub release" src="https://img.shields.io/github/release/neuralmagic/sparsezoo.svg?style=for-the-badge" height=25>
    </a>
    <a href="https://github.com/neuralmagic/sparsezoo/blob/main/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/neuralmagic/sparsezoo.svg?color=lightgray&style=for-the-badge" height=25>
    </a>
    <a href="https://github.com/neuralmagic/sparsezoo/blob/main/CODE_OF_CONDUCT.md">
        <img alt="Contributor Covenant" src="https://img.shields.io/badge/Contributor%20Covenant-v2.1%20adopted-ff69b4.svg?color=yellow&style=for-the-badge" height=25>
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

[SparseZoo is a constantly-growing repository](https://sparsezoo.neuralmagic.com) of sparsified (pruned and pruned-quantized) models with matching sparsification recipes for neural networks. 
It simplifies and accelerates your time-to-value in building performant deep learning models with a collection of inference-optimized models and recipes to prototype from. 
Read [more about sparsification](https://docs.neuralmagic.com/user-guides/sparsification).

Available via API and hosted in the cloud, the SparseZoo contains both baseline and models sparsified to different degrees of inference performance vs. baseline loss recovery. 
Recipe-driven approaches built around sparsification algorithms allow you to use the models as given, transfer-learn from the models onto private datasets, or transfer the recipes to your architectures.

The [GitHub repository](https://github.com/neuralmagic/sparsezoo) contains the Python API code to handle the connection and authentication to the cloud.

<img alt="SparseZoo Flow" src="https://docs.neuralmagic.com/docs/source/infographics/sparsezoo.png" width="960px" />


## 🚨 New SparseZoo Models 🚨
### 🎃 October 2023 🎃
**Generative AI**
- Sparse MPT Models - [21 variants](https://sparsezoo.neuralmagic.com/?architectures=mpt&ungrouped=true)
    - ⚡ Highlighted Model ⚡ :  [mpt-7b-gsm8k_mpt_pretrain-pruned80_quantized](https://sparsezoo.neuralmagic.com/models/mpt-7b-gsm8k_mpt_pretrain-pruned80_quantized?hardware=deepsparse-c6i.12xlarge&comparison=mpt-7b-gsm8k_mpt_pretrain-base)
- Sparse OPT Models - [12 variants](https://sparsezoo.neuralmagic.com/?architectures=opt&ungrouped=true)
    - ⚡ Highlighted Model ⚡ :  [opt-6.7b-opt_pretrain-pruned50_quantW8A8](https://sparsezoo.neuralmagic.com/models/opt-6.7b-opt_pretrain-pruned50_quantW8A8?hardware=deepsparse-c6i.12xlarge&comparison=opt-6.7b-opt_pretrain-base)
- Sparse Codegen (mono,multi) Models - [10 variants](https://sparsezoo.neuralmagic.com/?ungrouped=true&architectures=codegen_mono%2Ccodegen_multi)
    - ⚡ Highlighted Model ⚡ :  [codegen_multi-350m-bigquery_thepile-pruned50_quantized](https://sparsezoo.neuralmagic.com/models/codegen_multi-350m-bigquery_thepile-pruned50_quantized?hardware=deepsparse-c6i.12xlarge&comparison=codegen_multi-350m-bigquery_thepile-base) 
 

## Highlights

- [Model Stub Architecture Overview](https://docs.neuralmagic.com/sparsezoo/source/models.html)
- [Available Model Recipes](https://docs.neuralmagic.com/sparsezoo/source/recipes.html)
- [sparsezoo.neuralmagic.com](https://sparsezoo.neuralmagic.com)

## Installation

This repository is tested on Python 3.8-3.11, and Linux/Debian systems.
It is recommended to install in a [virtual environment](https://docs.python.org/3/library/venv.html) to keep your system in order.

Install with pip using:

```bash
pip install sparsezoo
```

## Quick Tour

The SparseZoo Python API enables you to search and download sparsified models. Code examples are given below.
We encourage users to load SparseZoo models by copying a stub directly from a [model page]((https://sparsezoo.neuralmagic.com/)).

### Introduction to Model Class Object

The `Model` is a fundamental object that serves as a main interface with the SparseZoo library. 
It represents a SparseZoo model, together with all its directories and files.

#### Creating a Model Class Object From SparseZoo Stub
```python
from sparsezoo import Model

stub = "zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/pruned95_quant-none"

model = Model(stub)
print(str(model))

>> Model(stub=zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/pruned95_quant-none)
```

#### Creating a Model Class Object From Local Model Directory
```python
from sparsezoo import Model

directory = ".../.cache/sparsezoo/eb977dae-2454-471b-9870-4cf38074acf0"

model = Model(directory)
print(str(model))

>> Model(directory=.../.cache/sparsezoo/eb977dae-2454-471b-9870-4cf38074acf0)
```

#### Manually Specifying the Model Download Path

Unless specified otherwise, the model created from the SparseZoo stub is saved to the local sparsezoo cache directory. 
This can be overridden by passing the optional `download_path` argument to the constructor:

```python
from sparsezoo import Model

stub = "zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/pruned95_quant-none"
download_directory = "./model_download_directory"

model = Model(stub, download_path = download_directory)
```
#### Downloading the Model Files
Once the model is initialized from a stub, it may be downloaded either by calling the `download()` method or by invoking a `path` property. Both pathways are universal for all the files in SparseZoo. Invoking the `path` property will always trigger file download unless the file has already been downloaded.

```python
# method 1
model.download() 

# method 2 
model_path = model.path
```

#### Inspecting the Contents of the SparseZoo Model

We call the `available_files` method to inspect which files are present in the SparseZoo model. Then, we select a file by calling the appropriate attribute:

```python
model.available_files

>> {'training': Directory(name=training), 
>> 'deployment': Directory(name=deployment), 
>> 'sample_inputs': Directory(name=sample_inputs.tar.gz), 
>> 'sample_outputs': {'framework': Directory(name=sample_outputs.tar.gz)}, 
>> 'sample_labels': Directory(name=sample_labels.tar.gz), 
>> 'model_card': File(name=model.md), 
>> 'recipes': Directory(name=recipe), 
>> 'onnx_model': File(name=model.onnx)}
```
Then, we might take a closer look at the contents of the SparseZoo model:
```python
model_card = model.model_card
print(model_card)

>> File(name=model.md)
```
```python
model_card_path = model.model_card.path
print(model_card_path)

>> .../.cache/sparsezoo/eb977dae-2454-471b-9870-4cf38074acf0/model.md
```


### Model, Directory, and File

In general, every file in the SparseZoo model shares a set of attributes: `name`, `path`, `URL`, and `parent`:
- `name` serves as an identifier of the file/directory
- `path` points to the location of the file/directory 
- `URL` specifies the server address of the file/directory in question
- `parent` points to the location of the parent directory of the file/directory in question

A directory is a unique type of file that contains other files. For that reason, it has an additional `files` attribute.

```python
print(model.onnx_model)

>> File(name=model.onnx)

print(f"File name: {model.onnx_model.name}\n"
      f"File path: {model.onnx_model.path}\n"
      f"File URL: {model.onnx_model.url}\n"
      f"Parent directory: {model.onnx_model.parent_directory}")
      
>> File name: model.onnx
>> File path: .../.cache/sparsezoo/eb977dae-2454-471b-9870-4cf38074acf0/model.onnx
>> File URL: https://models.neuralmagic.com/cv-classification/...
>> Parent directory: .../.cache/sparsezoo/eb977dae-2454-471b-9870-4cf38074acf0
```

```python
print(model.recipes)

>> Directory(name=recipe)

print(f"File name: {model.recipes.name}\n"
      f"Contains: {[file.name for file in model.recipes.files]}\n"
      f"File path: {model.recipes.path}\n"
      f"File URL: {model.recipes.url}\n"
      f"Parent directory: {model.recipes.parent_directory}")
      
>> File name: recipe
>> Contains: ['recipe_original.md', 'recipe_transfer-classification.md']
>> File path: /home/user/.cache/sparsezoo/eb977dae-2454-471b-9870-4cf38074acf0/recipe
>> File URL: None
>> Parent directory: /home/user/.cache/sparsezoo/eb977dae-2454-471b-9870-4cf38074acf0
```

### Selecting Checkpoint-Specific Data

A SparseZoo model may contain several checkpoints. The model may contain a checkpoint that had been saved before the model was quantized - that checkpoint would be used for transfer learning. Another checkpoint might have been saved after the quantization step - that one is usually directly used for inference.

The recipes may also vary depending on the use case. We may want to access a recipe that was used to sparsify the dense model (`recipe_original`) or the one that enables us to sparse transfer learn from the already sparsified model (`recipe_transfer`). 

There are two ways to access those specific files.

#### Accessing Recipes (Through Python API)
```python
available_recipes = model.recipes.available
print(available_recipes)

>> ['original', 'transfer-classification']

transfer_recipe = model.recipes["transfer-classification"]
print(transfer_recipe)

>> File(name=recipe_transfer-classification.md)

original_recipe = model.recipes.default # recipe defaults to `original`
original_recipe_path = original_recipe.path # downloads the recipe and returns its path
print(original_recipe_path)

>> .../.cache/sparsezoo/eb977dae-2454-471b-9870-4cf38074acf0/recipe/recipe_original.md
```

#### Accessing Checkpoints (Through Python API)
In general, we are expecting the following checkpoints to be included in the model: 

- `checkpoint_prepruning`
- `checkpoint_postpruning`
- `checkpoint_preqat`
- `checkpoint_postqat` 

The checkpoint that the model defaults to is the `preqat` state (just before the quantization step).

```python
from sparsezoo import Model

stub = "zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/pruned_quant_3layers-aggressive_84"

model = Model(stub)
available_checkpoints = model.training.available
print(available_checkpoints)

>> ['preqat']

preqat_checkpoint = model.training.default # recipe defaults to `preqat`
preqat_checkpoint_path = preqat_checkpoint.path # downloads the checkpoint and returns its path
print(preqat_checkpoint_path)

>> .../.cache/sparsezoo/0857c6f2-13c1-43c9-8db8-8f89a548dccd/training

[print(file.name) for file in preqat_checkpoint.files]

>> vocab.txt
>> special_tokens_map.json
>> pytorch_model.bin
>> config.json
>> training_args.bin
>> tokenizer_config.json
>> trainer_state.json
>> tokenizer.json
```


#### Accessing Recipes (Through Stub String Arguments)

You can also directly request a specific recipe/checkpoint type by appending the appropriate URL query arguments to the stub:
```python
from sparsezoo import Model

stub = "zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/pruned95_quant-none?recipe=transfer"

model = Model(stub)

# Inspect which files are present.
# Note that the available recipes are restricted
# according to the specified URL query arguments
print(model.recipes.available)

>> ['transfer-classification']

transfer_recipe = model.recipes.default # Now the recipes default to the one selected by the stub string arguments
print(transfer_recipe)

>> File(name=recipe_transfer-classification.md)
```

### Accessing Sample Data

The user may easily request a sample batch of data that represents the inputs and outputs of the model.

```python
sample_data = model.sample_batch(batch_size = 10)

print(sample_data['sample_inputs'][0].shape)
>> (10, 3, 224, 224) # (batch_size, num_channels, image_dim, image_dim)

print(sample_data['sample_outputs'][0].shape)
>> (10, 1000) # (batch_size, num_classes)
```

### Model Search
The function `search_models` enables the user to quickly filter the contents of SparseZoo repository to find the stubs of interest:

```python
from sparsezoo import search_models

args = {
    "domain": "cv",
    "sub_domain": "segmentation",
    "architecture": "yolact",
}

models = search_models(**args)
[print(model) for model in models]

>> Model(stub=zoo:cv/segmentation/yolact-darknet53/pytorch/dbolya/coco/pruned82_quant-none)
>> Model(stub=zoo:cv/segmentation/yolact-darknet53/pytorch/dbolya/coco/pruned90-none)
>> Model(stub=zoo:cv/segmentation/yolact-darknet53/pytorch/dbolya/coco/base-none)
```

### Environmental Variables

Users can specify the directory where models (temporarily during download) and its required credentials will be saved in your working machine. 
`SPARSEZOO_MODELS_PATH` is the path where the downloaded models will be saved temporarily. Default `~/.cache/sparsezoo/`
`SPARSEZOO_CREDENTIALS_PATH` is the path where `credentials.yaml` will be saved. Default `~/.cache/sparsezoo/`

### Console Scripts

In addition to the Python APIs, a console script entry point is installed with the package `sparsezoo`.
This enables easy interaction straight from your console/terminal.

#### Downloading

Download command help

```shell script
sparsezoo.download -h
```

<br>Download ResNet-50 Model

```shell script
sparsezoo.download zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/base-none
```

<br>Download pruned and quantized ResNet-50 Model

```shell script
sparsezoo.download zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/pruned_quant-moderate
```

#### Searching

Search command help

```shell script
sparsezoo search -h
```

<br>Searching for all classification MobileNetV1 models in the computer vision domain

```shell script
sparsezoo search --domain cv --sub-domain classification --architecture mobilenet_v1
```

<br>Searching for all ResNet-50 models

```shell script
sparsezoo search --domain cv --sub-domain classification \
    --architecture resnet_v1 --sub-architecture 50
```

For a more in-depth read, check out [SparseZoo documentation.](https://docs.neuralmagic.com/sparsezoo/)

## Resources

### Learning More

- Documentation: [SparseML,](https://docs.neuralmagic.com/sparseml/) [SparseZoo,](https://docs.neuralmagic.com/sparsezoo/) [Sparsify,](https://docs.neuralmagic.com/sparsify/) [DeepSparse](https://docs.neuralmagic.com/deepsparse/)
- Neural Magic: [Blog,](https://www.neuralmagic.com/blog/) [Resources](https://www.neuralmagic.com/resources/)

### Release History

Official builds are hosted on PyPI

- stable: [sparsezoo](https://pypi.org/project/sparsezoo/)
- nightly (dev): [sparsezoo-nightly](https://pypi.org/project/sparsezoo-nightly/)

Additionally, more information can be found via [GitHub Releases.](https://github.com/neuralmagic/sparsezoo/releases)

### License

The project is licensed under the [Apache License Version 2.0.](https://github.com/neuralmagic/sparsezoo/blob/main/LICENSE)

## Community

### Contribute

We appreciate contributions to the code, examples, integrations, and documentation as well as bug reports and feature requests! [Learn how here.](https://github.com/neuralmagic/sparsezoo/blob/main/CONTRIBUTING.md)

### Join

For user help or questions about SparseZoo, sign up or log in to our [**Neural Magic Community Slack**](https://neuralmagic.com/community/). We are growing the community member by member and happy to see you there. Bugs, feature requests, or additional questions can also be posted to our [GitHub Issue Queue.](https://github.com/neuralmagic/sparsezoo/issues)

You can get the latest news, webinar and event invites, research papers, and other ML Performance tidbits by [subscribing](https://neuralmagic.com/subscribe/) to the Neural Magic community.

For more general questions about Neural Magic, please fill out this [form.](http://neuralmagic.com/contact/)
