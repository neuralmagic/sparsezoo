# Sparsezoo developer guide

## Quickstart
### Creating a model class object

`Model` is a fundamental object, that serves as an interface with the `sparsezoo` library. It represents a sparsezoo model, together with all its directories and files.

```python
from sparsezoo import Model

stub = "zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/pruned95_quant-none"

# create a model from a stub
model = Model(stub)
print(str(model))

>> Model(stub=zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/pruned95_quant-none)
```

```python
from sparsezoo import Model

directory = "..."

# create a model from a local directory

model = Model(directory)
print(str(model))

>> Model(directory=/home/user/.cache/sparsezoo/eb977dae-2454-471b-9870-4cf38074acf0)
```
Unless specified otherwise, the model is saved to the local sparsezoo cache directory. This can be overridden by passing the additional download argument to the constructor:

```python
from sparsezoo import Model

stub = "zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/pruned95_quant-none"
download_directory = "./model_download_directory"

model = Model(stub, download_path = download_directory)
```

If the model is initialized from a stub, it may be downloaded either by using the `download()` method or by calling a property `path`. Both pathways are universal for all the files in sparsezoo. Calling the `path` property will always trigger file download unless the file has already been downloaded.

```python
# method 1
model.download() 

# method 2 (
model_path = model.path
```

We call the' available' method to inspect which files are present in the model. Then, we select a file by calling the appropriate attribute:

```python
model.available

>> {'training': Directory(name=training), 
>> 'deployment': Directory(name=deployment), 
>> 'sample_inputs': Directory(name=sample_inputs.tar.gz), 
>> 'sample_outputs': {'framework': Directory(name=sample_outputs.tar.gz)}, 
>> 'sample_labels': Directory(name=sample_labels.tar.gz), 
>> 'model_card': File(name=model.md), 
>> 'recipes': Directory(name=recipe), 
>> 'onnx_model': File(name=model.onnx)}
```

```python
model_card = model.model_card
print(model_card)

>> File(name=model.md)
```
```python
model_card_path = model.model_card.path
print(model_card_path)

>> /home/user/.cache/sparsezoo/eb977dae-2454-471b-9870-4cf38074acf0/model.md
```


## Model, Directory, and File

In general, every file in sparsezoo shares a set of attributes: `name`, `path`, `URL`, and `parent` directory. A directory is a unique type of file that contains other files. For that reason, it has an additional attribute `files`.

```python
print(model.onnx_model)

>> File(name=model.onnx)

print(f"File name: {model.onnx_model.name}\n"
      f"File path: {model.onnx_model.path}\n"
      f"File URL: {model.onnx_model.url}\n"
      f"Parent directory: {model.onnx_model.parent_directory}")
      
>> File name: model.onnx
>> File path: /home/user/.cache/sparsezoo/eb977dae-2454-471b-9870-4cf38074acf0/model.onnx
>> File URL: https://models.neuralmagic.com/cv-classification/...
>> Parent directory: /home/user/.cache/sparsezoo/eb977dae-2454-471b-9870-4cf38074acf0
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

## Selecting checkpoint-specific data

In `sparsezoo` a model may contain several checkpoints. An example would be one checkpoint that had been saved before the model was quantized -that checkpoint would be used for transfer learning. Another checkpoint might have been saved after the quantization. The recipes may also vary depending on the use case. We may want to access a recipe that was used to sparsify the dense model (`recipe_original`) or the one that enables us to sparse transfer learn from the already sparsified model (`recipe_transfer`). 

There are two ways to access those specific files.

### Through Python API
#### Accessing Recipes
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

>> /home/user/.cache/sparsezoo/eb977dae-2454-471b-9870-4cf38074acf0/recipe/recipe_original.md
```

#### Accessing Checkpoints
Generally expected checkpoints to be included in the model are the following (but not required): 

- checkpoint_prepruning
- checkpoint_postpruning
- checkpoint_preqat 
- checkpoint_postqat. 

The checkpoint that the model defaults to is the "preqat" state.

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

>> /home/damian/.cache/sparsezoo/0857c6f2-13c1-43c9-8db8-8f89a548dccd/training

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


### Through stub string arguments
```python
from sparsezoo import Model

stub = "zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/pruned95_quant-none?recipe=transfer"

model = Model(stub)

# Inspect which files are present.
# Note that the available recipes are restricted
# according to the specified string arguments
print(model.recipes.available)

>> ['transfer-classification']

transfer_recipe = model. recipes.default # Now the recipes default to the one selected by the stub string arguments
print(transfer_recipe)

>> File(name=recipe_transfer-classification.md)
```


## New interface for model search
`ModelArgs` class object is used to serve a similar purpose as a stub - containing information about the stub of interest. Even though it has been deprecated, the function `search_models` remained backward compatible and still consumes a similar type of data.


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











