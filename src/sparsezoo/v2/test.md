# API User pathway

## Download one file

### Through the string arguments
```python
from sparsezoo import Model

stub = "zoo:cv/classification/mobilenet_v1-1.0/pytorch/sparseml/imagenet/pruned-moderate"
stub = stub + "?" + "recipe=transfer_learn"
# will read all the files from the server, except all the recipe files. The only recipe 
# file that will be "seen" by the `model` is "transfer learn"
model = Model(path=stub)
model.download() # optional: may be called to download all the files, but is not required (see below). Will be omitted in further examples
recipes: str = model.recipes.downloaded_path() # will automatically download the file (if not already downloaded) and return the path
...
```
```python
from sparsezoo import Model

stub = "zoo:cv/classification/mobilenet_v1-1.0/pytorch/sparseml/imagenet/pruned-moderate"
stub = stub + "?" + "checkpoint=preqat"
# will read all the files from the server, except for the full training folder. Training folder
# will be identical to prequat checkpoint directory
model = Model(path=stub)
training: str = model.training.downloaded_path() 
...
```
### Through the API purely
```python
from sparsezoo import Model

stub = "zoo:cv/classification/mobilenet_v1-1.0/pytorch/sparseml/imagenet/pruned-moderate"
model = Model(path=stub)
# ALTERNATIVE ONE: filtering on download
model.recipes.download(recipe="transfer_learn")
recipe: str = model.recipes.downloaded_path()
# ALTERNATIVE TWO: filtering on .downloaded_path()
recipe: str = model.recipes.downloaded_path(recipe="transfer_learn")
```

```python
from sparsezoo import Model

stub = "zoo:cv/classification/mobilenet_v1-1.0/pytorch/sparseml/imagenet/pruned-moderate"
model = Model(path=stub)
# ALTERNATIVE ONE: filtering on download
model.training.download(checkpoint="preqat")
training: str = model.training.downloaded_path()
# ALTERNATIVE TWO: filtering on .downloaded_path()
training: str = model.training.downloaded_path(checkpoint="preqat")
```

## Download multiple chosen files (also in parallel)

### Through the string arguments
```python
from sparsezoo import Model

stub = "zoo:cv/classification/mobilenet_v1-1.0/pytorch/sparseml/imagenet/pruned-moderate"
# choose the files through the string arguments
stub = stub + "?" + "recipe=transfer_learn" + "&" + "checkpoint=preqat" + "&" + "recipe=original"
model = Model(path=stub)
recipes: List[str] = model.recipes.downloaded_path() 
training: str = model.training.downloaded_path() # training will be a Directory with a preqat checkpoint
...
```
### Through the API purely
```python
from sparsezoo import Model

stub = "zoo:cv/classification/mobilenet_v1-1.0/pytorch/sparseml/imagenet/pruned-moderate"
model = Model(path=stub)
# explicitly define which files to download
model.recipes.download()
model.training.download(checkpoint="preqat")

recipes: List[str] = model.recipes.downloaded_path() # downloaded_path() may be either of type: str or List[str]
training: str = model.training.downloaded_path() 
```

## Download all files, no "filtering"
```python
from sparsezoo import Model

stub = "zoo:cv/classification/mobilenet_v1-1.0/pytorch/sparseml/imagenet/pruned-moderate"
model = Model(path=stub)
model.download()
training: str = model.training.downloaded_path() 
deployment: str = model.deployment.downloaded_path()
model_card: str = model.model_card.downloaded_path()
...
```
## QnA
Q: What happens if user requests a file/directory by the invalid string param 
A: Error on Model class construction, that says that the param is invalid and proposes the valid params in the error message

Q: What happens if the user calls `.download()` on the file that fails to get downloaded
A: Verbose error, that either explains why 
- a file cannot be downloaded (e.g. if the user calls `.download()` on the file that we do not have data on from the API)
- a file that should be downloaded does not get downloaded (e.g. if timeout on waiting from the response from the server)
--> ideally, we would like the user to avoid this pathway and directly use `.downloaded_path()`

Q: What happens if the user calls `.downloaded_path()` on a file that does not have one
A: The file should realize the it has not been downloaded yet, download itself and then return the correct path automatically

## Questions from my side:
1. Should the user be able to download and access any file? I am talking about "private" ones like `benchmark.yaml`, `eval.yaml`,
`sample_outputs` or  `logs`. I would be mostly worried about the heavy ones like `onnx folder`.
2. Should the user be able to download each and any files using string parameters? Or just `recipe`, `checkpoint` and `deployment`.
3. Did we agree how do we hash the folder names in `~/../.cache/sparsezoo` - so that we are more human readable?
