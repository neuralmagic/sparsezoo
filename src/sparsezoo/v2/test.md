# API User pathway

## Download one file

### Through the string arguments
```python
from sparsezoo import Model

stub = "zoo:cv/classification/mobilenet_v1-1.0/pytorch/sparseml/imagenet/pruned-moderate"
stub = stub + "?" + "recipe=transfer_learn"
model = Model(path=stub)
model.download() # will only download only recipe file (as indicated by the string argument). If no argument provided, .download() will download to cache
recipes: str = model.recipes.downloaded_path() 
...
```
```python
from sparsezoo import Model

stub = "zoo:cv/classification/mobilenet_v1-1.0/pytorch/sparseml/imagenet/pruned-moderate"
stub = stub + "?" + "deployment" # if we want the whole directory, the parameter does not have to have value
model = Model(path=stub)
model.download()
deployment: str = model.recipes.downloaded_path() 
...
```
### Through the API purely
```python
from sparsezoo import Model

stub = "zoo:cv/classification/mobilenet_v1-1.0/pytorch/sparseml/imagenet/pruned-moderate"
model = Model(path=stub)
recipe: str = model.recipes.download(recipe="transfer_learn")
```

```python
from sparsezoo import Model

stub = "zoo:cv/classification/mobilenet_v1-1.0/pytorch/sparseml/imagenet/pruned-moderate"
model = Model(path=stub)
recipe: str = model.deployment.download()
```

## Download multiple chosen files (also in parallel)

### Through the string arguments
```python
from sparsezoo import Model

stub = "zoo:cv/classification/mobilenet_v1-1.0/pytorch/sparseml/imagenet/pruned-moderate"
# choose the files through the string arguments
stub = stub + "?" + "recipe=transfer_learn" + "&" + "checkpoint=preqat" + "&" + "recipe=original"
model = Model(path=stub)
model.download() # will only download the specified files
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

Q: What happens if the user calls `.downloaded_path()` on a file that does not have one
A: Verbose error, that explains why one cannot call `.downloaded_path()` and shortly proposes a solution (if exists)

## Questions from my side:
1. Should the user be able to download and access any file? I am talking about "private" ones like `benchmark.yaml`, `eval.yaml`,
`sample_outputs` or  `logs`.
2. Should the user be able to download each and any files using string parameters? Or just `recipe`, `checkpoint` and `deployment`.

