"""
Script to download a model from sparse zoo


##########
Command help:
usage: sparsezoo.py [-h] {download,search} ...

Download objects from the SparseZoo

positional arguments:
  {download,search}

optional arguments:
  -h, --help         show this help message and exit


##########
usage: sparsezoo.py search [-h] --domain DOMAIN --sub-domain SUB_DOMAIN
                                [--architecture ARCHITECTURE]
                                [--sub-architecture SUB_ARCHITECTURE]
                                [--framework FRAMEWORK] [--repo REPO]
                                [--dataset DATASET]
                                [--training-scheme TRAINING_SCHEME]
                                [--optim-name OPTIM_NAME]
                                [--optim-category OPTIM_CATEGORY]
                                [--optim-target OPTIM_TARGET]
                                [--release-version RELEASE_VERSION]
                                [--page PAGE] [--page-length PAGE_LENGTH]

Search for objects from the repo.

optional arguments:
  -h, --help            show this help message and exit
  --domain DOMAIN       The domain of the model the object belongs to; e.g.
                        cv, nlp
  --sub-domain SUB_DOMAIN
                        The sub domain of the model the object belongs to;
                        e.g. classification, segmentation
  --architecture ARCHITECTURE
                        The architecture of the model the object belongs to;
                        e.g. resnet_v1, mobilenet_v1
  --sub-architecture SUB_ARCHITECTURE
                        The sub architecture (scaling factor) of the model the
                        object belongs to; e.g. 50, 101, 152
  --framework FRAMEWORK
                        The framework the model the object belongs to was
                        trained on; e.g. pytorch, tensorflow
  --repo REPO           The source repo for the model the object belongs to;
                        e.g. sparseml, torchvision
  --dataset DATASET     The dataset the model the object belongs to was
                        trained on; e.g. imagenet, cifar10
  --training-scheme TRAINING_SCHEME
                        The training scheme used on the model the object
                        belongs to if any; e.g. augmented
  --optim-name OPTIM_NAME
                        The name describing the optimization of the model the
                        object belongs to, e.g. base, pruned, pruned_quant
  --optim-category OPTIM_CATEGORY
                        The degree of optimization of the model the object
                        belongs to; e.g. none, conservative (~100 baseline
                        metric), moderate (>=99 baseline metric), aggressive
                        (<99 baseline metric)
  --optim-target OPTIM_TARGET
                        The deployment target of optimization of the model the
                        object belongs to; e.g. edge, deepsparse,
                        deepsparse_throughput, gpu
  --release-version RELEASE_VERSION
                        the max release version of the model in semantic
                        version format
  --page PAGE           The page of search results to view
  --page-length PAGE_LENGTH
                        The amount of search results per page to view


##########
usage: sparsezoo.py download [-h] --domain DOMAIN --sub-domain SUB_DOMAIN
                                  --architecture ARCHITECTURE
                                  [--sub-architecture SUB_ARCHITECTURE]
                                  --framework FRAMEWORK --repo REPO --dataset
                                  DATASET [--training-scheme TRAINING_SCHEME]
                                  --optim-name OPTIM_NAME --optim-category
                                  OPTIM_CATEGORY [--optim-target OPTIM_TARGET]
                                  [--release-version RELEASE_VERSION]
                                  [--save-dir SAVE_DIR]

Download a specific model from the repo.

optional arguments:
  -h, --help            show this help message and exit
  --domain DOMAIN       The domain of the model the object belongs to; e.g.
                        cv, nlp
  --sub-domain SUB_DOMAIN
                        The sub domain of the model the object belongs to;
                        e.g. classification, segmentation
  --architecture ARCHITECTURE
                        The architecture of the model the object belongs to;
                        e.g. resnet_v1, mobilenet_v1
  --sub-architecture SUB_ARCHITECTURE
                        The sub architecture (scaling factor) of the model the
                        object belongs to; e.g. 50, 101, 152
  --framework FRAMEWORK
                        The framework the model the object belongs to was
                        trained on; e.g. pytorch, tensorflow
  --repo REPO           The source repo for the model the object belongs to;
                        e.g. sparseml, torchvision
  --dataset DATASET     The dataset the model the object belongs to was
                        trained on; e.g. imagenet, cifar10
  --training-scheme TRAINING_SCHEME
                        The training scheme used on the model the object
                        belongs to if any; e.g. augmented
  --optim-name OPTIM_NAME
                        The name describing the optimization of the model the
                        object belongs to, e.g. base, pruned, pruned_quant
  --optim-category OPTIM_CATEGORY
                        The degree of optimization of the model the object
                        belongs to; e.g. none, conservative (~100 baseline
                        metric), moderate (>=99 baseline metric), aggressive
                        (<99 baseline metric)
  --optim-target OPTIM_TARGET
                        The deployment target of optimization of the model the
                        object belongs to; e.g. edge, deepsparse,
                        deepsparse_throughput, gpu
  --release-version RELEASE_VERSION
                        the max release version of the model in semantic
                        version format
  --save-dir SAVE_DIR   The directory to save the model files in, defaults to
                        the cwd with the model description as a sub folder


##########
Example search:
python3 scripts/sparsezoo.py search --domain cv --sub-domain classification


##########
Example search for MobileNetV1:
python3 scripts/sparsezoo.py search --domain cv --sub-domain classification \
    --architecture mobilenet_v1


#########
Example download MobileNetV1:
sparsezoo download --domain cv --sub-domain classification --architecture mobilenet_v1 \
    --sub-architecture 1.0 --framework pytorch --repo torchvision --dataset imagenet \
    --optim-name base --optim-category none

#########
Example download MobileNetV1 with maximum release version:
sparsezoo download --domain cv --sub-domain classification --architecture mobilenet_v1 \
    --sub-architecture 1.0 --framework pytorch --repo torchvision --dataset imagenet \
    --optim-name base --optim-category none --release-version 0.1.0


"""
from sparsezoo.main import main


if __name__ == "__main__":
    main()
