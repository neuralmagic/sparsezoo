"""
Script to download a model from sparse zoo


##########
Command help:
usage: model_download.py [-h] {search,download} ...

Download deployed models

positional arguments:
  {search,download}

optional arguments:
  -h, --help       show this help message and exit


##########
usage: model_download.py search [-h] --dom DOM --sub-dom SUB_DOM [--arch ARCH]
                                [--sub-arch SUB_ARCH] [--dataset DATASET]
                                [--framework FRAMEWORK]
                                [--repo-source REPO_SOURCE]
                                [--optimization-name OPTIMIZATION_NAME]
                                [--release-version RELEASE_VERSION]
                                [--page PAGE] [--page-length PAGE_LENGTH]

Search for models from the repo.

optional arguments:
  -h, --help            show this help message and exit
  --dom DOM             the domain the model belongs to; ex: cv, nlp, etc
  --sub-dom SUB_DOM     the sub domain the model belongs to; ex:
                        classification, detection, etc
  --arch ARCH           the architecture the model belongs to; ex: resnet-v1,
                        mobilenet-v1, etc
  --sub-arch SUB_ARCH   the sub architecture the model belongs to; ex: 50,
                        101, etc
  --dataset DATASET     the dataset used to train the model; ex: imagenet,
                        cifar, etc
  --framework FRAMEWORK
                        the framework used to train the model; ex: tensorflow,
                        pytorch, keras, onnx, etc
  --repo-source REPO_SOURCE
                        the source of the model; ex: torchvision
  --optimization-name OPTIMIZATION_NAME
                        the optimization name of the model; ex: base, recal,
                        recal-perf
  --release-version RELEASE_VERSION
                        the max release version of the model in semantic
                        version format
  --page PAGE           The page of search results to view
  --page-length PAGE_LENGTH
                        The amount of search results per page to view


##########
usage: model_download.py download [-h] --dom DOM --sub-dom SUB_DOM --arch ARCH
                                  [--sub-arch SUB_ARCH] --dataset DATASET
                                  --framework FRAMEWORK --repo-source
                                  REPO_SOURCE --optimization-name
                                  OPTIMIZATION_NAME
                                  [--release-version RELEASE_VERSION]
                                  [--save-dir SAVE_DIR]

Download a specific model from the repo.

optional arguments:
  -h, --help            show this help message and exit
  --dom DOM             the domain the model belongs to; ex: cv, nlp, etc
  --sub-dom SUB_DOM     the sub domain the model belongs to; ex:
                        classification, detection, etc
  --arch ARCH           the architecture the model belongs to; ex: resnet-v1,
                        mobilenet-v1, etc
  --sub-arch SUB_ARCH   the sub architecture the model belongs to; ex: 50,
                        101, etc
  --dataset DATASET     the dataset used to train the model; ex: imagenet,
                        cifar, etc
  --framework FRAMEWORK
                        the framework used to train the model; ex: tensorflow,
                        pytorch, keras, onnx, etc
  --repo-source REPO_SOURCE
                        the source of the model; ex: torchvision
  --optimization-name OPTIMIZATION_NAME
                        the optimization name of the model; ex: base, recal,
                        recal-perf
  --release-version RELEASE_VERSION
                        the max release version of the model in semantic
                        version format
  --save-dir SAVE_DIR   The directory to save the model files in, defaults to
                        the cwd with the model description as a sub folder


##########
Example search:
python3 scripts/model_download.py search --dom cv --sub-dom classification


##########
Example search for mobilenet v1:
python3 scripts/model_download.py search --dom cv --sub-dom classification \
    --arch mobilenet_v1


#########
Example download mobilenet v1:
python3 scripts/model_download.py download --dom cv --sub-dom classification \
    --arch mobilenet_v1 --sub-arch 1.0 --dataset imagenet --framework pytorch \
    --optimization-name base --repo-source torchvision


#########
Example download mobilenet v1 with maximum release version:
python3 scripts/model_download.py download --dom cv --sub-dom classification \
    --arch mobilenet_v1 --sub-arch 1.0 --dataset imagenet --framework pytorch \
    --optimization-name base --repo-source torchvision --release-version 1.5


"""
import argparse
import logging
from pprint import pprint

from sparsezoo.api import download_model, search_models


DOWNLOAD_COMMAND = "download"
SEARCH_COMMAND = "search"

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger()


def add_model_arguments(parser, download_required=False):
    parser.add_argument(
        "--dom",
        type=str,
        required=True,
        help="the domain the model belongs to; ex: cv, nlp, etc",
    )
    parser.add_argument(
        "--sub-dom",
        type=str,
        required=True,
        help="the sub domain the model belongs to; "
        "ex: classification, detection, etc",
    )
    parser.add_argument(
        "--arch",
        type=str,
        required=download_required,
        help="the architecture the model belongs to; ex: resnet-v1, mobilenet-v1, etc",
    )
    parser.add_argument(
        "--sub-arch",
        type=str,
        default="none" if download_required else None,
        help="the sub architecture the model belongs to; ex: 50, 101, etc",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=download_required,
        help="the dataset used to train the model; ex: imagenet, cifar, etc",
    )
    parser.add_argument(
        "--framework",
        type=str,
        required=download_required,
        help="the framework used to train the model; "
        "ex: tensorflow, pytorch, keras, onnx, etc",
    )
    parser.add_argument(
        "--repo-source",
        type=str,
        required=download_required,
        help="the source of the model; ex: torchvision",
    )
    parser.add_argument(
        "--optimization-name",
        type=str,
        required=download_required,
        help="the optimization name of the model; ex: base, recal, recal-perf",
    )
    parser.add_argument(
        "--release-version",
        type=str,
        help="the max release version of the model in semantic version format",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download models from the sparse zoo repo"
    )

    subparsers = parser.add_subparsers(dest="command")
    download_parser = subparsers.add_parser(
        DOWNLOAD_COMMAND,
        description="Download a specific model from the repo.",
    )
    search_parser = subparsers.add_parser(
        SEARCH_COMMAND,
        description="Search for models from the repo.",
    )
    add_model_arguments(download_parser, download_required=True)
    add_model_arguments(search_parser)

    download_parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="The directory to save the model files in, "
        "defaults to the cwd with the model description as a sub folder",
    )

    search_parser.add_argument(
        "--page", type=int, default=1, help="The page of search results to view"
    )

    search_parser.add_argument(
        "--page-length",
        type=int,
        default=20,
        help="The amount of search results per page to view",
    )

    return parser.parse_args()


def main(args):
    logging.basicConfig(level=logging.INFO)

    if args.command == DOWNLOAD_COMMAND:
        LOGGER.info("Downloading files from model...")
        download_model(
            args.dom,
            args.sub_dom,
            args.arch,
            args.sub_arch,
            args.dataset,
            args.framework,
            args.repo_source,
            args.optimization_name,
            release_version=args.release_version,
            save_dir=args.save_dir,
        )
    elif args.command == SEARCH_COMMAND:
        LOGGER.info("loading available models...")
        models = search_models(
            args.dom,
            args.sub_dom,
            architecture=args.arch,
            sub_architecture=args.sub_arch,
            dataset=args.dataset,
            framework=args.framework,
            repo_source=args.repo_source,
            optimization_name=args.optimization_name,
            release_version=args.release_version,
            page=args.page,
            page_length=args.page_length,
        )
        pprint([model.dict() for model in models])


if __name__ == "__main__":
    main(parse_args())
