"""
Script to download a model from sparse zoo


##########
Command help:
usage: model_download.py [-h] {download,search} ...

Download objects from the sparse zoo repo

positional arguments:
  {download,search}

optional arguments:
  -h, --help         show this help message and exit


##########
usage: model_download.py search [-h] --dom DOM --sub-dom SUB_DOM [--arch ARCH]
                                [--sub-arch SUB_ARCH] [--framework FRAMEWORK]
                                [--repo REPO] [--dataset DATASET]
                                [--training_scheme TRAINING_SCHEME]
                                [--optim-name OPTIM_NAME]
                                [--optim-category OPTIM_CATEGORY]
                                [--optim-target OPTIM_TARGET]
                                [--release-version RELEASE_VERSION]
                                [--page PAGE] [--page-length PAGE_LENGTH]

Search for objects from the repo.

optional arguments:
  -h, --help            show this help message and exit
  --dom DOM             The domain of the model the object belongs to; e.g.
                        cv, nlp
  --sub-dom SUB_DOM     The sub domain of the model the object belongs to;
                        e.g. classification, segmentation
  --arch ARCH           The architecture of the model the object belongs to;
                        e.g. resnet_v1, mobilenet_v1
  --sub-arch SUB_ARCH   The sub architecture (scaling factor) of the model the
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
usage: model_download.py download [-h] --dom DOM --sub-dom SUB_DOM --arch ARCH
                                  [--sub-arch SUB_ARCH] --framework FRAMEWORK
                                  --repo REPO --dataset DATASET
                                  [--training_scheme TRAINING_SCHEME]
                                  --optim-name OPTIM_NAME --optim-category
                                  OPTIM_CATEGORY [--optim-target OPTIM_TARGET]
                                  [--release-version RELEASE_VERSION]
                                  [--save-dir SAVE_DIR]

Download a specific model from the repo.

optional arguments:
  -h, --help            show this help message and exit
  --dom DOM             The domain of the model the object belongs to; e.g.
                        cv, nlp
  --sub-dom SUB_DOM     The sub domain of the model the object belongs to;
                        e.g. classification, segmentation
  --arch ARCH           The architecture of the model the object belongs to;
                        e.g. resnet_v1, mobilenet_v1
  --sub-arch SUB_ARCH   The sub architecture (scaling factor) of the model the
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

from sparsezoo.objects import Model


DOWNLOAD_COMMAND = "download"
SEARCH_COMMAND = "search"

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger()


def add_model_arguments(parser, download_required=False):
    parser.add_argument(
        "--dom",
        type=str,
        required=True,
        help="The domain of the model the object belongs to; e.g. cv, nlp",
    )
    parser.add_argument(
        "--sub-dom",
        type=str,
        required=True,
        help="The sub domain of the model the object belongs to; "
        "e.g. classification, segmentation",
    )
    parser.add_argument(
        "--arch",
        type=str,
        required=download_required,
        help="The architecture of the model the object belongs to; "
        "e.g. resnet_v1, mobilenet_v1",
    )
    parser.add_argument(
        "--sub-arch",
        type=str,
        default=None,
        help="The sub architecture (scaling factor) of the model "
        "the object belongs to; e.g. 50, 101, 152",
    )
    parser.add_argument(
        "--framework",
        type=str,
        required=download_required,
        help="The framework the model the object belongs to was trained on; "
        "e.g. pytorch, tensorflow",
    )
    parser.add_argument(
        "--repo",
        type=str,
        required=download_required,
        help="The source repo for the model the object belongs to; "
        "e.g. sparseml, torchvision",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=download_required,
        help="The dataset the model the object belongs to was trained on; "
        "e.g. imagenet, cifar10",
    )
    parser.add_argument(
        "--training-scheme",
        type=str,
        default=None,
        help="The training scheme used on the model the object "
        "belongs to if any; e.g. augmented",
    )
    parser.add_argument(
        "--optim-name",
        type=str,
        required=download_required,
        help="The name describing the optimization of the model "
        "the object belongs to, e.g. base, pruned, pruned_quant",
    )
    parser.add_argument(
        "--optim-category",
        type=str,
        required=download_required,
        help="The degree of optimization of the model the object "
        "belongs to; e.g. none, conservative (~100 baseline metric), "
        "moderate (>=99 baseline metric), aggressive (<99 baseline metric)",
    )
    parser.add_argument(
        "--optim-target",
        type=str,
        default=None,
        help="The deployment target of optimization of the model "
        "the object belongs to; e.g. edge, deepsparse, deepsparse_throughput, gpu",
    )
    parser.add_argument(
        "--release-version",
        type=str,
        help="the max release version of the model in semantic version format",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download objects from the sparse zoo repo"
    )

    subparsers = parser.add_subparsers(dest="command")
    download_parser = subparsers.add_parser(
        DOWNLOAD_COMMAND, description="Download a specific model from the repo.",
    )
    search_parser = subparsers.add_parser(
        SEARCH_COMMAND, description="Search for objects from the repo.",
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
        model = Model.get_downloadable(
            domain=args.dom,
            sub_domain=args.sub_dom,
            architecture=args.arch,
            sub_architecture=args.sub_arch,
            framework=args.framework,
            repo=args.repo,
            dataset=args.dataset,
            training_scheme=args.training_scheme,
            optim_name=args.optim_name,
            optim_category=args.optim_category,
            optim_target=args.optim_target,
            release_version=args.release_version,
            override_parent_path=args.save_dir,
        )
        model.download()

        print("Download results")
        print("====================")
        print("")
        print(model.model_url_path)
        print(f"downloaded to {model.dir_path}")
    elif args.command == SEARCH_COMMAND:
        LOGGER.info("loading available objects...")
        models = Model.search_downloadable(
            domain=args.dom,
            sub_domain=args.sub_dom,
            architecture=args.arch,
            sub_architecture=args.sub_arch,
            framework=args.framework,
            repo=args.repo,
            dataset=args.dataset,
            training_scheme=args.training_scheme,
            optim_name=args.optim_name,
            optim_category=args.optim_category,
            optim_target=args.optim_target,
            release_version=args.release_version,
            page=args.page,
            page_length=args.page_length,
        )

        print("Search results")
        print("====================")

        for model in models:
            print("")
            print("Model:")
            print(f"    tag: {model.model_url_path}")
            sub_arch_command = (
                f"--sub-arch {model.sub_architecture}" if model.sub_architecture else ""
            )
            training_scheme_command = (
                f"--training-scheme {model.training_scheme}"
                if model.training_scheme
                else ""
            )
            optim_target_command = (
                f"--optim-target {model.training_scheme}"
                if model.training_scheme
                else ""
            )
            print(
                f"    command: python3 scripts/model_download.py download "
                f"--dom {model.domain} --sub-dom {model.sub_domain} "
                f"--arch {model.architecture} {sub_arch_command} "
                f"--framework {model.framework} --repo {model.repo} "
                f"--dataset {model.dataset} {training_scheme_command} "
                f"--optim-name {model.optim_name} "
                f"--optim-category {model.optim_category} {optim_target_command}"
            )


if __name__ == "__main__":
    main(parse_args())
