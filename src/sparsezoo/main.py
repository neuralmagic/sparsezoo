# Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Script to download a model from sparse zoo


##########
Command help:
usage: main.py [-h] {download,search} ...

Download objects from the SparseZoo

positional arguments:
  {download,search}

optional arguments:
  -h, --help         show this help message and exit


##########
usage: main.py search [-h] --domain DOMAIN --sub-domain SUB_DOMAIN
                                [--architecture ARCHITECTURE]
                                [--sub-architecture SUB_ARCHITECTURE]
                                [--framework FRAMEWORK] [--repo REPO]
                                [--dataset DATASET]
                                [--training-scheme TRAINING_SCHEME]
                                [--sparse-name OPTIM_NAME]
                                [--sparse-category OPTIM_CATEGORY]
                                [--sparse-target OPTIM_TARGET]
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
  --sparse-name OPTIM_NAME
                        The name describing the sparsification of the model the
                        object belongs to, e.g. base, pruned, pruned_quant
  --sparse-category OPTIM_CATEGORY
                        The degree of sparsification of the model the object
                        belongs to; e.g. none, conservative (~100 baseline
                        metric), moderate (>=99 baseline metric), aggressive
                        (<99 baseline metric)
  --sparse-target OPTIM_TARGET
                        The deployment target of sparsification of the model the
                        object belongs to; e.g. edge, deepsparse,
                        deepsparse_throughput, gpu
  --release-version RELEASE_VERSION
                        the max release version of the model in semantic
                        version format
  --page PAGE           The page of search results to view
  --page-length PAGE_LENGTH
                        The amount of search results per page to view


##########
usage: main.py download [-h] --domain DOMAIN --sub-domain SUB_DOMAIN
                                  --architecture ARCHITECTURE
                                  [--sub-architecture SUB_ARCHITECTURE]
                                  --framework FRAMEWORK --repo REPO --dataset
                                  DATASET [--training-scheme TRAINING_SCHEME]
                                  --sparse-name OPTIM_NAME --sparse-category
                                  OPTIM_CATEGORY [--sparse-target OPTIM_TARGET]
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
  --sparse-name OPTIM_NAME
                        The name describing the sparsification of the model the
                        object belongs to, e.g. base, pruned, pruned_quant
  --sparse-category OPTIM_CATEGORY
                        The degree of sparsification of the model the object
                        belongs to; e.g. none, conservative (~100 baseline
                        metric), moderate (>=99 baseline metric), aggressive
                        (<99 baseline metric)
  --sparse-target OPTIM_TARGET
                        The deployment target of sparsification of the model the
                        object belongs to; e.g. edge, deepsparse,
                        deepsparse_throughput, gpu
  --release-version RELEASE_VERSION
                        the max release version of the model in semantic
                        version format
  --save-dir SAVE_DIR   The directory to save the model files in, defaults to
                        the cache directory of the sparsezoo


##########
Example search:
python3 scripts/main.py search --domain cv --sub-domain classification


##########
Example search for MobileNetV1:
python3 scripts/main.py search --domain cv --sub-domain classification \
    --architecture mobilenet_v1


#########
Example download MobileNetV1:
sparsezoo download --domain cv --sub-domain classification --architecture mobilenet_v1 \
    --sub-architecture 1.0 --framework pytorch --repo sparseml --dataset imagenet \
    --sparse-name base --sparse-category none

#########
Example download MobileNetV1 with maximum release version:
sparsezoo download --domain cv --sub-domain classification --architecture mobilenet_v1 \
    --sub-architecture 1.0 --framework pytorch --repo sparseml --dataset imagenet \
    --sparse-name base --sparse-category none --release-version 0.1.0


"""
import argparse
import logging

from sparsezoo import Model, model_args_to_stub, search_models


__all__ = ["main"]

DOWNLOAD_COMMAND = "download"
SEARCH_COMMAND = "search"

LOGGER = logging.getLogger()


def add_model_arguments(parser, download_required=False):
    parser.add_argument(
        "--domain",
        type=str,
        required=True,
        help="The domain of the model the object belongs to; e.g. cv, nlp",
    )
    parser.add_argument(
        "--sub-domain",
        type=str,
        required=True,
        help="The sub domain of the model the object belongs to; "
        "e.g. classification, segmentation",
    )
    parser.add_argument(
        "--architecture",
        type=str,
        required=download_required,
        help="The architecture of the model the object belongs to; "
        "e.g. resnet_v1, mobilenet_v1",
    )
    parser.add_argument(
        "--sub-architecture",
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
        "--sparse-name",
        type=str,
        required=download_required,
        help="The name describing the sparsification of the model "
        "the object belongs to, e.g. base, pruned, pruned_quant",
    )
    parser.add_argument(
        "--sparse-category",
        type=str,
        required=download_required,
        help="The degree of sparsification of the model the object "
        "belongs to; e.g. none, conservative (~100 baseline metric), "
        "moderate (>=99 baseline metric), aggressive (<99 baseline metric)",
    )
    parser.add_argument(
        "--sparse-target",
        type=str,
        default=None,
        help="The deployment target of sparsification of the model "
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
        DOWNLOAD_COMMAND,
        description="Download a specific model from the repo.",
    )
    search_parser = subparsers.add_parser(
        SEARCH_COMMAND,
        description="Search for objects from the repo.",
    )
    add_model_arguments(download_parser, download_required=True)
    add_model_arguments(search_parser)

    download_parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="The directory to save the model files in, "
        "defaults to the cache directory of the sparsezoo",
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


def _get_command_from_model(model: Model):
    download_command = "sparsezoo download "

    fields = [
        "domain",
        "sub_domain",
        "architecture",
        "sub_architecture",
        "framework",
        "repo",
        "dataset",
        "training_scheme",
        "sparse_name",
        "sparse_category",
        "sparse_target",
    ]

    command_strings = [
        "--{} {}".format(field.replace("_", "-"), getattr(model, field))
        for field in fields
        if hasattr(model, field) and getattr(model, field) is not None
    ]

    command_string = download_command + " ".join(command_strings)
    return command_string


def search(args):
    LOGGER.info("loading available models...")
    models = search_models(
        domain=args.domain,
        sub_domain=args.sub_domain,
        architecture=args.architecture,
        sub_architecture=args.sub_architecture,
        framework=args.framework,
        repo=args.repo,
        dataset=args.dataset,
        training_scheme=args.training_scheme,
        sparse_name=args.sparse_name,
        sparse_category=args.sparse_category,
        sparse_target=args.sparse_target,
        release_version=args.release_version,
        page=args.page,
        page_length=args.page_length,
    )

    print("Search results")
    print("====================")
    result_start = (args.page - 1) * args.page_length + 1
    result_end = (args.page) * args.page_length
    print(f"Showing results {result_start} - {result_end}")
    print("")

    for index, model in enumerate(models):
        result_index = (index + 1) + (args.page_length * (args.page - 1))
        header = f"{result_index}) {str(model)}"
        print(header)
        print("-------------------------")


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    if args.command == DOWNLOAD_COMMAND:
        LOGGER.info("Downloading files from model...")
        stub = model_args_to_stub(
            domain=args.domain,
            sub_domain=args.sub_domain,
            architecture=args.architecture,
            sub_architecture=args.sub_architecture,
            framework=args.framework,
            repo=args.repo,
            dataset=args.dataset,
            training_scheme=args.training_scheme,
            sparse_name=args.sparse_name,
            sparse_category=args.sparse_category,
            sparse_target=args.sparse_target,
            release_version=args.release_version,
        )

        if args.save_dir:
            model = Model(stub, download_path=args.save_dir)
        else:
            model = Model(stub)

        model.download()

        print("Download results")
        print("====================")
        print("")
        print(f"downloaded to {model.path}")
    elif args.command == SEARCH_COMMAND:
        search(args)


if __name__ == "__main__":
    main()
