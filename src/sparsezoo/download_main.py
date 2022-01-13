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
Script to download a model from SparseZoo

##########
Command help:
usage: sparsezoo.download [-h] [--save-dir SAVE_DIR] [--overwrite] model_stub

Download specific models from the SparseZoo repo

positional arguments:
  model_stub           Path to a SparseZoo model stub i.e.
                       zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/pruned-moderate

optional arguments:
  -h, --help           show this help message and exit
  --save-dir SAVE_DIR  The directory to save the model files in, defaults to
                       the cwd with the model description as a sub folder
  --overwrite          Overwrites existing model files if previously
                       downloaded

#########
Example download ResNet50:
sparsezoo.download \
    zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/base-none

#########
Example download SQuAD BERT base uncased
sparsezoo.download \
    zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/base-none


"""
import argparse
import logging

from sparsezoo.models import Zoo


__all__ = ["main"]

LOGGER = logging.getLogger()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download specific models from the SparseZoo repo"
    )

    parser.add_argument(
        "model_stub",
        type=str,
        help="Path to a SparseZoo model stub i.e. "
        "zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/pruned-moderate",
    )

    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="The directory to save the model files in, "
        "defaults to the cwd with the model description as a sub folder",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrites existing model files if previously downloaded",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    LOGGER.info("Downloading files from model '{}'".format(args.model_stub))

    if not isinstance(args.model_stub, str):
        raise ValueError("Model stub be a string")

    if not args.model_stub.startswith("zoo:"):
        raise ValueError("Model stub must start with 'zoo:'")

    model = Zoo.download_model_from_stub(
        stub=args.model_stub,
        override_parent_path=args.save_dir,
        overwrite=args.overwrite,
    )

    print("Download results")
    print("====================")
    print()
    print(f"{model.display_name} downloaded to {model.dir_path}")


if __name__ == "__main__":
    main()
