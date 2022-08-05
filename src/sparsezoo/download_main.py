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
usage: sparsezoo.download [-h] [--save-dir SAVE_DIR] model_stub

Download specific models from the SparseZoo repo

positional arguments:
  model_stub           Path to a SparseZoo model stub i.e.
                       zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/pruned-moderate

optional arguments:
  -h, --help           show this help message and exit
  --save-dir SAVE_DIR  The directory to save the model files in, defaults to
                       the cwd with the model description as a sub folder

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

from sparsezoo import Model


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
        "defaults to the cache directory of the sparsezoo",
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

    if args.save_dir:
        model = Model(args.model_stub, download_path=args.save_dir)
    else:
        model = Model(args.model_stub)
    model.download()

    print("Download results")
    print("====================")
    print()
    print(f"{str(model)} downloaded to {model.path}")


if __name__ == "__main__":
    main()
