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

import tempfile

import pytest

from sparsezoo.v2.objects import Directory, File, Model


@pytest.mark.parametrize(
    "stub, args, should_pass",
    [
        #         (
        #             "zoo:cv/classification/mobilenet_v1-1.0/pytorch/sparseml/imagenet/pruned-moderate",  # noqa E501
        #             ("recipe", "transfer_learn"),
        #             True
        #
        #         ),
        # (
        #             "zoo:cv/classification/mobilenet_v1-1.0/pytorch/sparseml/imagenet/pruned-moderate",  # noqa E501
        #             ("recipe", "some_dummy_name"),
        #             False
        #
        #         ),
        (
            "zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/pruned_quant-aggressive_95",  # noqa E501
            ("deployment", ""),
            True,
        ),
        #
        # (
        #             "zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/pruned_quant-aggressive_95",  # noqa E501
        #             ("checkpoint", ""),
        #             True
        #
        #         ),
    ],
    scope="function",
)
def test_model_from_stub(stub, args, should_pass):
    temp_dir = tempfile.TemporaryDirectory(dir="/tmp")
    path = stub + "?" + args[0] + "=" + args[1]
    if should_pass:
        model = Model(path)
        model.download(directory_path=temp_dir.name)
        _assert_correct_files_downloaded(model, args)
    else:
        with pytest.raises(ValueError):
            model = Model(path)
            model.download(directory_path=temp_dir.name)


def _assert_correct_files_downloaded(model, args):
    for file_name, file in model._files_dictionary.items():
        if args[0] == "recipe" and file_name == "recipes":
            assert isinstance(file, File)
        elif args[0] == "deployment" and file_name == "deployment":
            assert isinstance(file, Directory)

        elif args[0] == "checkpoint" and file_name == "training":
            assert isinstance(file, Directory)

        else:
            assert file is None
