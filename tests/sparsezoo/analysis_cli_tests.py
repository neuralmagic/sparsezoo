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

from typing import List

import pytest

from click.testing import CliRunner
from sparsezoo.analyze_cli import main


def _run_with_cli_runner(args: List[str]):
    runner = CliRunner()
    result = runner.invoke(main, args=args)
    return result


@pytest.mark.parametrize(
    "cli_args",
    [
        "zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/pruned95-none",
        "zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet"
        "/pruned95_quant-none",
    ],
)
def test_valid_invocation(cli_args):
    result = _run_with_cli_runner(cli_args.split())
    assert result.exit_code == 0
