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
from unittest.mock import patch

import pytest

from click.testing import CliRunner
from sparsezoo.deployment_package.cli import _csv_callback, _get_template, main


def _run_with_cli_runner(args: List[str]):
    runner = CliRunner()
    result = runner.invoke(main, args=args)
    return result


@pytest.mark.parametrize(
    "cli_args",
    [
        "--task blah",
        "--task ic --optimizing_metric blah",
        "--dataset blah",
        "--dataset mnli --task blah",
        "-d",
        "-m",
        "-t",
    ],
)
@patch("sparsezoo.deployment_package_module.cli.deployment_package")
def test_click_error_on_invalid_invocation(package_function, cli_args):
    result = _run_with_cli_runner(cli_args.split())
    assert result.exit_code >= 1


@pytest.mark.parametrize("cli_args", ["", "--optimizing-metric accuracy"])
@patch("sparsezoo.deployment_package_module.cli.deployment_package")
def test_value_error_when_dataset_and_task_not_provided(package_function, cli_args):
    with pytest.raises(ValueError):
        main.main(cli_args.split(), standalone_mode=False)


@pytest.mark.parametrize(
    "cli_args",
    [
        "--task ic",
        "--dataset imagenette",
        "--task ic --dataset imagenette"
        "deployment_directory --task ic --dataset imagenette --optimizing-metric "
        "accuracy",
        "--task ic --optimizing-metric accuracy,compression",
    ],
)
@patch("sparsezoo.deployment_package_module.cli.deployment_package")
@patch("sparsezoo.deployment_package_module.cli._download_deployment_directory")
def test_valid_invocation(mocked_download_func, package_function, cli_args):
    mocked_download_func.return_value = "."
    result = _run_with_cli_runner(cli_args.split())
    assert result.exit_code == 0


@pytest.mark.parametrize(
    "results",
    [
        {"stub": "zoo://", "metrics": {"accuracy": 99, "compression": 234567}},
    ],
)
@patch("sparsezoo.deployment_package_module.cli._download_deployment_directory")
def test_get_template_has_model_metrics(mocked_function, results):
    mocked_function.return_value = "."
    output = _get_template(results=results)
    assert isinstance(output, str)
    assert str(results.get("metrics")) in output


@pytest.mark.parametrize(
    "value, expected",
    [
        ("accuracy, compression", ["accuracy", "compression"]),
        ("accuracy,compression", ["accuracy", "compression"]),
        ("accuracy, c", ValueError()),
    ],
)
def test_csv_callback(value, expected):
    if isinstance(expected, ValueError):
        with pytest.raises(ValueError):
            _csv_callback(ctx=None, self=None, value=value)
    else:
        actual = _csv_callback(ctx=None, self=None, value=value)
        assert actual == expected
