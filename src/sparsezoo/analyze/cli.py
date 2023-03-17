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
File containing click CLI options and helpers for analyze api
"""

import functools

import click


CONTEXT_SETTINGS = dict(
    token_normalize_func=lambda x: x.replace("-", "_"),
    show_default=True,
    ignore_unknown_options=True,
    allow_extra_args=True,
)


def analyze_options(command: callable):
    """
    A decorator that takes in a click command and adds analyze api options
    to it, this method is meant to be a single source of truth across all
    analyze api(s). This decorator can be directly imported and used on
    top of another click command.

    :param command: A click callable command
    :return: The same click callable but with analyze api options attached
    """

    @click.argument(
        "model_path",
        type=str,
        required=True,
    )
    @click.option(
        "--save",
        default=None,
        type=click.Path(
            file_okay=True, dir_okay=False, readable=True, resolve_path=True
        ),
        help="Path to a yaml file to write results to, note: file will be "
        "overwritten if exists",
    )
    @functools.wraps(command)
    def wrap_common_options(*args, **kwargs):
        """
        Wrapper that adds analyze options to command
        """
        return command(*args, **kwargs)

    return wrap_common_options


@click.command()
@analyze_options
def main():
    pass


if __name__ == "__main__":
    main()
