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

import argparse
import glob
import os
import subprocess
import sys
from typing import List, NamedTuple


def parse_args():
    """
    Setup and parse command line arguments for using the script
    """
    parser = argparse.ArgumentParser(
        description="Create and package documentation for the repository"
    )
    parser.add_argument(
        "--src",
        type=str,
        required=True,
        help="the source directory to read the source for the docs from",
    )
    parser.add_argument(
        "--dest",
        type=str,
        required=True,
        help="the destination directory to put the built docs",
    )

    return parser.parse_args()


def create_docs(src: str, dest: str):
    print("running sphinx-multiversion")
    res = subprocess.run(["sphinx-multiversion", src, dest])

    if not res.returncode == 0:
        raise Exception(f"{res.stdout} {res.stderr}")

    print("completed sphinx build")


def package_docs(dest: str):
    print("packaging docs")
    version_folders = os.listdir(dest)
    print(version_folders)


def main():
    args = parse_args()
    create_docs(args.src, args.dest)
    package_docs(args.dest)


if __name__ == "__main__":
    main()
