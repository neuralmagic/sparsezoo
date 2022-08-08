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


import json
from typing import List, Optional

import yaml

from pydantic import BaseModel, validator


class ModelCommands(BaseModel):
    deploy: str
    train: str
    benchmark: str


class ModelCardValidation(BaseModel):
    """
    Pydantic model to validate the model_card


    https://docs.google.com/document/d/1kyuZ8EhF8L6CEnpCQ_zhrT4cu-PHwPJX7fJXHcM0dL0/edit#

    card_version: 1.0
    base: (new repo stub of migrated base model)
    parent: (none) (optional)
    domain: (prev domain)
    task: (prev sub_domain)
    architecture: (prev architecture)
    sub_architecture: (prev sub_architecture)
    framework: (prev framework) (optional)
    repo: (prev repo) (optional)
    source_dataset: (prev dataset)
    train_dataset : (prev dataset for NLP) (optional - if not supplied match source_dataset)
    optimizations: (new, to be filled in later) (list strings)
    display_name: (prev display_name)
    tags: (prev tags) (list strings)
    commands: (optional)
            deploy:
                    [type]: [command]
            train:
                    [type]: [command]
            benchmark:
                    [type]: [command]
    """

    card_version: str
    # base: str # stub of the base model
    parent: Optional[str] = None  # stub of the parent model
    domain: str
    sub_domain: Optional[str]
    task: str = ""
    architecture: str
    framework: Optional[str] = None
    repo: Optional[str] = None
    dataset: Optional[str] = None
    source_dataset: str = ""
    # train_dataset: str
    # optimizations: str
    display_name: str
    tags: List[str]
    # commands: ModelCommands

    @validator("source_dataset", always=True)
    def validate_source_dataset(cls, value, values):
        if value:
            return value
        if "dataset" in values and values["dataset"]:
            print("Field name 'dataset' is outdated. Please change to 'source_dataset'")
            return values["dataset"]

        raise ValueError("Please add dataset in the model card")

    @validator("task", always=True)
    def validate_task(cls, value, values):
        if value:
            return value
        if "sub_domain" in values and values["sub_domain"]:
            print("Field name 'sub_domain' is outdated. Please change to 'task'")
            return values["sub_domain"]

        raise ValueError("Please add task in the model card")
