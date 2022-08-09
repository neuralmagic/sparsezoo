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


from typing import Dict, List, Optional

from pydantic import BaseModel, validator


class ModelCommands(BaseModel):
    deploy: List[Dict]
    train: List[Dict]
    benchmark: List[Dict]


class ModelCardValidation(BaseModel):
    """
    Pydantic model to validate the model_card
    """

    card_version: str
    base: str
    parent: Optional[str] = None
    domain: str
    sub_domain: Optional[str]
    task: str = ""
    architecture: str
    framework: Optional[str] = None
    repo: Optional[str] = None
    dataset: Optional[str] = None
    source_dataset: str
    train_dataset: Optional[str] = None
    optimizations: str
    display_name: str
    tags: List[str]
    commands: Optional[ModelCommands] = None

    @validator("source_dataset", always=True, pre=True)
    def validate_source_dataset(cls, value, values):
        if value:
            return value
        if "source_dataset" in values and values["dataset"]:
            print("Field name 'dataset' is outdated. Please change to 'source_dataset'")
            return values["dataset"]

        raise ValueError("Please add dataset in the model card")

    @validator("train_dataset", always=True)
    def validate_train_dataset(cls, value, values):
        if value:
            return value
        if "source_dataset" in values and values["source_dataset"]:
            print(
                "train_dataset set to source_dataset. If not desired, "
                "please add field 'train_dataset' in model.md"
            )
            return values["source_dataset"]

        raise ValueError("Please add dataset in the model card")

    @validator("task", always=True)
    def validate_task(cls, value, values):
        if value:
            return value
        if "sub_domain" in values and values["sub_domain"]:
            print("Field name 'sub_domain' is outdated. Please change to 'task'")
            return values["sub_domain"]

        raise ValueError("Please add task in the model card")
