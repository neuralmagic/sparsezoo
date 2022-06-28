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
Validator class that checks, whether the contents of ModelDirectory
class objects are valid
"""

import os
from typing import Callable, Dict, Optional, Set, Tuple, Union

from sparsezoo.v2.directory import Directory
from sparsezoo.v2.file import File
from sparsezoo.v2.integration_validation import validate_cv_classification, validate_nlp
from sparsezoo.v2.model_directory import ModelDirectory


__all__ = ["IntegrationValidator"]

# files/directories that are required for minimal validation
ESSENTIAL_FILES = {"deployment", "training", "model.onnx"}
INTEGRATION_NAME_TO_DATA = {"nlp": validate_nlp, "cv": validate_cv_classification}


class IntegrationValidator:
    """
    Helper class to facilitate the validation of the ModelDirectory class.
    It is called by the "validate" method of ModelDirectory.

    :param model_directory: ModelDirectory class object to be
        validated
    :param essential_files: a minimal set of file names in ModelDirectory,
        that need to be present, so that the validation is
        successful
    :param integration2data: mapping from the integration tag (string) from
        ModelDirectory's model card to validation function
    """

    def __init__(
        self,
        model_directory: ModelDirectory,
        essential_files: Set[str] = ESSENTIAL_FILES,
        integration2data: Dict[str, Callable] = INTEGRATION_NAME_TO_DATA,
    ):
        self.model_directory = model_directory
        self.essential_files = essential_files
        self.integration2data = integration2data
        self.minimal_validation = None

    def validate(self, minimal_validation: bool) -> bool:
        """
        Validates the structure and contents of ModelDirectory.

        1. Validate the general ModelDirectory structure (to see whether the
            directory contains either minimal or a full set of expected files)
            using self._validate_structure() method.

        2. Iterate over files in ModelDirectory and validate them. Some files may
            have additional validation methods implemented (e.g `training`
            directory).

        :param minimal_validation: boolean flag; if True, only the essential files
            in the ModelDirectory should be validated. Else, the ModelDirectory is
            expected to contain a full set of Directory/File class objects.
        :return: boolean flag; True if files are valid and no errors arise
        """
        validations = {}

        self.minimal_validation = minimal_validation

        # validate whether file names match
        if not self.validate_structure():
            mode = "minimal" if self.minimal_validation else "full"
            raise ValueError(
                "The attempt to validate ModelDirectory's structure "
                f"(against the {mode} expected set of files) failed."
            )
        # validate files one by one
        integration_name = self._get_integration_name(
            model_card=self.model_directory.model_card
        )
        (
            training_files,
            optional_training_files,
            additional_deployment_files,
        ) = self.integration2data[integration_name]()
        for file in self.model_directory.files:
            # checker for dict-type file
            if isinstance(file, dict):
                validations[file.__repr__()] = all(
                    _file.validate() for _file in file.values()
                )
            # checker for list-type file
            elif isinstance(file, list):
                validations[file.__repr__()] = all(_file.validate() for _file in file)
            else:
                # checker for File/Directory class objects
                if file.name == "training":
                    self._validate_training_directory(
                        training_directory=file,
                        files=training_files,
                        optional_files=optional_training_files,
                    )

                elif file.name == "deployment":
                    self._validate_deployment_directory(
                        deployment_directory=file,
                        files=training_files | additional_deployment_files,
                        optional_files=optional_training_files,
                    )

                validations[file.name] = file.validate()

        if all(validations.values()):
            return True
        else:
            raise ValueError(
                "Following files (or directories): "
                f"{[key for key, value in validations.items() if not value]} "
                "were not successfully validated."
            )
            return False

    def validate_structure(self) -> bool:
        """
        Iterate over all files in ModelDirectory and check
        whether the names of the files present
        match the (minimum) expected set of files

        :return boolean flag; True if ModelDirector structure
            is valid and no errors arise
        """
        if self.minimal_validation:
            # iterate over all files in ModelDirectory and
            # check whether they match the expected essential
            # set of files
            for file in self.model_directory.files:
                if isinstance(file, list) or isinstance(file, dict):
                    continue
                else:
                    self.essential_files.discard(file.name)
            return not self.essential_files

        else:
            # make sure that all default files in ModelDirectory are present
            return all([file is not None for file in self.model_directory.files])

    def _validate_deployment_directory(
        self,
        deployment_directory: File,
        files: Set,
        optional_files: Optional[Set] = set(),
    ) -> None:
        if any([isinstance(file, Directory) for file in deployment_directory.files]):
            raise ValueError(
                "Found nested directories within `deployment` directory. "
                "The directory may only contain files, not directories."
            )
        file_names = set(deployment_directory.get_file_names())
        for optional_file in optional_files:
            file_names.discard(optional_file)
        if files != file_names:
            raise ValueError(
                f"Failed to find expected files "
                f"{files.difference(file_names)} "
                f"in the `deployment` directory {deployment_directory.name}."
            )
        return True

    def _validate_training_directory(
        self,
        training_directory: File,
        files: Set,
        optional_files: Optional[Set] = set(),
    ) -> None:
        if isinstance(training_directory.files[0], Directory):
            # Training directory only contains subfolders
            # with names 'checkpoint_{...}'
            for file in training_directory.files:
                expected_name_prefix = "checkpoint_"
                if not file.name.startswith(expected_name_prefix):
                    raise ValueError(
                        f"Found a directory in `training` directory "
                        f"with the name: {file.name}. "
                        f"The name of the directory should "
                        f"start with '{expected_name_prefix}'."
                    )
                self._validate_training_directory(
                    training_directory=file, files=files, optional_files=optional_files
                )
        else:
            # Training directory does not contain any subfolders,
            # but the training files directly.
            file_names = set(training_directory.get_file_names())
            for optional_file in optional_files:
                file_names.discard(optional_file)

            if files != file_names:
                raise ValueError(
                    f"Failed to find expected files "
                    f"{files.difference(file_names)} "
                    f"in the `training` directory {training_directory.name}."
                )
            return True

    def _get_integration_data(
        self, model_card: Union[str, File]
    ) -> Tuple[Set, Set, Set]:
        integration_name = self._get_integration_name(model_card)
        return self.integration2data[integration_name]

    @staticmethod
    def _get_integration_name(model_card: Union[str, File]):
        if isinstance(model_card, str):
            name = os.path.basename(model_card)
            model_card = File(name=name, path=model_card)

        yaml_dict = model_card._validate_model_card()
        # TODO: swap this hack for parsing the proper "integration" field
        # in the future
        integration_name = f"{yaml_dict['domain']}"
        return integration_name
