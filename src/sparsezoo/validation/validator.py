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

from sparsezoo.objects import Directory, File
from sparsezoo.validation import (
    validate_cv_classification,
    validate_cv_detection,
    validate_cv_segmentation,
    validate_nlp,
)


__all__ = ["IntegrationValidator"]

# files/directories that are required for minimal validation
REQUIRED_FILES = {"deployment", "training", "model.onnx"}

INTEGRATION_NAME_TO_DATA = {
    "nlp": validate_nlp,
    "cv_classification": validate_cv_classification,
    "cv_detection": validate_cv_detection,
    "cv_segmentation": validate_cv_segmentation,
}


class IntegrationValidator:
    """
    Helper class to facilitate the validation of the ModelDirectory class.
    It is called by the "validate" method of ModelDirectory.

    :param model: ModelDirectory class object to be
        validated
    :param required_files: a minimal set of file names in ModelDirectory,
        that need to be present, so that the validation is
        successful
    :param integration_to_data: mapping from the integration tag (string) from
        ModelDirectory's model card to validation function
    """

    def __init__(
        self,
        model: "Model",  # noqa F821
        required_files: Set[str] = REQUIRED_FILES,
        integration_to_data: Dict[str, Callable] = INTEGRATION_NAME_TO_DATA,
    ):
        self.model = model
        self.required_files = required_files
        self.integration_to_data = integration_to_data
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

        :param minimal_validation: boolean flag; if True, only the required files
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
        integration_name = self._get_integration_name(model_card=self.model.model_card)
        (
            training_files_validation,
            optional_training_files_validation,
            deployment_files_validation,
        ) = self.integration_to_data[integration_name]()
        for file in self.model.files:
            if file is None:
                continue
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
                    self.validate_directory(
                        directory=file,
                        validation_files=training_files_validation,
                        optional_validation_files=optional_training_files_validation,
                    )

                elif file.name == "deployment":
                    self.validate_directory(
                        directory=file,
                        validation_files=deployment_files_validation,
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
            for file in self.model.files:
                if isinstance(file, list) or isinstance(file, dict) or (file is None):
                    continue
                else:
                    self.required_files.discard(file.name)
            return not self.required_files

        else:
            # make sure that all default files in ModelDirectory are present
            return all([file is not None for file in self.model.files])

    def validate_directory(
        self,
        directory: Directory,
        validation_files: Set,
        optional_validation_files: Optional[Set] = set(),
    ) -> bool:
        """
        Validate the structure of the directory against a set of the
        expected files

        :param directory: The directory to be validated
        :validation_files: A set of file names expected to be
            found in the directory
        :optional_validation_files: Additional set of file names
            that may be optionally found in the directory
        :return True if the directory is valid
        """

        # partition contents of the `directory`
        # into loose files and subdirectories
        loose_files, subdirectories = [], []
        for file in directory.files:
            (loose_files, subdirectories)[isinstance(file, Directory)].append(file)

        # parse subdirectories in the
        # recursive manner
        for subdirectory in subdirectories:
            if isinstance(file, Directory):
                self.validate_directory(
                    directory=subdirectory,
                    validation_files=validation_files,
                    optional_validation_files=optional_validation_files,
                )

        # parse loose files
        if subdirectories and not loose_files:
            # directory contains only subdirectories and no files
            # this is fine
            return True
        file_names = {file.name for file in loose_files}
        for optional_file in optional_validation_files:
            file_names.discard(optional_file)

        missing_validation_files = validation_files - file_names
        if missing_validation_files:
            raise ValueError(
                "Failed to find expected files "
                f"{missing_validation_files} "
                f"in the `{directory.name}` directory."
            )

        return True

    def _get_integration_data(
        self, model_card: Union[str, File]
    ) -> Tuple[Set, Set, Set]:
        integration_name = self._get_integration_name(model_card)
        return self.integration_to_data[integration_name]

    @staticmethod
    def _get_integration_name(model_card: Union[str, File]):
        if isinstance(model_card, str):
            name = os.path.basename(model_card)
            model_card = File(name=name, path=model_card)

        yaml_dict = model_card._validate_model_card()

        if yaml_dict["domain"] == "nlp":
            integration_name = "nlp"
        else:
            integration_name = f"{yaml_dict['domain']}_{yaml_dict['sub_domain']}"
        return integration_name
