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


import logging

from sparsezoo.v2.directory import Directory
from sparsezoo.v2.file import File
from sparsezoo.v2.model_directory import ModelDirectory


__all__ = ["IntegrationValidator"]


class IntegrationValidator:
    """
    Helper class to facilitate the validation of the ModelDirectory class.
    It is called by the "validate" method of ModelDirectory.

    :param model_directory: ModelDirectory class object to be
        validated
    """

    def __init__(self, model_directory: ModelDirectory):

        # mapping from the integration name to the helper function
        # that outputs a set of expected `training` files
        self.integration_to_expected_training_files = {
            "nlp-question_answering": self._validate_nlp,
            "nlp-token_classification": self._validate_nlp,
            "nlp-text_classification": self._validate_nlp,
            "nlp-masked_language_modelling": self._validate_nlp,
            "nlp-sentiment_analysis": self._validate_nlp,
            "cv-classification": self._validate_cv_classification,
            "cv-detection": self._validate_cv_detection,
            "cv-segmentation": self._validate_cv_segmentation,
        }
        # set containing optional files that may be present
        # in `training` files
        self.optional_training_files = {"recipe.yaml"}

        # set containing files that must present
        # in `deployment` files on top of those that
        # need to be present in `training` files
        self.additional_deployment_files = {"model.onnx"}

        # set containing essential files that must be present
        # in `model_directory`
        self.essential_model_directory_files = {"deployment", "training", "model.onnx"}

        self.model_directory = model_directory
        self.minimal_validation = None
        self.integration_name = self._get_integration_name()

    def validate(self, minimal_validation: bool) -> bool:
        """
        Validates the structure and contents of ModelDirectory.

        1. Validate the general ModelDirectory structure (either against essential or
            full set of expected files) using self._validate_structure() method.

        2. Iterate over files in ModelDirectory and validate them. Some files have
            additional validation methods implemented.

        :param minimal_validation: boolean flag; if True, only the essential files
            in the ModelDirectory should be validated. Else, the ModelDirectory is
            expected to contain a full set of Directory/File class objects.
        :return: boolean flag; True if files are valid and no errors arise
        """
        validations = {}
        self.minimal_validation = minimal_validation
        if not self._validate_structure():
            mode = "essential" if self.minimal_validation else "full"
            raise ValueError(
                "The attempt to validate ModelDirectory's structure "
                f"(against the {mode} expected set of files)."
            )

        for file in self.model_directory.files:
            if isinstance(file, dict):
                validations[file.__repr__()] = all(
                    _file.validate() for _file in file.values()
                )
            elif isinstance(file, list):
                validations[file.__repr__()] = all(_file.validate() for _file in file)
            else:
                if file.name == "training":
                    self._validate_training_directory(training_directory=file)
                elif file.name == "deployment":
                    self._validate_deployment_directory(deployment_directory=file)

                validations[file.name] = file.validate()

        if all(validations.values()):
            return True
        else:
            logging.warning(
                "Following files (or directories): "
                f"{[key for key, value in validations.items() if not value]} "
                "were not successfully validated."
            )
            return False

    def _validate_structure(self):
        if self.minimal_validation:
            # iterate over all files in ModelDirectory and
            # check whether they match the expected essential
            # set of files
            for file in self.model_directory.files:
                if isinstance(file, list) or isinstance(file, dict):
                    continue
                else:
                    self.essential_model_directory_files.discard(file.name)
            return not self.essential_model_directory_files

        else:
            # make sure that all default files in ModelDirectory are present
            return [file for file in self.model_directory.files]

    def _validate_deployment_directory(self, deployment_directory: File) -> None:
        expected_file_names = self.integration_to_expected_training_files[
            self.integration_name
        ](deployment=True)
        for file in deployment_directory.files:
            if isinstance(file, Directory):
                raise ValueError(
                    "Found nested directories within `deployment` directory. "
                    "The directory may only contain files, not directories."
                )
            else:
                file_names = set(deployment_directory.get_file_names())
                if expected_file_names != file_names:
                    raise ValueError(
                        f"Failed to find expected files "
                        f"{expected_file_names.difference(file_names)} "
                        f"in the `deployment` directory {deployment_directory.name}."
                    )
                return

    def _validate_training_directory(self, training_directory: File) -> None:
        expected_file_names = self.integration_to_expected_training_files[
            self.integration_name
        ]()
        for file in training_directory.files:
            if isinstance(file, Directory):
                expected_name_prefix = "checkpoint_"
                # `training directory` can either contain
                # expected training files or
                # sub folders with names "checkpoint_{...}",
                # that contain expected training files
                if not file.name.startswith(expected_name_prefix):
                    raise ValueError(
                        f"Found a directory in `training` directory "
                        f"with the name: {file.name}. "
                        f"The name of the directory should "
                        f"start with '{expected_name_prefix}'."
                    )
                self._validate_training_directory(training_directory=file)
            else:
                file_names = set(training_directory.get_file_names())
                for optional_file in self.optional_training_files:
                    file_names.discard(optional_file)

                if expected_file_names != file_names:
                    raise ValueError(
                        f"Failed to find expected files "
                        f"{expected_file_names.difference(file_names)} "
                        f"in the `training` directory {training_directory.name}."
                    )
                return

    def _validate_nlp(self, deployment: bool = False):
        file_names = {
            "config.json",
            "pytorch_model.bin",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "trainer_state.json",
            "training_args.bin",
            "vocab.txt",
        }
        if deployment:
            file_names.update(self.additional_deployment_files)

        return file_names

    def _validate_cv_classification(self):
        raise NotImplementedError()

    def _validate_cv_detection(self):
        raise NotImplementedError()

    def _validate_cv_segmentation(self):
        raise NotImplementedError()

    def _get_integration_name(self):
        model_card = self.model_directory.model_card
        yaml_dict = model_card._validate_model_card()
        integration_name = f"{yaml_dict['domain']}-{yaml_dict['sub_domain']}"
        return integration_name
