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
from typing import Set

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
        self.integration_to_expected_files = {
            "nlp-question_answering": self._validate_nlp,
            "nlp-token_classification": self._validate_nlp,
            "nlp-text_classification": self._validate_nlp,
            "nlp-masked_language_modelling": self._validate_nlp,
            "nlp-sentiment_analysis": self._validate_nlp,
            "cv-classification": self._validate_cv_classification,
            "cv-detection": self._validate_cv_detection,
            "cv-segmentation": self._validate_cv_segmentation,
        }
        self.model_directory = model_directory
        # TODO add minimal validation for every integration
        self.minimal_validation = None
        self.integration_name = self._get_integration_name()

    def validate(self, minimal_validation: bool) -> bool:
        """
        Validates the structure of ModelDirectory.

        :param minimal_validation: boolean flag; if True, only the essential files
            in the `training` folder are validated. Else, the `training` folder is
            expected to contain a full set of framework files.
        :return: boolean flag; True if files are valid and no errors arise
        """
        validations = {}
        self.minimal_validation = minimal_validation
        for file in self.model_directory.files:
            if isinstance(file, dict):
                validations[file.__repr__()] = all(
                    _file.validate() for _file in file.values()
                )
            elif isinstance(file, list):
                validations[file.__repr__()] = all(_file.validate() for _file in file)
            else:
                if file.name == "training":
                    self._validate_integration(file)

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

    def _validate_integration(self, training_directory: File) -> None:
        expected_file_names = self.integration_to_expected_files[
            self.integration_name
        ]()
        for file in training_directory.files:
            if isinstance(file, Directory):
                expected_name_prefix = "checkpoint_"
                # `training directory` can either contain expected files or
                # sub folders with names "checkpoint_{...}", that contain expected files
                if not file.name.startswith(expected_name_prefix):
                    raise ValueError(
                        f"Found a directory in `training` directory "
                        f"with the name: {file.name}. "
                        f"The name of the directory should "
                        f"start with '{expected_name_prefix}'."
                    )
                self._validate_integration(training_directory=file)
            else:
                file_names = training_directory.get_file_names()
                # if `self.minimal_validation`, we expect the training directory to
                # be a superset of the set of essential files
                if self.minimal_validation and not expected_file_names.issubset(
                    set(file_names)
                ):
                    self._throw_validate_integration_error(
                        training_directory=training_directory,
                        expected_file_names=expected_file_names,
                    )

                # if not `self.minimal_validation`, we expect the training directory to
                # be equal to the set of essential files
                elif not self.minimal_validation and expected_file_names != set(
                    file_names
                ):
                    self._throw_validate_integration_error(
                        training_directory=training_directory,
                        expected_file_names=expected_file_names,
                    )

    def _validate_nlp(self):
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
        minimum_file_names = {}

        return minimum_file_names if self.minimal_validation else file_names

    def _validate_cv_classification(self):
        raise NotImplementedError()

    def _validate_cv_detection(self):
        raise NotImplementedError()

    def _validate_cv_segmentation(self):
        raise NotImplementedError()

    @staticmethod
    def _throw_validate_integration_error(
        training_directory: Directory, expected_file_names: Set[str]
    ) -> None:
        raise ValueError(
            f"Failed to find expected files "
            f"{expected_file_names.difference(training_directory.get_file_names())} "
            f"in the training directory {training_directory.name}."
        )

    def _get_integration_name(self):
        model_card = self.model_directory.model_card
        yaml_dict = model_card._validate_model_card()
        integration_name = f"{yaml_dict['domain']}-{yaml_dict['sub_domain']}"
        return integration_name
