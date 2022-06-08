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


# from sparsezoo.v2.model_directory import ModelDirectory
from sparsezoo.v2.directory import Directory
from sparsezoo.v2.file import File


__all__ = ["IntegrationValidator"]


class IntegrationValidator:
    """
    IntegrationValidator will be called by the "validate" method of ModelDirectory.
    """

    def __init__(self, model_directory):

        self.validation_paths = {
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
        self.integration_name = self._get_integration_name(model_directory.model_card)

    def validate(self, minimal_validation):
        self.minimal_validation = minimal_validation
        for file in self.model_directory.files:
            if file.name == "training":
                self._validate_integration(file)

            file.validate()

    def _validate_integration(self, training_directory: File):
        expected_file_names = self.validation_paths[self.integration_name]
        for file in training_directory.files:
            if isinstance(file, Directory):
                self._validate_integration(training_directory=file)

            else:
                file_names = training_directory.get_file_names()
                if self.minimal_validation and not expected_file_names.issubset(
                    set(file_names)
                ):
                    raise ValueError("")

                elif not self.minimal_validation and expected_file_names != set(
                    file_names
                ):
                    raise ValueError("")

                else:
                    return True

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
    def _get_integration_name(model_card):
        yaml_dict = model_card._validate_model_card()
        integration_name = f"{yaml_dict['domain']}-{yaml_dict['sub_domain']}"
        return integration_name
