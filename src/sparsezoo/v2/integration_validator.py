import os
from sparsezoo.v2.model_directory import ModelDirectory


class IntegrationValidator:
    """
    IntegrationValidator will be called by the "validate" method of ModelDirectory.
    """

    def __init__(self, model_directory: ModelDirectory):

        validation_paths = {"nlp-question_answering": self._validate_nlp,
                            "nlp-token_classification": self._validate_nlp,
                            "nlp-text_classification": self._validate_nlp,
                            "nlp-masked_language_modelling": self._validate_nlp,
                            "nlp-sentiment_analysis": self._validate_nlp,
                            "cv-classification": self._validate_cv_classification,
                            "cv-detection":self._validate_cv_detection,
                            "cv-segmentation":self._validate_cv_segmentation
        }
        integration_name = validate(model_directory.model_card)
        validation_method = validation_paths[integration_name]
        validation_method(model_directory)

    def _validate(self):
        pass


    def _validate_nlp(self):
        pass

    def _validate_cv_classification(self):
        pass

    def _validate_cv_detection(self):
        pass

    def _validate_cv_classification(self):
        pass

    def _validate_cv_segmentation(self):
        pass




