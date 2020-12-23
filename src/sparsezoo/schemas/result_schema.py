"""
Code related to a model repo results
"""

from typing import Dict
from sparsezoo.schemas.object_schema import ObjectSchema

__all__ = ["RepoResult"]


class RepoResult(ObjectSchema):
    """
    A model repo result

    :param result_id: the result id
    :param display_name: the display name for the result
    :param result_type: the result type e.g. benchmark or performance
    :param result_category: the result category e.g. cpu or gpu
    :param model_id: the model id of the model the result is for
    :param recorded_value: the recorded value of the result
    :param recorded_units: the units the recorded value was in
    :param recorded_format: any information of recorded format
    """

    def __init__(self, **kwargs):
        super(RepoResult, self).__init__(**kwargs)
        self._result_id = kwargs["result_id"]
        self._display_name = kwargs["display_name"]
        self._result_type = kwargs["result_type"]
        self._result_category = kwargs["result_category"]
        self._model_id = kwargs["model_id"]
        self._recorded_value = kwargs["recorded_value"]
        self._recorded_units = kwargs["recorded_units"]
        self._recorded_format = kwargs["recorded_format"]

    @property
    def result_id(self) -> str:
        return self._result_id

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def display_name(self) -> str:
        return self._display_name

    @property
    def result_type(self) -> str:
        return self._result_type

    @property
    def result_category(self) -> str:
        return self._result_category

    @property
    def recorded_value(self) -> float:
        return self._recorded_value

    @property
    def recorded_units(self) -> str:
        return self._recorded_units

    @property
    def recorded_format(self) -> str:
        return self._recorded_format

    def dict(self) -> Dict:
        return {
            "result_id": self.result_id,
            "model_id": self.model_id,
            "display_name": self.display_name,
            "result_type": self.result_type,
            "result_category": self.result_category,
            "recorded_value": self.recorded_value,
            "recorded_units": self.recorded_units,
            "recorded_format": self.recorded_format,
        }
