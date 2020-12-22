from typing import Dict
import logging

from sparsezoo.schemas.file_schema import FileSchema
from sparsezoo.schemas.downloadable_schema import ModelDownloadableSchema

__all__ = ["OptimizationSchema"]

_LOGGER = logging.getLogger(__name__)


class OptimizationSchema(ModelDownloadableSchema):
    def __init__(self, **kwargs):
        super(OptimizationSchema, self).__init__(**kwargs)
        self._optimization_id = kwargs["optimization_id"]
        self._model_id = kwargs["model_id"]
        self._display_name = kwargs["display_name"]
        self._display_description = kwargs["display_description"]
        self._optimization_type = kwargs["optimization_type"]

        if "file" in kwargs:
            self._file = kwargs["file"]
        else:
            self._file = None

    @property
    def display_name(self):
        return self._display_name

    @property
    def display_description(self):
        return self._display_description

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def optimization_id(self) -> str:
        return self._optimization_id

    @property
    def optimization_type(self) -> str:
        return self._optimization_type

    @property
    def file(self) -> FileSchema:
        return self._file

    def download(
        self,
        overwrite: bool = False,
        save_dir: str = None,
        save_path: str = None,
    ) -> str:
        _LOGGER.info(
            f"Downloading recipe files for {self.display_name} from model {self.model_id}."
        )
        if self.file is None:
            raise Exception(
                "File {} has not been signed. Please use download API to obtain a signed version of this model.".format(
                    self.model_id
                )
            )
        else:
            self.file.download(
                overwrite=overwrite, save_dir=save_dir, save_path=save_path
            )

    def dict(self) -> Dict:
        return {
            "optimization_id": self.optimization_id,
            "optimization_type": self.optimization_type,
            "model_id": self.model_id,
            "display_description": self.display_description,
            "display_name": self.display_name,
            "file": self.file,
        }
