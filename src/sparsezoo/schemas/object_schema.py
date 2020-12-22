__all__ = ["ObjectSchema"]


class ObjectSchema:
    def __init__(self, **kwargs):
        self._created = kwargs["created"]
        self._modified = kwargs["modified"]

    @property
    def created(self) -> str:
        return self._created

    @property
    def modified(self) -> str:
        return self._modified
