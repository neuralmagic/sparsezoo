from typing import Dict
from sparsezoo.schemas.object_schema import ObjectSchema

__all__ = ["UserSchema"]


class UserSchema(ObjectSchema):
    def __init__(self, **kwargs):
        super(UserSchema, self).__init__(**kwargs)
        self._email = kwargs["email"]
        self._name = kwargs["name"]
        self._user_id = kwargs["user_id"]
        self._trusted = kwargs["trusted"]

    @property
    def email(self) -> str:
        return self._email

    @property
    def name(self) -> str:
        return self._name

    @property
    def user_id(self) -> str:
        return self._user_id

    @property
    def trusted(self) -> bool:
        return self._trusted

    def dict(self) -> Dict:
        return {
            "user_id": self.user_id,
            "email": self.email,
            "name": self.name,
            "trusted": self.trusted,
        }
