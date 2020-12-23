"""
Code related to a model repo user
"""

from typing import Dict
from sparsezoo.schemas.object_schema import ObjectSchema

__all__ = ["RepoUser"]


class RepoUser(ObjectSchema):
    """
    A model repo user

    :param email: contact email
    :param name: name of user
    :param user_id: user id
    :param trusted: Whether the user is a trusted source
    """

    def __init__(self, **kwargs):
        super(RepoUser, self).__init__(**kwargs)
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
