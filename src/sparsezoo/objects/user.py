"""
Code related to a model repo user
"""

from sparsezoo.objects.base import BaseObject


__all__ = ["User"]


class User(BaseObject):
    """
    A model repo user

    :param email: contact email
    :param name: name of user
    :param user_id: user id
    :param trusted: Whether the user is a trusted source
    """

    def __init__(
        self,
        email: str,
        name: str,
        user_id: str,
        trusted: bool,
        **kwargs,
    ):
        super(User, self).__init__(**kwargs)
        self._email = email
        self._name = name
        self._user_id = user_id
        self._trusted = trusted

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
