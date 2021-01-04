from typing import Dict


__all__ = ["SparseZooObject"]


class SparseZooObject:
    """
    A sparse zoo object

    :param created: the date created
    :param modified: the date modifed
    """

    def __init__(self, **kwargs):
        self._created = kwargs["created"]
        self._modified = kwargs["modified"]

    @property
    def created(self) -> str:
        """
        :return: the date created
        """
        return self._created

    @property
    def modified(self) -> str:
        """
        :return: the date modifed
        """
        return self._modified

    def _get_properties(self) -> str:
        return vars(self).keys()

    def dict(self) -> Dict:
        """
        :return: The object as a dictionary
        """
        prop_dict = {}
        for prop in self._get_properties():
            if prop[0] == "_":
                prop = prop[1:]

            if not hasattr(self, prop):
                continue

            prop_value = getattr(self, prop)

            if isinstance(prop_value, SparseZooObject) or issubclass(
                type(prop_value), SparseZooObject
            ):
                prop_dict[prop] = prop_value.dict()
            elif isinstance(prop_value, list):
                prop_dict[prop] = [
                    elem.dict()
                    if isinstance(elem, SparseZooObject)
                    or issubclass(type(elem), SparseZooObject)
                    else elem
                    for elem in prop_value
                ]
            else:
                prop_dict[prop] = prop_value

        return prop_dict
