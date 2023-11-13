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

"""
Universal registry to support registration and loading of child classes and plugins
of neuralmagic utilities
"""

import importlib
from collections import defaultdict
from typing import Any, Dict, List, Optional, Type, Union


__all__ = [
    "RegistryMixin",
    "register",
    "get_from_registry",
    "registered_names",
]


_REGISTRY: Dict[Type, Dict[str, Any]] = defaultdict(dict)


class RegistryMixin:
    """
    Universal registry to support registration and loading of child classes and plugins
    of neuralmagic utilities.

    Classes that require a registry or plugins may add the `RegistryMixin` and use
    `register` and `load` as the main entrypoints for adding new implementations and
    loading requested values from its registry.

    If a class should only have its child classes in its registry, the class should
    set the static attribute `registry_requires_subclass` to True

    example
    ```python
    class Dataset(RegistryMixin):
        pass


    # register with default name
    @Dataset.register()
    class ImageNetDataset(Dataset):
        pass

    # load as "ImageNetDataset"
    imagenet = Dataset.load("ImageNetDataset")

    # register with custom name
    @Dataset.register(name="cifar-dataset")
    class Cifar(Dataset):
        pass

    # register with multiple aliases
    @Dataset.register(name=["cifar-10-dataset", "cifar-100-dataset"])
    class Cifar(Dataset):
        pass

    # load as "cifar-dataset"
    cifar = Dataset.load_from_registry("cifar-dataset")

    # load from custom file that implements a dataset
    mnist = Dataset.load_from_registry("/path/to/mnnist_dataset.py:MnistDataset")
    ```
    """

    # set to True in child class to add check that registered/retrieved values
    # implement the class it is registered to
    registry_requires_subclass: bool = False

    @classmethod
    def register(cls, name: Union[List[str], str, None] = None):
        """
        Decorator for registering a value (ie class or function) wrapped by this
        decorator to the base class (class that .register is called from)

        :param name: name or list of names to register the wrapped value as,
            defaults to value.__name__
        :return: register decorator
        """

        def decorator(value: Any):
            cls.register_value(value, name=name)
            return value

        return decorator

    @classmethod
    def register_value(cls, value: Any, name: Union[List[str], str, None] = None):
        """
        Registers the given value to the class `.register_value` is called from
        :param value: value to register
        :param name: name or list of names to register the wrapped value as,
            defaults to value.__name__
        """
        names = name if isinstance(name, list) else [name]

        for name in names:
            register(
                parent_class=cls,
                value=value,
                name=name,
                require_subclass=cls.registry_requires_subclass,
            )

    @classmethod
    def load_from_registry(cls, name: str, **constructor_kwargs) -> object:
        """
        :param name: name of registered class to load
        :param constructor_kwargs: arguments to pass to the constructor retrieved
            from the registry
        :return: loaded object registered to this class under the given name,
            constructed with the given kwargs. Raises error if the name is
            not found in the registry
        """
        constructor = cls.get_value_from_registry(name=name)
        return constructor(**constructor_kwargs)

    @classmethod
    def get_value_from_registry(cls, name: str):
        """
        :param name: name to retrieve from the registry
        :return: value from retrieved the registry for the given name, raises
            error if not found
        """
        return get_from_registry(
            parent_class=cls,
            name=name,
            require_subclass=cls.registry_requires_subclass,
        )

    @classmethod
    def registered_names(cls) -> List[str]:
        """
        :return: list of all names registered to this class
        """
        return registered_names(cls)


def register(
    parent_class: Type,
    value: Any,
    name: Optional[str] = None,
    require_subclass: bool = False,
):
    """
    :param parent_class: class to register the name under
    :param value: the value to register
    :param name: name to register the wrapped value as, defaults to value.__name__
    :param require_subclass: require that value is a subclass of the class this
        method is called from
    """
    if name is None:
        # default name
        name = value.__name__

    if require_subclass:
        _validate_subclass(parent_class, value)

    if name in _REGISTRY[parent_class]:
        # name already exists - raise error if two different values are attempting
        # to share the same name
        registered_value = _REGISTRY[parent_class][name]
        if registered_value is not value:
            raise RuntimeError(
                f"Attempting to register name {name} as {value} "
                f"however {name} has already been registered as {registered_value}"
            )
    else:
        _REGISTRY[parent_class][name] = value


def get_from_registry(
    parent_class: Type, name: str, require_subclass: bool = False
) -> Any:
    """
    :param parent_class: class that the name is registered under
    :param name: name to retrieve from the registry of the class
    :param require_subclass: require that value is a subclass of the class this
        method is called from
    :return: value from retrieved the registry for the given name, raises
        error if not found
    """

    if ":" in name:
        # user specifying specific module to load and value to import
        module_path, value_name = name.split(":")
        retrieved_value = _import_and_get_value_from_module(module_path, value_name)
    else:
        # look up name in registry
        retrieved_value = _REGISTRY[parent_class].get(name)
        if retrieved_value is None:
            raise KeyError(
                f"Unable to find {name} registered under type {parent_class}. "
                f"Registered values for {parent_class}: "
                f"{registered_names(parent_class)}"
            )

    if require_subclass:
        _validate_subclass(parent_class, retrieved_value)

    return retrieved_value


def registered_names(parent_class: Type) -> List[str]:
    """
    :param parent_class: class to look up the registry of
    :return: all names registered to the given class
    """
    return list(_REGISTRY[parent_class].keys())


def _import_and_get_value_from_module(module_path: str, value_name: str) -> Any:
    # import the given module path and try to get the value_name if it is included
    # in the module

    # load module
    spec = importlib.util.spec_from_file_location(
        f"plugin_module_for_{value_name}", module_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # get value from module
    value = getattr(module, value_name, None)

    if not value:
        raise RuntimeError(
            f"Unable to find attribute {value_name} in module {module_path}"
        )
    return value


def _validate_subclass(parent_class: Type, child_class: Type):
    if not issubclass(child_class, parent_class):
        raise ValueError(
            f"class {child_class} is not a subclass of the class it is "
            f"registered for: {parent_class}."
        )
