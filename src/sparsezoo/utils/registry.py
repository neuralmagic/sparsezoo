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
from typing import Any, Dict, Optional, Type


_REGISTRY: Dict[Type, Dict[str, Any]] = defaultdict(dict)


class RegistryMixin:
    """
    Universal registry to support registration and loading of child classes and plugins
    of neuralmagic utilities.

    Classes that require a registry or plugins may add the `RegistryMixin` and use
    `register` and `load` as the main entrypoints for adding new implementations and
    loading requested values from its registry.

    example
    ```python
    class Dataset(RegistryMixin):
        pass


    # register with default name
    @Dataset.register()
    class ImageNetDataset(Dataset)
        pass

    # load as "ImageNetDataset"
    imagenet = Dataset.load("ImageNetDataset")

    # register with custom name
    @Dataset.register(name="cifar-dataset")
    class Cifar(Dataset):
        pass

    # load as "cifar-dataset"
    cifar = Dataset.load_from_registry("cifar-dataset")

    # load from custom file that implements a dataset
    mnist = Dataset.load_from_registry("/path/to/mnnist_dataset.py:MnistDataset")
    ```
    """

    @classmethod
    def register(cls, name: Optional[str] = None):
        def decorator(value: Any):
            cls.register_value(value, name=name)
            return value

        return decorator

    @classmethod
    def register_value(
        cls, value: Any, name: Optional[str] = None, require_subclass: bool = False
    ):
        _register(
            parent_class=cls,
            value=value,
            name=name,
            require_subclass=require_subclass,
        )

    @classmethod
    def load_from_registry(
        cls, class_name: str, require_subclass: bool = False, **constructor_kwargs
    ):
        constructor = cls.get_value(
            class_name=class_name, require_subclass=require_subclass
        )
        return constructor(**constructor_kwargs)

    @classmethod
    def get_value_from_registry(cls, class_name: str, require_subclass: bool = False):
        return _get_from_registry(
            parent_class=cls, name=class_name, require_subclass=require_subclass
        )


def _register(
    parent_class: Type,
    value: Any,
    name: Optional[str] = None,
    require_subclass: bool = False,
):
    if name is None:
        name = value.__name__

    if require_subclass:
        _validate_subclass(parent_class, value)

    if name in _REGISTRY[parent_class]:
        registered_value = _REGISTRY[parent_class][name]
        if registered_value is not value:
            raise RuntimeError(
                f"Attempting to register name {name} as {value} "
                f"however {name} has already been registered as {registered_value}"
            )
    else:
        _REGISTRY[parent_class][name] = value


def _get_from_registry(
    parent_class: Type, name: str, require_subclass: bool = False
) -> Any:

    if ":" in name:
        # user specifying specific module to load and value to import
        module_path, value_name = name.split(":")
        retrieved_value = _import_and_get_value_from_module(module_path, value_name)
    else:
        retrieved_value = _REGISTRY[parent_class].get(name)
        if retrieved_value is None:
            raise ValueError(
                f"Unable to find {name} registered under type {parent_class}. "
                f"Registered values for {parent_class}: "
                f"{list(_REGISTRY[parent_class].keys())}"
            )

    if require_subclass:
        _validate_subclass(parent_class, retrieved_value)

    return retrieved_value


def _import_and_get_value_from_module(module_path: str, value_name: str) -> Any:
    # load module
    spec = importlib.util.spec_from_file_location(
        f"plugin_module_for_{value_name}", module_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # get value from module
    value = getattr(module, value_name)

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
