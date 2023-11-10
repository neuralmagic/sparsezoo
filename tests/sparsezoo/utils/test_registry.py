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

import pytest

from sparsezoo.utils.registry import RegistryMixin


def test_registery_flow_single():
    class Foo(RegistryMixin):
        pass

    @Foo.register()
    class Foo1(Foo):
        pass

    assert {"Foo1"} == set(Foo.registered_names())

    @Foo.register(name="name_2")
    class Foo2(Foo):
        pass

    assert {"Foo1", "name_2"} == set(Foo.registered_names())

    @Foo.register(name=["name_3", "name_4"])
    class Foo3(Foo):
        pass

    assert {"Foo1", "name_2", "name_3", "name_4"} == set(Foo.registered_names())

    with pytest.raises(KeyError):
        Foo.get_value_from_registry("Foo2")

    assert Foo.get_value_from_registry("Foo1") is Foo1
    assert isinstance(Foo.load_from_registry("name_2"), Foo2)
    assert (
        Foo.get_value_from_registry("name_3")
        is Foo3
        is Foo.get_value_from_registry("name_4")
    )


def test_registry_flow_multiple():
    class Foo(RegistryMixin):
        pass

    class Bar(RegistryMixin):
        pass

    @Foo.register()
    class Foo1(Foo):
        pass

    @Bar.register()
    class Bar1(Bar):
        pass

    assert ["Foo1"] == Foo.registered_names()
    assert ["Bar1"] == Bar.registered_names()

    assert Foo.get_value_from_registry("Foo1") is Foo1
    assert Bar.get_value_from_registry("Bar1") is Bar1


def test_registry_requires_subclass():
    class Foo(RegistryMixin):
        registry_requires_subclass = True

    @Foo.register()
    class Foo1(Foo):
        pass

    with pytest.raises(ValueError):

        @Foo.register()
        class NotFoo:
            pass
