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

from sparsezoo.utils.registry import _ALIAS_REGISTRY, _REGISTRY, RegistryMixin


@pytest.fixture()
def foo():
    class Foo(RegistryMixin):
        pass

    yield Foo
    _ALIAS_REGISTRY.clear()
    _REGISTRY.clear()


class TestFooRegistry:
    def test_single_item(self, foo):
        @foo.register()
        class Foo1(foo):
            pass

        assert {"Foo1"} == set(foo.registered_names())
        assert set() == set(foo.registered_aliases())

    def test_single_item_custom_name(self, foo):
        @foo.register(name="name_2")
        class Foo1(foo):
            pass

        assert {"name_2"} == set(foo.registered_names())
        assert {"name-2"} == set(foo.registered_aliases())

    def test_alias(self, foo):
        @foo.register(alias=["name-3", "name_4"])
        class Foo1(foo):
            pass

        assert {"Foo1"} == set(foo.registered_names())
        assert {"name-3", "name-4", "name_3", "name_4"} == set(foo.registered_aliases())

    def test_alias_with_custom_name(self, foo):
        @foo.register(name="name_2", alias=["name-3", "name_4"])
        class Foo1(foo):
            pass

        assert {"name_2"} == set(foo.registered_names())
        assert {"name-3", "name-4", "name_3", "name_4", "name-2"} == set(
            foo.registered_aliases()
        )

    def test_get_value_from_registry(self, foo):
        @foo.register(alias=["name-3"])
        class Foo1(foo):
            pass

        @foo.register()
        class Foo2(foo):
            pass

        with pytest.raises(KeyError):
            foo.get_value_from_registry("Foo3")

        assert foo.get_value_from_registry("Foo1") is Foo1
        assert isinstance(foo.load_from_registry("Foo2"), Foo2)
        assert foo.get_value_from_registry("Foo2") is Foo2
        assert foo.get_value_from_registry("name_3") is Foo1
        assert foo.get_value_from_registry("name-3") is Foo1


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
