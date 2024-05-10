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


@pytest.fixture()
def bar():
    class Bar(RegistryMixin):
        pass

    yield Bar
    _ALIAS_REGISTRY.clear()
    _REGISTRY.clear()


class TestRegistryFlowSingle:
    def test_single_item(self, foo):
        # register Foo1 in registry foo under the default name Foo1
        @foo.register()
        class Foo1(foo):
            pass

        assert {"foo1"} == set(foo.registered_names())
        assert set() == set(foo.registered_aliases())

    def test_single_item_custom_name(self, foo):
        # register Foo1 in registry foo under the name name_2
        # (this will turn name_2 into name-2, as we standardize names)
        @foo.register(name="name_2")
        class Foo1(foo):
            pass

        assert {"name-2"} == set(foo.registered_names())
        assert set() == set(foo.registered_aliases())
        # because we registered under a custom name, we can't
        # use the original object name
        with pytest.raises(KeyError):
            foo.get_value_from_registry("Foo1")

    def test_alias(self, foo):
        # register Foo1 in registry foo under the default name Foo1
        # and alias names name-3 and name_4
        @foo.register(alias=["name-3", "name_4"])
        class Foo1(foo):
            pass

        assert {"foo1"} == set(foo.registered_names())
        assert {"name-3", "name-4"} == set(foo.registered_aliases())

    def test_alias_is_standardized(self, foo):
        alias = "Name_3"
        expected_standardized_alias = "name-3"

        @foo.register(alias=[alias])
        class Foo1(foo):
            pass

        assert {expected_standardized_alias} == set(foo.registered_aliases())

    def test_key_error_on_duplicate_alias(self, foo):
        # once we register an object under one alias, we can't
        # register it under the same alias once again
        @foo.register(alias=["name-3"])
        class Foo1(foo):
            pass

        with pytest.raises(KeyError):

            @foo.register(alias=["name-3"])
            class Foo2(foo):
                pass

    def test_key_error_alias_equal_name(self, foo):
        # once we register the object under name-3
        # (not name_3, as we standardize names), we can't
        # register it under the same alias
        @foo.register(name="name_3")
        class Foo1(foo):
            pass

        with pytest.raises(KeyError):

            @foo.register(alias=["name-3"])
            class Foo2(foo):
                pass

    def test_key_error_alias_equal_name_simultaneously(self, foo):
        with pytest.raises(KeyError):

            @foo.register(name="name_3", alias=["name-3"])
            class Foo2(foo):
                pass

    def test_alias_with_custom_name(self, foo):
        @foo.register(name="name_2", alias=["name-3", "name_4"])
        class Foo1(foo):
            pass

        assert {"name-2"} == set(foo.registered_names())
        assert {"name-3", "name-4"} == set(foo.registered_aliases())

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
        assert foo.get_value_from_registry("name-3") is Foo1
        assert foo.get_value_from_registry("name_3") is Foo1
        assert foo.get_value_from_registry("name 3") is Foo1
        assert isinstance(foo.load_from_registry("name_3"), Foo1)


class TestRegistryFlowMultiple:
    def test_single_item(self, foo, bar):
        @foo.register()
        class Foo1(foo):
            pass

        @bar.register()
        class Bar1(bar):
            pass

        assert ["foo1"] == foo.registered_names()
        assert ["bar1"] == bar.registered_names()

        assert foo.get_value_from_registry("foo1") is Foo1
        assert bar.get_value_from_registry("bar1") is Bar1


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
