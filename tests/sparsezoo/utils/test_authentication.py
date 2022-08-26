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

from sparsezoo.utils.authentication import (
    get_auth_header,
    _maybe_load_token,
    _save_token,
    CREDENTIALS_YAML_TOKEN_KEY,
    NM_TOKEN_HEADER,
)
import pytest
from datetime import datetime, timedelta
import yaml
from unittest.mock import patch, MagicMock


def test_load_token_no_path(tmp_path):
    path = str(tmp_path / "token.yaml")
    assert _maybe_load_token(path) is None


def test_load_token_yaml_fail(tmp_path):
    path = str(tmp_path / "token.yaml")
    with open(path, "w") as fp:
        fp.write("asdf")
    assert _maybe_load_token(path) is None


@pytest.mark.parametrize(
    "content",
    [
        {},
        {CREDENTIALS_YAML_TOKEN_KEY: {}},
        {CREDENTIALS_YAML_TOKEN_KEY: {"token": "asdf"}},
        {CREDENTIALS_YAML_TOKEN_KEY: {"created": "asdf"}},
        {
            CREDENTIALS_YAML_TOKEN_KEY: {
                "created": (datetime.now() - timedelta(days=40)).timestamp()
            }
        },
    ],
)
def test_load_token_failure_cases(tmp_path, content):
    path = str(tmp_path / "token.yaml")
    with open(path, "w") as fp:
        yaml.dump(content, fp)
    assert _maybe_load_token(path) is None


def test_load_token_valid(tmp_path):
    auth = {
        CREDENTIALS_YAML_TOKEN_KEY: {
            "created": datetime.now().timestamp(),
            "token": "asdf",
        }
    }
    path = str(tmp_path / "token.yaml")
    with open(path, "w") as fp:
        yaml.dump(auth, fp)
    assert _maybe_load_token(path) == "asdf"


def test_load_saved_token(tmp_path):
    path = str(tmp_path / "some" / "dirs" / "token.yaml")
    _save_token("asdf", datetime.now().timestamp(), path)
    assert _maybe_load_token(path) == "asdf"


@patch("requests.post", return_value=MagicMock(json=lambda: {"token": "qwer"}))
def test_get_auth_token(post_mock, tmp_path):
    path = tmp_path / "creds.yaml"
    assert not path.exists()
    assert get_auth_header(path=str(path)) == {NM_TOKEN_HEADER: "qwer"}
    assert path.exists()
    post_mock.assert_called()
