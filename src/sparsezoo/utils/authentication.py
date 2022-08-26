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

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Dict

import requests
import yaml

from . import BASE_API_URL
from .helpers import clean_path, create_parent_dirs


__all__ = ["get_auth_header"]

_LOGGER = logging.getLogger(__name__)

PUBLIC_AUTH_TYPE = "public"
CREDENTIALS_YAML_TOKEN_KEY = "nm_api_token"
AUTH_API = f"{BASE_API_URL}/auth"
NM_TOKEN_HEADER = "nm-token-header"


CREDENTIALS_YAML = os.path.abspath(
    clean_path(
        os.path.join(os.getenv("SPARSEZOO_CREDENTIALS_PATH"), "credentials.yaml")
    )
    if os.getenv("SPARSEZOO_CREDENTIALS_PATH")
    else clean_path(os.path.join("~", ".cache", "sparsezoo", "credentials.yaml"))
)


def get_auth_header(
    force_token_refresh: bool = False, path: str = CREDENTIALS_YAML
) -> Dict:
    """
    Obtain an authentication header token from either credentials file or from APIs
    if token is over 1 day old. Location of credentials file can be changed by setting
    the environment variable `NM_SPARSE_ZOO_CREDENTIALS`.

    Currently only 'public' authentication type is supported.

    :param force_token_refresh: forces a new token to be generated
    :return: An authentication header with key 'nm-token-header' containing the header
        token
    """
    token = _maybe_load_token(path)
    if token is None or force_token_refresh:
        _LOGGER.info("Obtaining new sparse zoo credentials token")
        response = requests.post(
            url=AUTH_API, data=json.dumps({"authentication_type": PUBLIC_AUTH_TYPE})
        )
        response.raise_for_status()
        token = response.json()["token"]
        created = time.time()
        _save_token(token, created, path)
    return {NM_TOKEN_HEADER: token}


def _maybe_load_token(path: str):
    if not os.path.exists(path):
        _LOGGER.debug(f"No sparse zoo credentials files found at {path}")
        return None

    _LOGGER.debug(f"Loading sparse zoo credentials from {path}")

    with open(path) as fp:
        creds = yaml.safe_load(fp)

    if creds is None or CREDENTIALS_YAML_TOKEN_KEY not in creds:
        _LOGGER.debug(f"No sparse zoo credentials found at {path}")
        return None

    info = creds[CREDENTIALS_YAML_TOKEN_KEY]
    if "token" not in info or "created" not in info:
        _LOGGER.debug(f"No sparse zoo credentials found at {path}")
        return None

    date_created = datetime.fromtimestamp(info["created"], tz=timezone.utc)
    creation_difference = datetime.now(tz=timezone.utc) - date_created

    if creation_difference.days > 30:
        _LOGGER.debug(f"Expired sparse zoo credentials at {path}")
        return None

    return info["token"]


def _save_token(token: str, created: float, path: str):
    """
    Save the jwt for accessing sparse zoo APIs. Will create the credentials file
    if it does not exist already.

    :param token: the jwt for accessing sparse zoo APIs
    :param created: the approximate time the token was created
    """
    _LOGGER.debug(f"Saving sparse zoo credentials at {CREDENTIALS_YAML}")
    if not os.path.exists(path):
        create_parent_dirs(path)
    with open(path, "w+") as fp:
        auth = {CREDENTIALS_YAML_TOKEN_KEY: dict(token=token, created=created)}
        yaml.safe_dump(auth, fp)
