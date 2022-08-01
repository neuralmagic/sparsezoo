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


class SparseZooCredentials:
    """
    Class wrapping around the sparse zoo credentials file.
    """

    def __init__(self):
        if os.path.exists(CREDENTIALS_YAML):
            _LOGGER.debug(f"Loading sparse zoo credentials from {CREDENTIALS_YAML}")
            with open(CREDENTIALS_YAML) as credentials_file:
                credentials_yaml = yaml.safe_load(credentials_file)
                if credentials_yaml and CREDENTIALS_YAML_TOKEN_KEY in credentials_yaml:
                    self._token = credentials_yaml[CREDENTIALS_YAML_TOKEN_KEY]["token"]
                    self._created = credentials_yaml[CREDENTIALS_YAML_TOKEN_KEY][
                        "created"
                    ]
                else:
                    self._token = None
                    self._created = None
        else:
            _LOGGER.debug(
                f"No sparse zoo credentials files found at {CREDENTIALS_YAML}"
            )
            self._token = None
            self._created = None

    def save_token(self, token: str, created: float):
        """
        Save the jwt for accessing sparse zoo APIs. Will create the credentials file
        if it does not exist already.

        :param token: the jwt for accessing sparse zoo APIs
        :param created: the approximate time the token was created
        """
        _LOGGER.debug(f"Saving sparse zoo credentials at {CREDENTIALS_YAML}")
        if not os.path.exists(CREDENTIALS_YAML):
            create_parent_dirs(CREDENTIALS_YAML)
        with open(CREDENTIALS_YAML, "w+") as credentials_file:
            credentials_yaml = yaml.safe_load(credentials_file)
            if credentials_yaml is None:
                credentials_yaml = {}
            credentials_yaml[CREDENTIALS_YAML_TOKEN_KEY] = {
                "token": token,
                "created": created,
            }
            self._token = token
            self._created = created

            yaml.safe_dump(credentials_yaml, credentials_file)

    @property
    def token(self):
        """
        :return: obtain the token if under 1 day old, else return None
        """
        _LOGGER.debug(f"Obtaining sparse zoo credentials from {CREDENTIALS_YAML}")
        if self._token and self._created is not None:
            creation_date = datetime.fromtimestamp(self._created, tz=timezone.utc)
            creation_difference = datetime.now(tz=timezone.utc) - creation_date
            if creation_difference.days < 30:
                return self._token
            else:
                _LOGGER.debug(f"Expired sparse zoo credentials at {CREDENTIALS_YAML}")
                return None
        else:
            _LOGGER.debug(f"No sparse zoo credentials found at {CREDENTIALS_YAML}")
            return None


def get_auth_header(
    authentication_type: str = PUBLIC_AUTH_TYPE,
    force_token_refresh: bool = False,
) -> Dict:
    """
    Obtain an authentication header token from either credentials file or from APIs
    if token is over 1 day old. Location of credentials file can be changed by setting
    the environment variable `NM_SPARSE_ZOO_CREDENTIALS`.

    Currently only 'public' authentication type is supported.

    :param authentication_type: authentication type for generating token
    :param force_token_refresh: forces a new token to be generated
    :return: An authentication header with key 'nm-token-header' containing the header
        token
    """
    credentials = SparseZooCredentials()
    token = credentials.token
    if token and not force_token_refresh:
        return {NM_TOKEN_HEADER: token}
    elif authentication_type.lower() == PUBLIC_AUTH_TYPE:
        _LOGGER.info("Obtaining new sparse zoo credentials token")
        created = time.time()
        response = requests.post(
            url=AUTH_API, data=json.dumps({"authentication_type": PUBLIC_AUTH_TYPE})
        )
        response.raise_for_status()
        token = response.json()["token"]
        credentials.save_token(token, created)
        return {NM_TOKEN_HEADER: token}
    else:
        raise Exception(f"Authentication type {PUBLIC_AUTH_TYPE} not supported.")
