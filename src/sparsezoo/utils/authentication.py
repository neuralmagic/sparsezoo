import os
import json
import logging
import time
from datetime import datetime, timezone
import yaml

import requests
from sparsezoo.utils.helpers import BASE_API_URL, clean_path

__all__ = ["get_auth_header"]

_LOGGER = logging.getLogger(__name__)

NM_TOKEN_HEADER = "nm-token-header"
AUTH_API = os.path.join(BASE_API_URL, "auth")
PUBLIC_AUTH_TYPE = "public"

CREDENTIALS_YAML = os.path.abspath(
    os.getenv("NM_SPARSE_ZOO_CREDENTIALS")
    if os.getenv("NM_SPARSE_ZOO_CREDENTIALS")
    else clean_path(os.path.join("~", ".cache", "nm_models", "credentials.yaml"))
)

CREDENTIALS_YAML_TOKEN_KEY = "nm_api_token"


class SparseZooCredentials:
    def __init__(self):
        if os.path.exists(CREDENTIALS_YAML):
            _LOGGER.info(f"Loading sparse zoo credentials from {CREDENTIALS_YAML}")
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
            _LOGGER.info(f"No sparse zoo credentials files found at {CREDENTIALS_YAML}")
            self._token = None
            self._created = None

    def save_token(self, token: str, created):
        _LOGGER.info(f"Saving sparse zoo credentials at {CREDENTIALS_YAML}")
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
        _LOGGER.info(f"Obtaining sparse zoo credentials from {CREDENTIALS_YAML}")
        if self._token and self._created is not None:
            creation_date = datetime.fromtimestamp(self._created, tz=timezone.utc)
            creation_difference = datetime.now(tz=timezone.utc) - creation_date
            if creation_difference.days == 0:
                return self._token
            else:
                _LOGGER.warning(f"Expired sparse zoo credentials at {CREDENTIALS_YAML}")
                return None
        else:
            _LOGGER.warning(f"No sparse zoo credentials found at {CREDENTIALS_YAML}")
            return None


def get_auth_header(
    user_id: str = None,
    app_id: str = None,
    authentication_type: str = PUBLIC_AUTH_TYPE,
    refresh_token: bool = False,
) -> str:
    credentials = SparseZooCredentials()
    token = credentials.token
    if token and not refresh_token:
        return {NM_TOKEN_HEADER: token}
    elif authentication_type.lower() == PUBLIC_AUTH_TYPE:
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
