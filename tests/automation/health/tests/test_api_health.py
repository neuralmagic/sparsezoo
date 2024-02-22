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

import logging

import requests


LOGGER = logging.getLogger(__name__)


API_URL = "https://api.neuralmagic.com/v2/graphql"
HEADERS = {
    "Content-Type": "application/json",
}


def test_graphql_api_health():
    try:
        payload = {
            "operationName": "GetModelsList",
            "variables": {},
            "query": """query GetModelsList {
                models(isPubliclyVisible: true) {
                    ...ModelFields
                    __typename
                }
            }
            fragment ModelFields on Model {
              architecture
              baseModelId
              displayDescription
              displayName
              domain
              domainDisplayName
              baseModelId
              downloadCount
              framework
              modelId
              modelOnnxSizeCompressedBytes
              modelOnnxSizeCompressionRatio
              repo
              repoName
              repoDisplayName
              sourceDataset
              sparseTag
              subArchitecture
              stub
              tags {
                displayName
                __typename
              }
              task
              taskDisplayName
              trainingDataset
              modelCardDownloadUrl
              __typename
            }""",
        }

        response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=30)
        response.raise_for_status()

        data = response.json()
        assert (
            "data" in data and "models" in data["data"]
        ), "Expected 'models' field in the response"
        assert (
            len(data["data"]["models"]) > 100
        ), f"Expected more than 100 models but got {len(data['data']['models'])}"

    except requests.RequestException as e:
        LOGGER.error(f"Error when accessing GraphQL API: {e}")
        raise AssertionError(f"Failed to access GraphQL API: {e}") from e
    except AssertionError as e:
        LOGGER.error(f"Data verification error: {e}")
        raise e
