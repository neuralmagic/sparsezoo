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


from typing import Any, Dict, List, Optional

import requests

from sparsezoo.utils import BASE_API_URL

from .query_parser import QueryParser
from .utils import map_keys, to_snake_case


QUERY_BODY = """
    {{
        {operation_body} {arguments}
            {{
               {fields}
            }}
    }}
"""


class GraphQLAPI:
    def get_file_download_url(
        self,
        model_id: str,
        file_name: str,
        base_url: str = BASE_API_URL,
    ):
        """Url to download a file"""
        return f"{base_url}/v2/models/{model_id}/files/{file_name}"

    def fetch(
        self,
        operation_body: str,
        arguments: Dict[str, str],
        fields: Optional[List[str]] = None,
        url: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Fetch data for models via api. Uses graohql convention of post,
            not get for requests.
        Input args are parsed to make a query body for the api request.
        For more details on the appropriate values, please refer to the
            url endpoint on the browser

        :param operation_body: The data object of interest
        :param arguments: Used to filter data object in the backend
        :param field: the object's field of interest
        """

        response_objects = self.make_request(
            operation_body=operation_body,
            arguments=arguments,
            fields=fields,
            url=url,
        )

        return [
            map_keys(dictionary=response_object, mapper=to_snake_case)
            for response_object in response_objects
        ]

    def make_request(
        self,
        operation_body: str,
        arguments: Dict[str, str],
        fields: Optional[List[str]] = None,
        url: Optional[str] = None,
    ) -> Dict:
        query = QueryParser(
            operation_body=operation_body, arguments=arguments, fields=fields
        )
        query.parse()

        response = requests.post(
            url=url or f"{BASE_API_URL}/v2/graphql",
            json={
                "query": QUERY_BODY.format(
                    operation_body=query.operation_body,
                    arguments=query.arguments,
                    fields=query.fields,
                )
            },
        )

        response.raise_for_status()
        response_json = response.json()

        return response_json["data"][query.operation_body]
