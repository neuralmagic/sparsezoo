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


from typing import Any, Dict, List, Optional, Union

import requests

from sparsezoo.utils import BASE_API_URL

from .exceptions import graphqlapi_exception_handler, validate_graphql_response
from .query_parser import QueryParser
from .utils import map_keys, to_snake_case


class GraphQLAPI:
    def fetch(
        self,
        operation_body: str,
        arguments: Optional[Dict[str, Any]] = None,
        fields: Optional[Union[List[str], Dict[str, Optional[Dict]]]] = None,
        url: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Fetch data for models via api. Uses graphql convention of post,
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

    @graphqlapi_exception_handler
    def make_request(
        self,
        operation_body: str,
        arguments: Optional[Dict[str, str]] = None,
        fields: Optional[List[str]] = None,
        url: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Given the input args, parse them to a graphql appropriate format
            and make an graph post request to get the desired raw response.
        Raw response's keys are in camelCase, not snake_case
        """

        query = QueryParser(
            operation_body=operation_body, arguments=arguments, fields=fields
        )

        response = requests.post(
            url=url or f"{BASE_API_URL}/v2/graphql", json={"query": query.query_body}
        )

        validate_graphql_response(response=response, query_body=query.query_body)
        response_json = response.json()

        return response_json["data"][query.operation_body]
