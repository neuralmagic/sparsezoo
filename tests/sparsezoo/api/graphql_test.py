from typing import Dict, List, Optional

from sparsezoo.version import version_base
print(version_base)

from sparsezoo.api import GraphQLAPI
# from sparsezoo import GraphQLAPI


class TestGraphQLAPI(GraphQLAPI):
    def __init__(
        self,
        expected_operation_body: str,
        response_objects: List[Dict],
    ):
        self._expected_operation_body = expected_operation_body
        self._response_objects = response_objects

    def make_request(
        self,
        operation_body: str,
        arguments: Dict[str, str],
        fields: Optional[List[str]] = None,
        url: Optional[str] = None,
    ) -> List[Dict]:
        assert self._expected_operation_body == operation_body

        return self._response_objects


def test_a_thing():
    subject = TestGraphQLAPI(
        expected_operation_body="blah",
        response_objects=[{
            "checkpoint": False,
            "created": None,
            "displayName": "special_tokens_map.json",
            "downloadUrl": "https://staging-api.neuralmagic.com/v2/models/835cc421-9763-4a0b-b818-4d0f031d9a88/files/special_tokens_map.json",
            "fileId": "0ce928b3-aef4-42ce-a58e-877a1251595b",
            "fileSize": 280,
            "fileType": "framework",
            "md5": "7b49a313ead23cad7e31f414fce47f91",
            "modelId": "835cc421-9763-4a0b-b818-4d0f031d9a88",
            "modified": None,
            "repoTagId": None,
            "s3VersionId": None,
            "version": 1
        }],
    )
    stuff = subject.fetch(operation_body="blah", arguments={})
    print(stuff)
