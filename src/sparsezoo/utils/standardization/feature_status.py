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

"""
Utilities for representing feature status
"""


__all__ = [
    "FeatureStatus",
]


_STATUS_TO_GITHUB_EMOJI = {
    "y": ":white_check_mark:",  # (yes) - implemented by NM
    "e": ":heavy_check_mark:",  # (external) - implemented by external integration
    "n": ":x:",  # (no) - not implemented yet
    "q": ":question:",  # (question) - not sure, not tested, or to be investigated
}

_STATUS_HELP_TEXT = """
### Key
 * :white_check_mark: - implemented by neuralmagic integration
 * :heavy_check_mark: - implemented by underlying integration
 * :x: - not implemented yet
 * :question: - not sure, not tested, or to be investigated
""".strip()

_YAML_HELP_TEXT = """
###########################################################
# Status Keys:
# y: yes - implemented by NM
# e: external - implemented by external integration
# n: no - not implemented yet
# q: question - not sure, not tested, or to be investigated
###########################################################
""".strip()


class FeatureStatus(str):
    """
    Valid feature status codes mapped to emojis to render in github

    Valid values:
    'y' - (yes) - implemented by NM
    'e' - (external) - implemented by external integration
    'n' - (no) - not implemented yet
    'q' - (question) - not sure, not tested, or to be investigated
    """

    MARKDOWN_HELP = _STATUS_HELP_TEXT
    YAML_HELP = _YAML_HELP_TEXT
    VALID_VALUES = list(_STATUS_TO_GITHUB_EMOJI.values())

    def github_emoji(self) -> str:
        """
        :return: github emoji to represent this status code
        """
        self.validate(str(self))
        return _STATUS_TO_GITHUB_EMOJI[self]

    @classmethod
    def __get_validators__(cls):
        # pydantic validation
        yield cls.validate

    @classmethod
    def validate(cls, value):
        """
        :param value: string value to validate
        :raises ValueError: if the value of this status is not in the
            approved set of status values
        """
        if value not in _STATUS_TO_GITHUB_EMOJI:
            raise ValueError(
                f"Invalid feature status code: {value}. "
                f"Valid codes: {cls.VALID_VALUES}"
            )
        return cls(value)
