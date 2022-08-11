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

# flake8: noqa

# GIT_SERVER_REGEX = r"^(.*)?git.neuralmagic.com[/:](?P<namespace>[a-zA-Z0-9]+)/(?P<name>[a-zA-Z0-9]+)"
GIT_SERVER_REGEX = r"^(?:git@git.neuralmagic.com:|https://git.neuralmagic.com/)(?P<namespace>[a-zA-Z0-9]+)/(?P<name>[a-zA-Z0-9]+)"
RAW_FILE_CONTENT_URL = (
    "https://git.neuralmagic.com/{namespace}/{name}/-/raw/{branch}/{file_name}"
)