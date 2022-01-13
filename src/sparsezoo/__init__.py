# flake8: noqa

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
- Functionality for accessing models, recipes, and supporting files in the SparseZoo
- Notify the user the last pypi package version
"""

# flake8: noqa
# isort: skip_file

from .version import *
from .main import *
from .download_main import *
from .models.zoo import *
from .objects import *
from .package import *


from sparsezoo.package import check_package_version as _check_package_version

_check_package_version(
    package_name=__name__ if is_release else f"{__name__}-nightly",
    package_version=version,
)
