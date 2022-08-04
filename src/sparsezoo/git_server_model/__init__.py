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
Git server model:

given one of {url, git clone https, git clone ssh},
output the model metadata

usage

url = https://git.neuralmagic.com/neuralmagic/cary
url = git@git.neuralmagic.com:neuralmagic/cary.git
url = https://git.neuralmagic.com/neuralmagic/cary.git

model = Model(url)
model.benchmark.metadata # benchmark metadata
model.model.metadata # model.md metadata

"""


from .models import *
from .utils import *
from .validations import *
