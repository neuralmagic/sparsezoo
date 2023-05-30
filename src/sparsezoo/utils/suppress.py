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

import sys
from contextlib import contextmanager

__all__ = ["suppress_stdout_stderr", "NullDevice"]


class NullDevice():
    def write(self, s):
        pass

    def flush(self):
        pass


@contextmanager
def suppress_stdout_stderr(suppress: bool = True):
    """
    Suppresses stdout and stderr for the duration of the context.
    """
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    null_device = NullDevice()

    try:
        if suppress:
            # Redirect stdout and stderr to the null device
            sys.stdout = null_device
            sys.stderr = null_device
        yield
    finally:
        # Restore the original stdout and stderr
        sys.stdout = original_stdout
        sys.stderr = original_stderr
