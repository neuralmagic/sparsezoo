/*
Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

import { API_ROOT, validateAPIResponseJSON } from "./utils";

const PUBLIC_AUTH_TYPE = "public";

export function requestPostAuth() {
  const url = `${API_ROOT}/auth`;
  return validateAPIResponseJSON(
    fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ authentication_type: PUBLIC_AUTH_TYPE }),
    })
  );
}
