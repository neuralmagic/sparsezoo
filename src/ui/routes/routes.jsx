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

import ModelsRoot from "./models-root";
import Models from "./models";

import { MODEL_TABLE_PATH, MODEL_TABLE_ROOT_PATH } from "./paths";

export function makeContentRoutes() {
  return [
    {
      path: MODEL_TABLE_ROOT_PATH,
      exact: true,
      component: ModelsRoot,
    },
    {
      path: MODEL_TABLE_PATH,
      exact: true,
      component: Models,
    },
  ];
}
