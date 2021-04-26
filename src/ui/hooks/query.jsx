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

import { useLocation } from "react-router-dom";

/**
 * Creates a hook for getting the query parameters of the url
 * @returns {object}
 */
export const useQuery = () => {
  const searchParams = new URLSearchParams(useLocation().search);

  const query = {};
  const entries = searchParams.entries();
  let { done, value } = entries.next();
  do {
    if (value) {
      query[value[0]] = value[1];
    }
    let entry = entries.next();
    done = entry.done;
    value = entry.value;
  } while (!done);
  return query;
};
