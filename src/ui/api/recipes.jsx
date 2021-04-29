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

/**
 * API action for searching for models in the model zoo
 * @param {object} requestBody the requestBody
 * @param {string} requestBody.domain the domain of the model search
 * @param {string} requestBody.subdomain the subdomain of the model search
 * @param {string} requestBody.token the token for the model search authentication
 * @param {string} requestBody.page the page of search results to return
 * @param {string} requestBody.page_legth the amount of search results to return
 * @param {{
 *  architecture: string,
 *  sub_architecture: string,
 *  repo: string,
 *  framework: string,
 *  dataset: string,
 *  training_scheme: string,
 *  sparse_name: string,
 *  sparse_category: string,
 *  sparse_target: string,
 *  recipe_type: string
 * }} requestBody.queries the additional queries for the search result
 * @returns {Promise<Array>}
 */
export function requestSearchRecipes({
  domain,
  subdomain,
  token,
  page = 1,
  page_length = 20,
  queries = {},
}) {
  let url = `${API_ROOT}/recipes/search/${domain}/${subdomain}?page=${page}&page_length=${page_length}`;
  if (queries) {
    for (const [key, value] of Object.entries(queries)) {
      url = `${url}&${key}=${value}`;
    }
  }
  return validateAPIResponseJSON(
    fetch(url, {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
        "nm-token-header": token,
      },
    })
  );
}
