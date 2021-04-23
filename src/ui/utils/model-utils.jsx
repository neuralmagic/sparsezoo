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

import lodash from "lodash";

/**
 *
 * @param {Object<string, Any>} model the model information
 * @returns {string} SparseZoo model stub
 */
export const getModelStub = (model) => {
  const archId = model.sub_architecture
    ? `${model.architecture}-${model.sub_architecture}`
    : model.architecture;

  const trainingId = model.training_scheme
    ? `${model.dataset}-${model.training_scheme}`
    : model.dataset;

  const sparseId = model.sparse_target
    ? `${model.sparse_name}-${model.sparse_category}-${model.sparse_target}`
    : `${model.sparse_name}-${model.sparse_category}`;

  return `zoo:${model.domain}/${model.sub_domain}/${archId}/${model.framework}/${model.repo}/${trainingId}/${sparseId}`;
};

/**
 * Formats model's result objects of a specific type
 * @param {object} model the model information
 * @param {{
 *  recorded_value: number,
 *  recorded_units: string,
 *  result_type: string
 * }[]} model.results the results of the model
 * @param {string} type the type of result to obtain from the model
 * @type {string[]}
 */
export const getFormattedData = (model, type) => {
  let results = lodash
    .get(model, "results", [])
    .filter((result) => result.result_type === type);
  if (results.length === 0) {
    return ["--"];
  } else {
    return results.map((result) => `${result.recorded_value} ${result.recorded_units}`);
  }
};
