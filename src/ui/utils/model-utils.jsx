import _ from "lodash";

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
 * @param {Object<string, Any>} model the model information
 * @param {string} category the category of result to obtain from the model
 * @returns {Array<string>} The results of category formatted with units
 */
export const getFormattedData = (model, category) => {
  let results = _.get(model, "results", []).filter(
    (result) => result.result_category === category
  );
  if (results.length === 0) {
    return ["--"];
  } else {
    return results.map((result) => `${result.recorded_value} ${result.recorded_units}`);
  }
};
