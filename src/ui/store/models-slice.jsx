import { createSlice, createAsyncThunk, createSelector } from "@reduxjs/toolkit";

import _ from "lodash";

import { getModelStub, getFormattedData } from "../utils";
import { requestSearchModels } from "../api";
/**
 * Async thunk for making a request to search for models
 *
 * @type {AsyncThunk<Promise<*>, {domain: string, subdomain: string, token: string}, {}>}
 */
export const searchModelsThunk = createAsyncThunk(
  "models/searchModels",
  async ({ domain, subdomain, token }) => {
    let page = 1;
    const models = [];
    let body;
    do {
      body = await requestSearchModels(domain, subdomain, token, page);
      models.push(...body.models);
      page += 1;
    } while (body.models.length > 0);

    return models;
  }
);

/**
 * Slice for handling the models state in the redux store.
 *
 * @type {Slice<{models: Object, error: null, status: Object}, {}, string>}
 */
const modelsSlice = createSlice({
  name: "models",
  initialState: {
    models: {},
    error: null,
    status: {},
  },
  reducers: {},
  extraReducers: {
    [searchModelsThunk.pending]: (state, action) => {
      const { domain, subdomain } = action.meta.arg;
      _.setWith(state.status, `${domain}.${subdomain}`, "loading", Object);
      state.error = null;
    },
    [searchModelsThunk.fulfilled]: (state, action) => {
      const { domain, subdomain } = action.meta.arg;
      _.setWith(state.status, `${domain}.${subdomain}`, "succeeded", Object);
      state.error = null;

      const models = action.payload.filter(
        (model) => !model.tags.map((tag) => tag.name).includes("demo")
      );
      _.setWith(state.models, `${domain}.${subdomain}`, models, Object);
    },
    [searchModelsThunk.rejected]: (state, action) => {
      const { domain, subdomain } = action.meta.arg;
      _.setWith(state.status, `${domain}.${subdomain}`, "failed", Object);
      state.error = action.error.message;
    },
  },
});

/***
 * Available actions for models redux store
 */
export const { defaultSearchStatus } = modelsSlice.actions;

/**
 * Simple selector to get the current models state
 *
 * @param state - the redux store state
 * @returns {Reducer<State> | Reducer<{models: Object,  error: null, status: Object}>}
 */
export const selectModelsState = (state) => state.models;

const DISPLAY_NAMES = {
  cv: {
    classification: "Image Classification",
    detection: "Object Detection",
  },
};

const FILTERABLE_FIELDS = [
  "architecture",
  "sub_architecture",
  "framework",
  "repo",
  "dataset",
  "training_scheme",
  "sparse_name",
  "sparse_category",
  "sparse_target",
];

const cvModelsToTableData = (domain, subdomain, models, status) => {
  const displayName = _.get(
    DISPLAY_NAMES,
    `${domain}.${subdomain}`,
    `${domain} ${subdomain}`
  );

  const data = models.map((model) => ({
    model,
    row: [
      model.display_name,
      getModelStub(model),
      getFormattedData(model, "training"),
      getFormattedData(model, "inference"),
    ],
  }));

  let filterOptions = FILTERABLE_FIELDS.map((field) => {
    const options = Array.from(
      models.reduce((values, model) => {
        values.add(_.get(model, field));
        return values;
      }, new Set())
    );

    return {
      field,
      options,
    };
  });

  filterOptions = filterOptions.filter(
    (filterOptions) => filterOptions.options.length > 1
  );

  return {
    domain,
    subdomain,
    displayName,
    headers: ["Model Name", "Model Stub", "Training Metric", "Inference Metric"],
    models,
    data,
    filterColumns: {
      "Model Stub": filterOptions,
    },
    status,
    aligns: ["left", "left", "right", "right"],
    width: ["20%", "40%", "20%", "20%"],
  };
};

export const selectModelTable = createSelector([selectModelsState], (modelsState) => {
  const table = {};
  for (let domain in modelsState.status) {
    table[domain] = {};
    for (let subdomain in modelsState.status[domain]) {
      let data;
      if (domain === "cv") {
        data = cvModelsToTableData(
          domain,
          subdomain,
          _.get(modelsState.models, `${domain}.${subdomain}`, []),
          _.get(modelsState.status, `${domain}.${subdomain}`, "idle")
        );
      }

      table[domain][subdomain] = data;
    }
  }
  return table;
});

export default modelsSlice.reducer;
