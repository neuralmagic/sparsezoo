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

import { createSlice, createAsyncThunk, createSelector } from "@reduxjs/toolkit";

import _ from "lodash";

import { getModelStub, getFormattedData } from "../utils";
import { requestSearchModels } from "../api";

const SEARCH_MODELS_PREFIX = "models/searchModels";
const PARTIAL_SEARCH_MODELS_TYPE = `${SEARCH_MODELS_PREFIX}/partial`;
/**
 * Async thunk for making a request to search for models
 *
 * @type {AsyncThunk<Promise<*>, {domain: string, subdomain: string, token: string, queries: {}}, {}>}
 */
export const searchModelsThunk = createAsyncThunk(
  SEARCH_MODELS_PREFIX,
  async ({ domain, subdomain, token, queries }, thunkApi) => {
    let page = 1;
    const models = [];
    let body;
    do {
      body = await requestSearchModels({ domain, subdomain, token, page, queries });
      models.push(...body.models);
      page += 1;
      thunkApi.dispatch({
        type: PARTIAL_SEARCH_MODELS_TYPE,
        payload: {
          domain,
          subdomain,
          models,
        },
      });
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
      _.setWith(state.status, `${domain}.${subdomain}`, "loading", {});
      state.error = null;
    },
    [searchModelsThunk.fulfilled]: (state, action) => {
      const { domain, subdomain } = action.meta.arg;
      _.setWith(state.status, `${domain}.${subdomain}`, "succeeded", {});
      state.error = null;

      const models = action.payload.filter(
        (model) => !model.tags.map((tag) => tag.name).includes("demo")
      );
      _.setWith(state.models, `${domain}.${subdomain}`, models, {});
    },
    [searchModelsThunk.rejected]: (state, action) => {
      const { domain, subdomain } = action.meta.arg;
      _.setWith(state.status, `${domain}.${subdomain}`, "failed", {});
      state.error = action.error.message;
    },
    [PARTIAL_SEARCH_MODELS_TYPE]: (state, action) => {
      let { domain, subdomain, models } = action.payload;
      _.setWith(state.status, `${domain}.${subdomain}`, "partial", {});
      state.error = null;

      models = models.filter(
        (model) => !model.tags.map((tag) => tag.name).includes("demo")
      );
      _.setWith(state.models, `${domain}.${subdomain}`, models, {});
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

const visionModelsToTableData = (domain, subdomain, models, status) => {
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

  let frequency = models.reduce((previous, model) => {
    filterOptions.forEach(({ field }) => {
      let frequencyForField = _.get(previous, `${field}`, {});
      let value = 1 + _.get(previous, `${field}.${model[field]}`, 0);
      _.setWith(
        previous,
        `${field}`,
        { ...frequencyForField, [model[field]]: value },
        {}
      );
    });
    return previous;
  }, {});

  filterOptions.forEach((filterOption) => {
    filterOption.options
      .sort((a, b) => {
        let aFreq = frequency[filterOption.field][a];
        let bFreq = frequency[filterOption.field][b];
        if (aFreq === bFreq) {
          return a - b;
        } else {
          return aFreq - bFreq;
        }
      })
      .reverse();
  });

  return {
    domain,
    subdomain,
    displayName,
    headers: ["Model Name", "Model Stub", "Training Metric", "Inference Metric"],
    models,
    data,
    filterOptions,
    status,
    aligns: "left",
  };
};

export const selectModelTable = createSelector([selectModelsState], (modelsState) => {
  const table = {};
  for (let domain in modelsState.status) {
    table[domain] = {};
    for (let subdomain in modelsState.status[domain]) {
      let data;
      if (domain === "cv") {
        data = visionModelsToTableData(
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
