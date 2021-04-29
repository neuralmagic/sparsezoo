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

import lodash from "lodash";

import { getRecipeStub, getFormattedData } from "../utils";
import { requestSearchRecipes } from "../api";

const SEARCH_RECIPES_PREFIX = "recipes/searchRecipes";
const PARTIAL_SEARCH_RECIPES_TYPE = `${SEARCH_RECIPES_PREFIX}/partial`;
/**
 * Async thunk for making a request to search for recipes
 *
 * @type {AsyncThunk<Promise<*>, {domain: string, subdomain: string, token: string, queries: {}}, {}>}
 */
export const searchRecipesThunk = createAsyncThunk(
  SEARCH_RECIPES_PREFIX,
  async ({ domain, subdomain, token, queries }, thunkApi) => {
    let page = 1;
    let recipes = [];
    let body;
    do {
      body = await requestSearchRecipes({ domain, subdomain, token, page, queries });
      if (body.recipes.length > 0) {
        recipes = [...recipes, ...body.recipes];
        page += 1;
        thunkApi.dispatch({
          type: PARTIAL_SEARCH_RECIPES_TYPE,
          payload: {
            domain,
            subdomain,
            recipes,
          },
        });
      }
    } while (body.recipes.length > 0);
    return recipes;
  }
);

/**
 * Slice for handling the recipes state in the redux store.
 *
 * @type {Slice<{recipes: Object, error: null, status: Object}, {}, string>}
 */
const recipesSlice = createSlice({
  name: "recipes",
  initialState: {
    recipes: {},
    error: null,
    status: {},
  },
  reducers: {},
  extraReducers: {
    [searchRecipesThunk.pending]: (state, action) => {
      const { domain, subdomain } = action.meta.arg;
      lodash.setWith(state.status, `${domain}.${subdomain}`, "loading", {});
      state.error = null;
    },
    [searchRecipesThunk.fulfilled]: (state, action) => {
      const { domain, subdomain } = action.meta.arg;
      lodash.setWith(state.status, `${domain}.${subdomain}`, "succeeded", {});
      state.error = null;

      const recipes = action.payload;
      lodash.setWith(state.recipes, `${domain}.${subdomain}`, recipes, {});
    },
    [searchRecipesThunk.rejected]: (state, action) => {
      const { domain, subdomain } = action.meta.arg;
      lodash.setWith(state.status, `${domain}.${subdomain}`, "failed", {});
      state.error = action.error.message;
    },
    [PARTIAL_SEARCH_RECIPES_TYPE]: (state, action) => {
      let { domain, subdomain, recipes } = action.payload;
      lodash.setWith(state.status, `${domain}.${subdomain}`, "partial", {});
      state.error = null;

      lodash.setWith(state.recipes, `${domain}.${subdomain}`, recipes, {});
    },
  },
});

/***
 * Available actions for recipes redux store
 */
export const { defaultSearchStatus } = recipesSlice.actions;

/**
 * Simple selector to get the current recipes state
 *
 * @param state - the redux store state
 * @returns {Reducer<State> | Reducer<{recipes: Object,  error: null, status: Object}>}
 */
export const selectRecipesState = (state) => state.recipes;

const DISPLAY_NAMES = {
  cv: {
    classification: "Image Classification",
    detection: "Object Detection",
  },
};

/**
 * Formats vision recipes to a table data format
 *
 * @param {string} domain domain of the recipes
 * @param {string} subdomain subdomain of the recipes
 * @param {{
 *  files: { checkpoint: boolean, file_type: string, file_size: number}[],
 *  results: { recorded_value: number, recorded_units: string, result_category: string  }
 * }[]} recipes the recipes of specified domain/subdomain
 * @param {string} status status of loading recipes of specified domain/subdomain
 */
const visionRecipesToTableData = (domain, subdomain, recipes, status) => {
  const displayName = lodash.get(
    DISPLAY_NAMES,
    `${domain}.${subdomain}`,
    `${domain} ${subdomain}`
  );

  const data = recipes.map((recipe) => {
    const file_size = `${(recipe.file_size / 1024).toFixed(2)} KB`;
    return {
      recipe,
      row: [
        recipe.model.display_name,
        getRecipeStub(recipe),
        file_size,
        recipe.downloads,
        getFormattedData(recipe, "training"),
        getFormattedData(recipe, "inference"),
      ],
    };
  });

  return {
    domain,
    subdomain,
    displayName,
    headers: [
      "Model Name",
      "Recipe Stub",
      "Content Size",
      "Downloads",
      "Training Metric",
      "Inference Metric",
    ],
    recipes,
    data,
    status,
    aligns: "left",
    copy: [false, true, false, false, false],
  };
};

/**
 * Selector for recipe table for the current loaded data
 */
export const selectRecipesTable = createSelector(
  [selectRecipesState],
  (recipesState) => {
    const table = {};
    for (let domain in recipesState.status) {
      table[domain] = {};
      for (let subdomain in recipesState.status[domain]) {
        let data;
        if (domain === "cv") {
          data = visionRecipesToTableData(
            domain,
            subdomain,
            lodash.get(recipesState.recipes, `${domain}.${subdomain}`, []),
            lodash.get(recipesState.status, `${domain}.${subdomain}`, "idle")
          );
        }

        table[domain][subdomain] = data;
      }
    }
    return table;
  }
);

export default recipesSlice.reducer;
