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

import { createSlice, createAsyncThunk } from "@reduxjs/toolkit";

import { requestPostAuth } from "../api";
/**
 * Async thunk for making a request to get authentication
 *
 * @type {AsyncThunk<Promise<*>, {}, {}>}
 */
export const authThunk = createAsyncThunk("auth/postAuth", async () => {
  const body = await requestPostAuth();

  return body.token;
});

/**
 * Slice for handling the auth state in the redux store.
 *
 * @type {Slice<{token: str, error: null, status: string}, {}, string>}
 */
const authSlice = createSlice({
  name: "auth",
  initialState: {
    token: null,
    error: null,
    status: "idle",
  },
  reducers: {
    setAuthToken: (state, action) => {
      state.status = "succeeded";
      state.token = action.payload;
    },
  },
  extraReducers: {
    [authThunk.pending]: (state, action) => {
      state.status = "loading";
      state.error = null;
    },
    [authThunk.fulfilled]: (state, action) => {
      state.status = "succeeded";
      state.error = null;
      state.token = action.payload;
    },
    [authThunk.rejected]: (state, action) => {
      state.status = "failed";
      state.error = action.error.message;
    },
  },
});

/***
 * Available actions for auth redux store
 */
// eslint-disable-next-line
export const { setAuthToken } = authSlice.actions;

/**
 * Simple selector to get the current auth state
 *
 * @param state - the redux store state
 * @returns {Reducer<State> | Reducer<{token: str,  error: null, status: string}>}
 */
export const selectAuthState = (state) => state.auth;

export default authSlice.reducer;
