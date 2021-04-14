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
  reducers: {},
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
export const {} = authSlice.actions;

/**
 * Simple selector to get the current auth state
 *
 * @param state - the redux store state
 * @returns {Reducer<State> | Reducer<{token: str,  error: null, status: string}>}
 */
export const selectAuthState = (state) => state.auth;

export default authSlice.reducer;
