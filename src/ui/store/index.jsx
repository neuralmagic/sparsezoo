import { configureStore } from "@reduxjs/toolkit";

import authReducer from "./auth-slice";
import modelsReducer from "./models-slice";

export default configureStore({
  reducer: {
    auth: authReducer,
    models: modelsReducer,
  },
});

export * from "./auth-slice";
export * from "./models-slice";
