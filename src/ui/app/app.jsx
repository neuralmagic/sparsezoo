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

import React, { useEffect } from "react";
import { Route } from "react-router-dom";
import { useDispatch, useSelector } from "react-redux";
import { ThemeProvider } from "@material-ui/core/styles";
import Cookies from "universal-cookie";
import jwt_decode from "jwt-decode";

import makeTheme, { useDarkMode } from "./app-theme";
import { authThunk, selectAuthState, setAuthToken } from "../store";
import { makeContentRoutes } from "./../routes";
import makeStyles from "./app-styles";

function App() {
  const useStyles = makeStyles();
  const classes = useStyles();
  const dispatch = useDispatch();

  const darkMode = useDarkMode()[0];
  const theme = makeTheme(darkMode);

  const authState = useSelector(selectAuthState);

  useEffect(() => {
    if (authState.status === "idle") {
      const cookies = new Cookies();
      const authToken = cookies.get("NM_AUTH_TOKEN");

      if (authToken) {
        const decodedToken = jwt_decode(authToken);
        if (decodedToken.exp * 1000 < new Date().getTime()) {
          dispatch(authThunk());
        }
        dispatch(setAuthToken(authToken));
      } else {
        dispatch(authThunk());
      }
    } else if (authState.status === "succeeded") {
      const cookies = new Cookies();

      const authToken = cookies.get("NM_AUTH_TOKEN");

      if (authToken !== authState.token) {
        cookies.set("NM_AUTH_TOKEN", authState.token);
      }
    }
  }, [authState, dispatch]);

  return (
    <ThemeProvider theme={theme}>
      <div className={classes.root}>
        {makeContentRoutes().map((route) => (
          <Route
            key={route.path}
            path={route.path}
            component={route.component}
            exact={route.exact}
          />
        ))}
      </div>
    </ThemeProvider>
  );
}

export default App;
