import React, { useEffect } from "react";
import { Route } from "react-router-dom";
import { useDispatch, useSelector } from "react-redux";
import { ThemeProvider } from "@material-ui/core/styles";

import makeTheme, { useDarkMode } from "./app-theme";
import { authThunk, selectAuthState } from "../store";
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
      dispatch(authThunk());
    }
  }, [authState.status, dispatch]);
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
