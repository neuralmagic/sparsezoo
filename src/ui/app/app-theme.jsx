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

import { createMuiTheme } from "@material-ui/core/styles";
import { useState } from "react";

/**
 * Creates a hook for using dark mode
 * @returns {boolean}
 */
export function useDarkMode() {
  return useState(false);
}

/**
 * Creates a material ui theme
 * @param {boolean} darkMode whether dark mode is enabled
 * @returns {object}
 */
export default function makeTheme(darkMode) {
  const theme = createMuiTheme({
    palette: {
      type: darkMode ? "dark" : "light",
      primary: { main: "#4652B1", contrastText: "#FFFFFF" },
      secondary: { main: "#ff9900", contrastText: "#FFFFFF" },
      disabled: { main: "#777777" },
      divider: "#000000",
    },
    typography: {
      fontFamily: ["Helvetica Neue", "Roboto", "Open Sans", "sans-serif"].join(","),
    },
  });

  if (darkMode) {
    // make the background color darker
    theme.palette.background.default = "#1D1D1D";
    theme.palette.background.paper = "#303030";

    theme.palette.overlay = "rgba(255, 255, 255, 0.3)";
  } else {
    theme.palette.overlay = "rgba(0, 0, 0, 0.7)";
  }

  return theme;
}

export const referenceDarkTheme = makeTheme(true);
export const referenceLightTheme = makeTheme(false);
