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

import { makeStyles } from "@material-ui/core/styles";

export default function makeLoaderLayoutStyles() {
  return makeStyles(
    (theme) => {
      return {
        root: {
          width: "100%",
        },
        error: {
          width: "100%",
          display: "flex",
          justifyContent: "center",
          alignItems: "center",
          color: theme.palette.error.main,
        },
        loader: {
          width: "100%",
          display: "flex",
          justifyContent: "center",
          alignItems: "center",
        },
        progressRoot: {
          display: "flex",
          alignItems: "center",
          flexDirection: "column",
        },
        progressContainer: {
          position: "relative",
        },
        progressTextContainer: {
          position: "absolute",
          top: 0,
          left: 0,
          height: "100%",
          width: "100%",
          display: "flex",
          justifyContent: "center",
          alignItems: "center",
        },
      };
    },
    { name: "LoaderLayout" }
  );
}
