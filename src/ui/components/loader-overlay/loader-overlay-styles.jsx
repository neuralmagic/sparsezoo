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

export default function makeLoaderOverlayStyles(transparent) {
  return makeStyles(
    (theme) => {
      return {
        root: {
          position: "absolute",
          width: "100%",
          height: "100%",
          top: 0,
          left: 0,
          backgroundColor: transparent ? "transparent" : theme.palette.overlay,
          color: theme.palette.primary.light,
          zIndex: 1200,
        },
        loader: {
          width: "100%",
          height: "100%",
        },
        error: {
          width: "80%",
          height: "100%",
          paddingLeft: "10%",
          paddingRight: "10%",
        },
        transitionEnter: {
          opacity: 0,
        },
        transitionEnterActive: {
          opacity: 1,
          transition: ({ transTime }) => `opacity ${transTime}ms`,
        },
        transitionExit: {
          opacity: 1,
        },
        transitionExitActive: {
          opacity: 0,
          transition: ({ transTime }) => `opacity ${transTime}ms`,
        },
      };
    },
    { name: "LoaderOverlay" }
  );
}
