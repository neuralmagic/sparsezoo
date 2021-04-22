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

export default function makeZooTableStyles() {
  return makeStyles(
    (theme) => ({
      root: {},
      loaderText: {
        color: theme.palette.info.contrastText,
      },
      pagination: {
        borderStyle: "solid",
        borderWidth: "0.5px 0 0 0",
        borderColor: theme.palette.divider,
      },
      paginatedTableContainer: {
        maxHeight: 37 + 330,
        position: "relative",
      },
      tableContainer: {
        position: "relative",
      },
      tableGroup: {
        position: "relative",
        borderStyle: "solid",
        borderWidth: "thin",
        border: theme.palette.divider,
        borderBottomWidth: 0,
      },
      paginatedTableGroup: {
        maxHeight: 37 + 330 + 52,
        position: "relative",
        borderStyle: "solid",
        borderWidth: "thin",
        border: theme.palette.divider,
      },
      table: {},
      paginatedTable: {
        maxHeight: 37 + 330,
        borderBottomWidth: 0,
      },
    }),
    { name: "ZooTable" }
  );
}