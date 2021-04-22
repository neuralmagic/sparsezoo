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

import React from "react";
import PropTypes from "prop-types";

import lodash from "lodash";

import TableBody from "@material-ui/core/TableBody";
import TableCell from "@material-ui/core/TableCell";
import TableRow from "@material-ui/core/TableRow";
import Typography from "@material-ui/core/Typography";

import makeStyles from "./zoo-table-styles";
import CopyButton from "../../copy-button";

function ZooTableBody({ rows, aligns, copy, width }) {
  const rowSize = lodash.get(rows, "0", 1);
  if (typeof copy === "boolean") {
    copy = Array(rowSize).fill(copy);
  }

  if (width) {
    if (typeof width == "string" || typeof width == "number") {
      width = Array(rowSize).fill(width);
    }
  } else {
    width = Array(rowSize).fill(`${100 / rowSize}%`);
  }

  const useStyles = makeStyles();
  const classes = useStyles();

  return (
    <TableBody className={classes.root}>
      {rows.map((row, index) => (
        <TableRow key={`row-${index}`}>
          {row.map((column, columnIndex) => (
            <TableCell
              key={`row-${index}-col-${columnIndex}`}
              align={
                typeof aligns === "string"
                  ? aligns
                  : lodash.get(aligns, columnIndex, "left")
              }
              className={classes.row}
              width={width}
              size="small"
            >
              <Typography>{column}</Typography>
              {copy[columnIndex] && <CopyButton text={column} />}
            </TableCell>
          ))}
        </TableRow>
      ))}
    </TableBody>
  );
}

ZooTableBody.propTypes = {
  rows: PropTypes.arrayOf(PropTypes.arrayOf(PropTypes.any)).isRequired,
  aligns: PropTypes.oneOfType([PropTypes.string, PropTypes.arrayOf(PropTypes.string)]),
  width: PropTypes.oneOfType([
    PropTypes.string,
    PropTypes.number,
    PropTypes.arrayOf(PropTypes.oneOfType([PropTypes.string, PropTypes.number])),
  ]),
  copy: PropTypes.oneOfType([PropTypes.bool, PropTypes.arrayOf(PropTypes.bool)]),
};

ZooTableBody.defaultProps = {
  rows: [],
  aligns: "left",
  copy: false,
};

export default ZooTableBody;
