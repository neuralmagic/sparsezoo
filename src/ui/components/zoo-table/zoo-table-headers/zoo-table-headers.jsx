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

import TableHead from "@material-ui/core/TableHead";
import TableRow from "@material-ui/core/TableRow";
import TableCell from "@material-ui/core/TableCell";
import Typography from "@material-ui/core/Typography";

import makeStyles from "./zoo-table-headers-styles";

function ZooTableHeaders({ headers, aligns, width }) {
  const useStyles = makeStyles();
  const classes = useStyles();

  if (width) {
    if (typeof width == "string" || typeof width == "number") {
      width = Array(headers.length).fill(width);
    }
  } else {
    width = Array(headers.length).fill(`${100 / headers.length}%`);
  }

  return (
    <TableHead className={classes.root}>
      <TableRow>
        {headers.map((header, headerIndex) => (
          <TableCell
            key={header}
            align={
              typeof aligns === "string"
                ? aligns
                : lodash.get(aligns, headerIndex, "left")
            }
            style={{
              width: width[headerIndex],
            }}
            className={classes.headerContainer}
          >
            <Typography className={classes.header}>{header}</Typography>
          </TableCell>
        ))}
      </TableRow>
    </TableHead>
  );
}

ZooTableHeaders.propTypes = {
  headers: PropTypes.arrayOf(PropTypes.string).isRequired,
  aligns: PropTypes.oneOfType([PropTypes.string, PropTypes.arrayOf(PropTypes.string)]),
  width: PropTypes.oneOfType([
    PropTypes.string,
    PropTypes.number,
    PropTypes.arrayOf(PropTypes.oneOfType([PropTypes.string, PropTypes.number])),
  ]),
};

ZooTableHeaders.defaultProps = {
  headers: [],
  aligns: "left",
  filterColumns: {},
};

export default ZooTableHeaders;
