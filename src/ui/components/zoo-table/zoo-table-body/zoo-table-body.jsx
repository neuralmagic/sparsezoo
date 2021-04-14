import React from "react";
import PropTypes from "prop-types";

import _ from "lodash";

import TableBody from "@material-ui/core/TableBody";
import TableCell from "@material-ui/core/TableCell";
import TableRow from "@material-ui/core/TableRow";

import makeStyles from "./zoo-table-styles";

function ZooTableBody({ rows, aligns }) {
  const useStyles = makeStyles();
  const classes = useStyles();

  return (
    <TableBody className={classes.root}>
      {rows.map((row, index) => (
        <TableRow key={`row-${index}`} className={classes.row}>
          {row.map((column, columnIndex) => (
            <TableCell
              key={`row-${index}-col-${columnIndex}`}
              align={
                typeof aligns === "string" ? aligns : _.get(aligns, columnIndex, "left")
              }
              width={`${100 / row.length}%`}
              size="small"
            >
              {column}
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
};

ZooTableBody.defaultProps = {
  rows: [],
  aligns: "left",
};

export default ZooTableBody;
