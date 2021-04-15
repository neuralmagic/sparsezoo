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

import React, { useState } from "react";
import PropTypes from "prop-types";

import Table from "@material-ui/core/Table";
import TableContainer from "@material-ui/core/TableContainer";
import TablePagination from "@material-ui/core/TablePagination";

import makeStyles from "./zoo-table-styles";
import ZooTableHeaders from "./zoo-table-headers";
import ZooTableBody from "./zoo-table-body";
import LoaderOverlay from "../loader-overlay";

function ZooTable({
  headers,
  rows,
  aligns,
  status,
  ariaLabel,
  paginationOptions,
  width,
}) {
  const useStyles = makeStyles();
  const classes = useStyles();
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(10);

  const handleChangePage = (event, newPage) => {
    setPage(newPage);
  };

  const handleChangeRowsPerPage = (event) => {
    setRowsPerPage(event.target.value);
    setPage(0);
  };

  const currentRows =
    status === "succeeded"
      ? rows.slice(page * rowsPerPage, (page + 1) * rowsPerPage)
      : Array(10).fill(Array(headers.length).fill("-"));
  return (
    <div className={classes.root}>
      {status !== "succeeded" && (
        <div className={classes.tableGroup}>
          <LoaderOverlay
            status={status}
            loaderSize={100}
            loaderChildren={
              <div className={classes.loaderTable}>
                <Table size="small" aria-label={ariaLabel}>
                  <ZooTableHeaders headers={headers} aligns={aligns} width={width} />
                  <ZooTableBody rows={currentRows} aligns={aligns} width={width} />
                </Table>
              </div>
            }
          />
        </div>
      )}

      {status === "succeeded" && (
        <div className={classes.tableGroup}>
          <TableContainer className={classes.tableContainer}>
            <div className={classes.table}>
              <Table stickyHeader size="small" aria-label={ariaLabel}>
                <ZooTableHeaders headers={headers} aligns={aligns} width={width} />
                <ZooTableBody rows={currentRows} aligns={aligns} width={width} />
              </Table>
            </div>
          </TableContainer>
          <div className={classes.pagination}>
            <TablePagination
              rowsPerPageOptions={paginationOptions}
              count={rows.length}
              component="div"
              rowsPerPage={rowsPerPage}
              page={page}
              align="right"
              onChangePage={handleChangePage}
              onChangeRowsPerPage={handleChangeRowsPerPage}
            />
          </div>
        </div>
      )}
    </div>
  );
}

ZooTable.propTypes = {
  ariaLabel: PropTypes.string,
  headers: PropTypes.arrayOf(PropTypes.string).isRequired,
  rows: PropTypes.arrayOf(PropTypes.arrayOf(PropTypes.any)).isRequired,
  aligns: PropTypes.oneOfType([PropTypes.string, PropTypes.arrayOf(PropTypes.string)]),
  status: PropTypes.string.isRequired,
  paginationOptions: PropTypes.arrayOf(PropTypes.number),
  width: PropTypes.oneOfType([
    PropTypes.string,
    PropTypes.number,
    PropTypes.arrayOf(PropTypes.oneOfType([PropTypes.string, PropTypes.number])),
  ]),
};

ZooTable.defaultProps = {
  ariaLabel: "ZooTable",
  headers: [],
  rows: [],
  aligns: "left",
  status: "idle",
  paginationOptions: [10, 25, 100],
};

export default ZooTable;
