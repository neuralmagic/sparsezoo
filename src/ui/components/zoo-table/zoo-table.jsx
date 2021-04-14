import React, { useState } from "react";
import PropTypes from "prop-types";

import Table from "@material-ui/core/Table";
import TableContainer from "@material-ui/core/TableContainer";
import TablePagination from "@material-ui/core/TablePagination";

import makeStyles from "./zoo-table-styles";
import ZooTableHeaders from "./zoo-table-headers";
import ZooTableBody from "./zoo-table-body";
import LoaderOverlay from "../loader-overlay";

function LoadingZooTable({ headers, aligns }) {
  const useStyles = makeStyles();
  const classes = useStyles();
  return (
    <div className={classes.loaderTable}>
      <Table size="small">
        <ZooTableHeaders headers={headers} aligns={aligns} />
        <ZooTableBody
          rows={Array(10).fill(Array(headers.length).fill("-"))}
          aligns={aligns}
        />
      </Table>
    </div>
  );
}

function ZooTable({
  headers,
  rows,
  aligns,
  status,
  ariaLabel,
  paginationOptions,
  width,
  filterColumns,
  onFilter,
}) {
  const useStyles = makeStyles();
  const classes = useStyles();
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(10);

  const handleChangePage = (event, newPage) => {
    setPage(newPage);
  };

  const handleChangeRowsPerPage = (event) => {
    console.log(event.target.value);
    setRowsPerPage(event.target.value);
    setPage(0);
  };

  return (
    <div className={classes.root}>
      {status !== "succeeded" && (
        <div className={classes.tableGroup}>
          <LoaderOverlay
            status={status}
            loaderSize={100}
            loaderChildren={<LoadingZooTable headers={headers} aligns={aligns} />}
          />
        </div>
      )}

      {status === "succeeded" && (
        <div className={classes.tableGroup}>
          <TableContainer className={classes.tableContainer}>
            <div className={classes.table}>
              <Table stickyHeader size="small" aria-label={ariaLabel}>
                <ZooTableHeaders
                  headers={headers}
                  aligns={aligns}
                  width={width}
                  filterColumns={filterColumns}
                  onFilter={onFilter}
                />
                <ZooTableBody
                  rows={rows.slice(page * rowsPerPage, (page + 1) * rowsPerPage)}
                  aligns={aligns}
                  width={width}
                />
              </Table>
            </div>
          </TableContainer>
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
