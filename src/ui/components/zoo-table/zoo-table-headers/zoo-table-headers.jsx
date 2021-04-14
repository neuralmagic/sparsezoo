import React, { useState } from "react";
import PropTypes from "prop-types";

import _ from "lodash";

import Popover from "@material-ui/core/Popover";

import TableHead from "@material-ui/core/TableHead";
import TableRow from "@material-ui/core/TableRow";
import TableCell from "@material-ui/core/TableCell";
import Typography from "@material-ui/core/Typography";
import IconButton from "@material-ui/core/IconButton";
import FilterListIcon from "@material-ui/icons/FilterList";

import Radio from "@material-ui/core/Radio";
import RadioGroup from "@material-ui/core/RadioGroup";
import FormControlLabel from "@material-ui/core/FormControlLabel";
import FormControl from "@material-ui/core/FormControl";
import FormLabel from "@material-ui/core/FormLabel";

import makeStyles from "./zoo-table-headers-styles";

function ZooTableHeaders({ headers, aligns, width, filterColumns, onFilter }) {
  const [filterOpen, setFilterOpen] = useState(false);
  const [filterAnchor, setFilterAnchor] = useState(null);
  const [filterOptions, setFilterOptions] = useState([]);
  const [selectedFilters, setSelectedFilters] = useState({});

  const useStyles = makeStyles();
  const classes = useStyles();

  const handeOpenFilter = (event, options) => {
    setFilterAnchor(event.currentTarget);
    setFilterOpen(true);
    setFilterOptions(options);
  };

  const handleCloseFilter = () => {
    setFilterOpen(false);
    setFilterAnchor(null);
    setFilterOptions([]);
  };

  const handleFilter = (field, value) => {
    setSelectedFilters({ ...selectedFilters, [field]: value });
    onFilter(field, value);
  };

  if (width) {
    if (typeof width == "string" || typeof width == "number") {
      width = Array(headers.length).fill(width);
    }
  } else {
    width = Array(headers.length).fill(`${100 / headers.length}%`);
  }

  console.log(filterOptions);

  return (
    <TableHead className={classes.root}>
      <TableRow>
        {headers.map((header, headerIndex) => (
          <TableCell
            key={header}
            align={
              typeof aligns === "string" ? aligns : _.get(aligns, headerIndex, "left")
            }
            width={width[headerIndex]}
          >
            <Typography className={classes.header}>{header}</Typography>
            {header in filterColumns && (
              <IconButton
                className={classes.filterButton}
                onClick={(event) =>
                  handeOpenFilter(event, _.get(filterColumns, header, []))
                }
              >
                <FilterListIcon />
              </IconButton>
            )}
          </TableCell>
        ))}
      </TableRow>
      <Popover
        open={filterOpen}
        anchorEl={filterAnchor}
        onClose={handleCloseFilter}
        anchorOrigin={{
          vertical: "bottom",
          horizontal: "center",
        }}
        transformOrigin={{
          vertical: "top",
          horizontal: "center",
        }}
      >
        <div className={classes.filterContainer}>
          {filterOptions.map(({ field, options }) => (
            <FormControl component="fieldset" key={field}>
              <FormLabel component="legend">{field}</FormLabel>
              <RadioGroup
                aria-label={field}
                name={field}
                value={_.get(selectedFilters, field, "all")}
                onChange={(event, value) => handleFilter(field, value)}
              >
                <FormControlLabel
                  value={"all"}
                  control={<Radio color="default" />}
                  label={"All"}
                />
                {options.map((option) => (
                  <FormControlLabel
                    key={String(option)}
                    value={option || ""}
                    control={<Radio color="default" />}
                    label={option ? option : "None"}
                  />
                ))}
              </RadioGroup>
            </FormControl>
          ))}
        </div>
      </Popover>
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
