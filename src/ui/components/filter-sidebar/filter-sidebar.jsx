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

import _ from "lodash";

import Chip from "@material-ui/core/Chip";
import Typography from "@material-ui/core/Typography";

import makeStyles from "./filter-sidebar-styles";

function FilterSidebar({
  open,
  filterOptions,
  selectedFilters,
  handleFilter,
  handleClear,
}) {
  const useStyles = makeStyles();
  const classes = useStyles();
  return (
    <div className={classes.root} variant="persistent" anchor="left" open={open}>
      {filterOptions.map(({ field, options }) => (
        <div key={field}>
          <div>
            <Typography className={classes.label}>{field.replace("_", " ")}</Typography>

            {_.get(selectedFilters, field, []).length > 0 && (
              <Typography
                variant="caption"
                className={classes.clearLabel}
                onClick={() => handleClear(field)}
                color="textSecondary"
              >
                Clear All
              </Typography>
            )}
          </div>

          <div>
            {options
              .filter((option) => _.get(selectedFilters, field, []).includes(option))
              .map((option) => (
                <Chip
                  key={option || "N/A"}
                  className={classes.chip}
                  size="small"
                  label={option || "N/A"}
                  onClick={() => handleFilter(field, option)}
                />
              ))}

            {options
              .filter((option) => !_.get(selectedFilters, field, []).includes(option))
              .map((option) => (
                <Chip
                  key={option || "N/A"}
                  className={classes.unselectedChip}
                  variant="outlined"
                  size="small"
                  label={option || "N/A"}
                  onClick={() => handleFilter(field, option)}
                />
              ))}
          </div>
        </div>
      ))}
    </div>
  );
}

export default FilterSidebar;
