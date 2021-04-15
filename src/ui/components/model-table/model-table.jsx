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

import React, { useState, useEffect } from "react";
import { useSelector, useDispatch } from "react-redux";

import _ from "lodash";

import Grid from "@material-ui/core/Grid";

import Toolbar from "@material-ui/core/Toolbar";
import IconButton from "@material-ui/core/IconButton";
import MenuIcon from "@material-ui/icons/Menu";
import Typography from "@material-ui/core/Typography";

import {
  selectAuthState,
  selectModelsState,
  searchModelsThunk,
  selectModelTable,
} from "../../store";
import makeStyles from "./model-table-styles";
import ZooTable from "../zoo-table";
import FilterSidebar from "../filter-sidebar";

function ModelTable({ domain, subdomain }) {
  const [filterOpen, setFilterOpen] = useState(false);
  const [selectedFilters, setSelectedFilters] = useState({});

  const useStyles = makeStyles();
  const classes = useStyles();
  const dispatch = useDispatch();

  const authState = useSelector(selectAuthState);
  const modelsState = useSelector(selectModelsState);

  const results = useSelector(selectModelTable);

  useEffect(() => {
    const status = _.get(modelsState.status, `${domain}.${subdomain}`, "idle");
    if (authState.token !== null && status === "idle") {
      dispatch(
        searchModelsThunk({
          domain,
          subdomain,
          token: authState.token,
        })
      );
    }
  }, [authState.token, modelsState.status, dispatch, domain, subdomain]);

  const handleFilter = (field, value) => {
    let selected = _.get(selectedFilters, field, []);
    if (!selected.includes(value)) {
      selected.push(value);
    } else {
      selected = selected.filter((current) => value !== current);
    }
    setSelectedFilters({ ...selectedFilters, [field]: selected });
  };

  const handleClearFilter = (field) => {
    setSelectedFilters({ ...selectedFilters, [field]: [] });
  };

  const handleToggleFilter = () => {
    setFilterOpen(!filterOpen);
  };

  const modelData = _.get(results, `${domain}.${subdomain}.data`, []).filter(
    ({ model }) => {
      for (let field in selectedFilters) {
        const selected = _.get(selectedFilters, field, []);
        if (selected.length > 0 && !selected.includes(model[field])) {
          return false;
        }
      }
      return true;
    }
  );
  const rows = modelData.map((data) => data.row);
  const status = _.get(results, `${domain}.${subdomain}.status`, "idle");

  let filterOptions = _.get(results, `${domain}.${subdomain}.filterOptions`, []);

  return (
    <Grid container className={classes.root}>
      {filterOpen && (
        <Grid item xs={3} className={`${classes.gridItem} ${classes.toolbar}`}>
          <FilterSidebar
            open={filterOpen}
            filterOptions={filterOptions}
            selectedFilters={selectedFilters}
            handleFilter={handleFilter}
            handleClear={handleClearFilter}
          />
        </Grid>
      )}
      <Grid item xs={filterOpen ? 9 : 12} className={classes.gridItem}>
        <Toolbar disableGutters className={classes.toollbar}>
          <IconButton
            color="inherit"
            aria-label="open drawer"
            edge="start"
            onClick={handleToggleFilter}
            disabled={status !== "succeeded"}
          >
            <MenuIcon />
          </IconButton>
          <Typography>
            {_.get(results, `${domain}.${subdomain}.displayName`, "")}
          </Typography>
        </Toolbar>
        <ZooTable
          ariaLabel={`${domain}.${subdomain}`}
          headers={_.get(results, `${domain}.${subdomain}.headers`, [])}
          rows={rows}
          status={status}
          aligns={_.get(results, `${domain}.${subdomain}.aligns`, "left")}
          width={_.get(results, `${domain}.${subdomain}.width`)}
        />
      </Grid>
    </Grid>
  );
}

export default ModelTable;
