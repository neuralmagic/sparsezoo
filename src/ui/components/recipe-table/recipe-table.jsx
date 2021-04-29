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

import React, { useEffect } from "react";
import { useSelector, useDispatch } from "react-redux";
import PropTypes from "prop-types";

import lodash from "lodash";

import Typography from "@material-ui/core/Typography";

import {
  selectAuthState,
  selectRecipesState,
  searchRecipesThunk,
  selectRecipesTable,
} from "../../store";
import makeStyles from "./recipe-table-styles";
import ZooTable from "../zoo-table";

function RecipeTable({ domain, subdomain, includePagination, includeHeader, queries }) {
  const useStyles = makeStyles();
  const classes = useStyles();
  const dispatch = useDispatch();

  const authState = useSelector(selectAuthState);
  const recipesState = useSelector(selectRecipesState);
  console.log(recipesState);
  const results = useSelector(selectRecipesTable);

  useEffect(() => {
    const status = lodash.get(recipesState.status, `${domain}.${subdomain}`, "idle");
    if (authState.token !== null && status === "idle") {
      dispatch(
        searchRecipesThunk({
          domain,
          subdomain,
          token: authState.token,
          queries,
        })
      );
    }
  }, [authState.token, recipesState.status, dispatch, domain, subdomain, queries]);

  const rows = lodash
    .get(results, `${domain}.${subdomain}.data`, [])
    .map((data) => data.row);
  const status = lodash.get(results, `${domain}.${subdomain}.status`, "idle");
  const loaded = status !== "idle" && status !== "loading";
  return (
    <div className={classes.root}>
      {loaded && includeHeader && (
        <Typography variant="h6">
          {lodash.get(
            results,
            `${domain}.${subdomain}.displayName`,
            `${domain} ${subdomain}`
          )}
        </Typography>
      )}
      <ZooTable
        ariaLabel={`${domain}.${subdomain}`}
        headers={lodash.get(results, `${domain}.${subdomain}.headers`, [])}
        rows={rows}
        status={status}
        error={authState.error || recipesState.error}
        aligns={lodash.get(results, `${domain}.${subdomain}.aligns`, "left")}
        copy={lodash.get(results, `${domain}.${subdomain}.copy`, false)}
        width={lodash.get(results, `${domain}.${subdomain}.width`)}
        includePagination={includePagination}
        loadingMessage="Loading recipes"
      />
    </div>
  );
}

RecipeTable.propTypes = {
  domain: PropTypes.string.isRequired,
  subdomain: PropTypes.string.isRequired,
  queries: PropTypes.object,
  includePagination: PropTypes.bool,
  includeHeader: PropTypes.bool,
};

RecipeTable.defaultProps = {
  includePagination: false,
  includeHeader: false,
  queries: {},
};

export default RecipeTable;
