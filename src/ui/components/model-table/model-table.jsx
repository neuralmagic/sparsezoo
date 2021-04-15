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

import _ from "lodash";

import Typography from "@material-ui/core/Typography";

import {
  selectAuthState,
  selectModelsState,
  searchModelsThunk,
  selectModelTable,
} from "../../store";
import makeStyles from "./model-table-styles";
import ZooTable from "../zoo-table";

function ModelTable({ domain, subdomain, includePagination, includeHeader, queries }) {
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
          queries,
        })
      );
    }
  }, [authState.token, modelsState.status, dispatch, domain, subdomain, queries]);

  const rows = _.get(results, `${domain}.${subdomain}.data`, []).map(
    (data) => data.row
  );
  const status = _.get(results, `${domain}.${subdomain}.status`, "idle");
  const loaded = status !== "idle" && status !== "loading";
  return (
    <div className={classes.root}>
      {loaded && includeHeader && (
        <Typography variant="h6">
          {_.get(
            results,
            `${domain}.${subdomain}.displayName`,
            `${domain} ${subdomain}`
          )}
        </Typography>
      )}
      <ZooTable
        ariaLabel={`${domain}.${subdomain}`}
        headers={_.get(results, `${domain}.${subdomain}.headers`, [])}
        rows={rows}
        status={status}
        aligns={_.get(results, `${domain}.${subdomain}.aligns`, "left")}
        width={_.get(results, `${domain}.${subdomain}.width`)}
        includePagination={includePagination}
      />
    </div>
  );
}

ModelTable.propTypes = {
  domain: PropTypes.string.isRequired,
  subdomain: PropTypes.string.isRequired,
  queries: PropTypes.object,
  includePagination: PropTypes.bool,
  includeHeader: PropTypes.bool,
};

ModelTable.defaultProps = {
  includePagination: false,
  includeHeader: false,
  queries: {},
};

export default ModelTable;
