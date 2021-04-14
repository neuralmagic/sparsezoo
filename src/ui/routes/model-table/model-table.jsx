import React, { useState, useEffect } from "react";
import { useSelector, useDispatch } from "react-redux";

import _ from "lodash";

import {
  selectAuthState,
  selectModelsState,
  searchModelsThunk,
  selectModelTable,
} from "../../store";
import makeStyles from "./model-table-styles";
import ZooTable from "../../components/zoo-table";

function ModelTable(props) {
  const { domain, subdomain } = props.match.params;
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
    setSelectedFilters({ ...selectedFilters, [field]: value });
  };

  const modelData = _.get(results, `${domain}.${subdomain}.data`, []).filter(
    ({ model }) => {
      for (let field in selectedFilters) {
        if (
          selectedFilters[field] !== "all" &&
          model[field] !== selectedFilters[field] &&
          !(model[field] === null && selectedFilters[field] === "")
        ) {
          return false;
        }
      }
      return true;
    }
  );
  const rows = modelData.map((data) => data.row);

  return (
    <div className={classes.root}>
      <ZooTable
        ariaLabel={`${domain}.${subdomain}`}
        headers={_.get(results, `${domain}.${subdomain}.headers`, [])}
        rows={rows}
        filterColumns={_.get(results, `${domain}.${subdomain}.filterColumns`, {})}
        onFilter={handleFilter}
        status={_.get(results, `${domain}.${subdomain}.status`, "idle")}
        aligns={_.get(results, `${domain}.${subdomain}.aligns`, "left")}
        width={_.get(results, `${domain}.${subdomain}.width`)}
      />
    </div>
  );
}

export default ModelTable;
