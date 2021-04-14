import React, { useEffect, useState } from "react";
import { useSelector, useDispatch } from "react-redux";
import Typography from "@material-ui/core/Typography";

import _ from "lodash";

import {
  selectAuthState,
  selectModelsState,
  searchModelsThunk,
  selectModelTable,
} from "../../store";
import makeStyles from "./model-table-root-styles";
import ZooTable from "../../components/zoo-table";

const DOMAINS_INFO = [
  {
    domain: "cv",
    subdomain: "classification",
  },
  {
    domain: "cv",
    subdomain: "detection",
  },
];

function ModelTableRoot() {
  const [selectedFilters, setSelectedFilters] = useState({});

  const useStyles = makeStyles();

  const dispatch = useDispatch();
  const classes = useStyles();

  const authState = useSelector(selectAuthState);
  const modelsState = useSelector(selectModelsState);

  const results = useSelector(selectModelTable);

  useEffect(() => {
    DOMAINS_INFO.forEach(({ domain, subdomain }) => {
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
    });
  }, [authState.token, modelsState.status, dispatch]);

  const handleFilter = (domain, subdomain, field, value) => {
    let domainFilters = _.get(selectedFilters, domain, {});
    let subdomainFilters = _.get(domainFilters, subdomain, {});
    subdomainFilters = { ...subdomainFilters, [field]: value };
    domainFilters = { ...domainFilters, [subdomain]: subdomainFilters };
    setSelectedFilters({ ...selectedFilters, [domain]: domainFilters });
  };

  const rows = DOMAINS_INFO.map(({ domain, subdomain }) => {
    const modelData = _.get(results, `${domain}.${subdomain}.data`, []).filter(
      ({ model }) => {
        const selectedFiltersByDomain = _.get(
          selectedFilters,
          `${domain}.${subdomain}`,
          {}
        );

        for (let field in selectedFiltersByDomain) {
          if (
            selectedFiltersByDomain[field] !== "all" &&
            model[field] !== selectedFiltersByDomain[field] &&
            !(model[field] === null && selectedFilters[field] === "")
          ) {
            return false;
          }
        }
        return true;
      }
    );

    return modelData.map((data) => data.row);
  });

  return (
    <div className={classes.root}>
      {DOMAINS_INFO.map(({ domain, subdomain }, index) => (
        <div key={`${domain}.${subdomain}`}>
          <Typography variant="h4" gutterBottom>
            {_.get(results, `${domain}.${subdomain}.displayName`) ||
              "Loading Header..."}
          </Typography>
          <ZooTable
            ariaLabel={`${domain}.${subdomain}`}
            headers={_.get(results, `${domain}.${subdomain}.headers`, [])}
            rows={rows[index]}
            filterColumns={_.get(results, `${domain}.${subdomain}.filterColumns`, {})}
            onFilter={(field, value) => {
              handleFilter(domain, subdomain, field, value);
            }}
            status={_.get(results, `${domain}.${subdomain}.status`, "idle")}
            aligns={_.get(results, `${domain}.${subdomain}.aligns`, "left")}
            width={_.get(results, `${domain}.${subdomain}.width`)}
          />
        </div>
      ))}
    </div>
  );
}

export default ModelTableRoot;
