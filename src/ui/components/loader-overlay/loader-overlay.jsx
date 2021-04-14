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
import PropTypes from "prop-types";

import makeStyles from "./loader-overlay-styles";
import LoaderLayout from "../loader-layout";
import FadeTransition from "../fade-transition";

const useStyles = makeStyles();

function LoaderOverlay({
  loading,
  status,
  error,
  errorTitle,
  progress,
  loaderSize,
  children,
  loaderChildren,
}) {
  const transTime = 300;
  const classes = useStyles({ transTime });

  if (!loaderSize) {
    loaderSize = 64;
  }

  const showError = !!error;
  const showLoading = loading || status === "idle" || status === "loading";

  return (
    <FadeTransition show={showError || showLoading}>
      <LoaderLayout
        loading={loading}
        status={status}
        error={error}
        errorTitle={errorTitle}
        progress={progress}
        loaderSize={loaderSize}
        rootClass={classes.root}
        loaderClass={classes.loader}
        loaderColor="inherit"
        errorClass={classes.error}
        loaderChildren={loaderChildren}
      >
        {children}
      </LoaderLayout>
    </FadeTransition>
  );
}

LoaderOverlay.propTypes = {
  loading: PropTypes.bool,
  status: PropTypes.string,
  error: PropTypes.string,
  errorTitle: PropTypes.string,
  progress: PropTypes.number,
  rootClass: PropTypes.string,
  loaderClass: PropTypes.string,
  errorClass: PropTypes.string,
  children: PropTypes.node,
  loaderChildren: PropTypes.node,
  loaderSize: PropTypes.number,
};

export default LoaderOverlay;
