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

import makeStyles from "./fade-transition-styles";
import CSSTransition from "react-transition-group/CSSTransition";

const useStyles = makeStyles();

function FadeTransition({ show, children, className }) {
  const transTime = 300;
  const classes = useStyles({ transTime });

  return (
    <CSSTransition
      in={show}
      timeout={transTime}
      unmountOnExit
      className={className}
      classNames={{
        enter: classes.transitionEnter,
        enterActive: classes.transitionEnterActive,
        exit: classes.transitionExit,
        exitActive: classes.transitionExitActive,
      }}
    >
      <div className={classes.child}>{children}</div>
    </CSSTransition>
  );
}

FadeTransition.propTypes = {
  show: PropTypes.bool,
  children: PropTypes.oneOfType([PropTypes.arrayOf(PropTypes.node), PropTypes.node]),
};

export default FadeTransition;
