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

import React, { useState } from "react";
import PropTypes from "prop-types";

import Tooltip from "@material-ui/core/Tooltip";
import IconButton from "@material-ui/core/IconButton";
import FileCopyIcon from "@material-ui/icons/FileCopy";
import { CopyToClipboard } from "react-copy-to-clipboard";

import makeStyles from "./copy-button-styles";

function CopyButton({ text, iconButtonSize }) {
  const [showCopiedTooltip, setShowCopiedTooltip] = useState(false);
  const useStyles = makeStyles();
  const classes = useStyles();

  return (
    <Tooltip
      open={showCopiedTooltip}
      onClose={() => setShowCopiedTooltip(false)}
      title="Copied to clipboard"
      classNames={classes.root}
    >
      <CopyToClipboard text={text} onCopy={() => setShowCopiedTooltip(true)}>
        <IconButton size={iconButtonSize}>
          <FileCopyIcon />
        </IconButton>
      </CopyToClipboard>
    </Tooltip>
  );
}

CopyButton.propTypes = {
  text: PropTypes.string.isRequired,
  iconButtonSize: PropTypes.oneOf(["small", "medium"]),
};

CopyButton.defaultProps = {
  iconButtonSize: "small",
};

export default CopyButton;
