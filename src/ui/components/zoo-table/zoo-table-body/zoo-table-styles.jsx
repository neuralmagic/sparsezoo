import { makeStyles } from "@material-ui/core/styles";

export default function makeZooTableStyles() {
  return makeStyles(
    (theme) => ({
      root: {},
      row: {
        "&:nth-of-type(odd)": {
          backgroundColor: theme.palette.action.hover,
        },
      },
    }),
    { name: "ZooTableBody" }
  );
}
