import { makeStyles } from "@material-ui/core/styles";

export default function makeZooTableStyles() {
  return makeStyles(
    (theme) => ({
      root: {},
      loaderTable: {
        position: "absolute",
        height: "100%",
        width: "100%",
        top: "0",
        zIndex: 200,
      },
      tableContainer: {
        height: 37 + 330,
        position: "relative",
      },
      tableGroup: {
        height: 37 + 330 + 52,
        position: "relative",
        borderStyle: "solid",
        borderWidth: "thin",
        border: theme.palette.divider,
        margin: theme.spacing(1),
      },
      table: {
        height: 37 + 330,
      },
    }),
    { name: "ZooTable" }
  );
}
