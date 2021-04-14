import { makeStyles } from "@material-ui/core/styles";

export default function makeZooTableStyles() {
  return makeStyles(
    (theme) => ({
      root: {},
      header: {
        fontWeight: "bold",
        display: "inline",
      },
      filterButton: {
        padding: 0,
        marginLeft: theme.spacing(1),
        marginRight: theme.spacing(1),
      },
      filterContainer: {
        padding: theme.spacing(1),
      },
    }),
    { name: "ZooTableHeader" }
  );
}
