import ModelsRoot from "./model-table-root";
import ModelsTable from "./model-table";

import { MODEL_TABLE_PATH, MODEL_TABLE_ROOT_PATH } from "./paths";

export function makeContentRoutes() {
  return [
    {
      path: MODEL_TABLE_ROOT_PATH,
      exact: true,
      component: ModelsRoot,
    },
    {
      path: MODEL_TABLE_PATH,
      exact: true,
      component: ModelsTable,
    },
  ];
}
