import { API_ROOT, validateAPIResponseJSON } from "./utils";

export function requestSearchModels(
  domain,
  subdomain,
  token,
  page = 1,
  page_length = 20
) {
  const url = `${API_ROOT}/models/search/${domain}/${subdomain}?page=${page}&page_length=${page_length}`;
  return validateAPIResponseJSON(
    fetch(url, {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
        "nm-token-header": token,
      },
    })
  );
}
