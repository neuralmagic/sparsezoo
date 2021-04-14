import { API_ROOT, validateAPIResponseJSON } from "./utils";

const PUBLIC_AUTH_TYPE = "public";

export function requestPostAuth() {
  const url = `${API_ROOT}/auth`;
  return validateAPIResponseJSON(
    fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ authentication_type: PUBLIC_AUTH_TYPE }),
    })
  );
}
