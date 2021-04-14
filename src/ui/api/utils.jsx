export const API_ROOT =
  process.env.REACT_APP_SPARSEZOO_API_URL || "https://api.neuralmagic.com";

/***
 * Utility function to validate and format a JSON response
 * from the sparsify APIs.
 *
 * @param {Promise<any>} responsePromise
 * @returns {Promise<any>}
 */
export function validateAPIResponseJSON(responsePromise) {
  return responsePromise
    .then((response) => {
      if (response.data) {
        // axios response
        return Promise.resolve({
          statusOk: response.statusText === "OK",
          status: response.status,
          body: response.data,
        });
      } else {
        // fetch response
        return response.json().then((data) => {
          return {
            statusOk: response.ok,
            status: response.status,
            body: data,
          };
        });
      }
    })
    .then((data) => {
      if (!data.statusOk) {
        return Promise.reject(Error(data.body.error_message));
      }

      return data.body;
    })
    .catch((err) => {
      return Promise.reject(err);
    });
}
