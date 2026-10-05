import { buildMetadata, getClient, resetClient, unaryCall } from "./channel.js";
import {
  CredentialServerClient,
  type Credentials,
  type Credential,
} from "./generated/credential_server.js";

const ensureClient = () => getClient(CredentialServerClient);

export function resetCredentialServer(): void {
  resetClient(CredentialServerClient);
}

/**
 * Retrieve credentials for an entity/service pair.
 *
 * @returns Key-value map of credentials
 */
export async function getCredentials(
  entity: string,
  service: string,
): Promise<Record<string, string>> {
  const c = ensureClient();
  const resp = await unaryCall<Credentials>((cb) => c.getCredentials(
      { entity, service },
      buildMetadata(), cb));
  const result: Record<string, string> = {};
  for (const cred of resp.credentials) {
    if (cred.key) result[cred.key] = cred.value ?? "";
  }
  return result;
}

/**
 * Store credentials for an entity/service pair.
 */
export async function setCredentials(
  entity: string,
  service: string,
  credentials: Record<string, string>,
): Promise<string> {
  const c = ensureClient();
  const creds: Credential[] = Object.entries(credentials).map(
    ([key, value]) => ({ key, value }),
  );
  const resp = await unaryCall<{ response?: string | undefined }>((cb) => c.setCredentials(
      { entity, service, credentials: creds },
      buildMetadata(), cb));
  return resp.response ?? "";
}
