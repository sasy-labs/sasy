import * as fs from "node:fs";
import * as grpc from "@grpc/grpc-js";
import type { SasyConfig, AuthHook } from "./types.js";

let cachedConfig: SasyConfig | undefined;
let cachedChannel: grpc.Channel | undefined;

/** A generated gRPC client constructor, `new (url, creds) => Client`. */
type ClientCtor<C> = new (
  address: string,
  creds: grpc.ChannelCredentials,
) => C;

/** Per-service stub cache, keyed by client constructor. Cleared by
 * {@link configure} and {@link resetChannel} so a config change rebuilds
 * every stub on next use. */
const clientCache = new Map<ClientCtor<unknown>, unknown>();

const DEFAULT_URL = "localhost:10089";

/** Canonical opt-in env flag: case-insensitive "1"/"true"/"yes"/"on"
 * (mirrors the server's `sasy_common::env_flag`). Anything else, or
 * unset, is false. */
function envFlag(name: string): boolean {
  const v = process.env[name]?.trim().toLowerCase();
  return v === "1" || v === "true" || v === "yes" || v === "on";
}

/** Explicit opt-in to plaintext gRPC via env (SASY_INSECURE=1). */
function envInsecure(): boolean {
  return envFlag("SASY_INSECURE");
}

/** Apply a partial update to the active configuration. On first use, omitted
 * fields come from the environment; later updates preserve configured fields.
 * Explicit undefined clears an optional field, while url/insecure retain their
 * current values when undefined. */
export function configure(config: Partial<SasyConfig>): SasyConfig {
  const previous = getConfig();
  resetChannel();
  cachedConfig = {
    ...previous,
    ...config,
    url: config.url ?? previous.url,
    insecure: config.insecure ?? previous.insecure,
  };
  return cachedConfig;
}

export function getConfig(): SasyConfig {
  if (!cachedConfig) {
    cachedConfig = {
      url: process.env.SASY_URL ?? DEFAULT_URL,
      caPath: process.env.TLS_CA_PATH,
      certPath: process.env.TLS_CERT_PATH,
      keyPath: process.env.TLS_KEY_PATH,
      insecure: envInsecure(),
    };
  }
  return cachedConfig;
}

export function buildCredentials(config: SasyConfig): grpc.ChannelCredentials {
  // Explicit CA (private CA / self-signed; optionally a client cert for mTLS).
  if (config.caPath) {
    const rootCerts = fs.readFileSync(config.caPath);
    const certChain = config.certPath
      ? fs.readFileSync(config.certPath)
      : null;
    const privateKey = config.keyPath
      ? fs.readFileSync(config.keyPath)
      : null;
    return grpc.credentials.createSsl(rootCerts, privateKey, certChain);
  }
  // Explicit opt-in to plaintext — for isolated/internal networks only.
  if (config.insecure) {
    return grpc.credentials.createInsecure();
  }
  // Secure default: verify the server cert against the system trust store.
  // Never silently fall back to plaintext — that would allow MITM of policy
  // decisions and leak credentials / API keys. Set SASY_INSECURE=1 (or pass
  // { insecure: true }) for a trusted plaintext sidecar.
  return grpc.credentials.createSsl();
}

export function getChannel(): grpc.Channel {
  if (cachedChannel) return cachedChannel;
  const config = getConfig();
  const creds = buildCredentials(config);
  cachedChannel = new grpc.Channel(config.url, creds, {});
  return cachedChannel;
}

export function resetChannel(): void {
  clientCache.clear();
  if (cachedChannel) {
    cachedChannel.close();
    cachedChannel = undefined;
  }
}

/**
 * Get (lazily build + cache) the gRPC stub for `Ctor`, using the active
 * config. Shared by every service module so the cache-and-rebuild logic
 * lives in one place and {@link configure} can invalidate all stubs at once.
 */
export function getClient<C>(Ctor: ClientCtor<C>): C {
  const key = Ctor as ClientCtor<unknown>;
  const existing = clientCache.get(key);
  if (existing) return existing as C;
  const config = getConfig();
  const client = new Ctor(config.url, buildCredentials(config));
  clientCache.set(key, client);
  return client;
}

/** Drop the cached stub for `Ctor` so the next call rebuilds it. */
export function resetClient<C>(Ctor: ClientCtor<C>): void {
  clientCache.delete(Ctor as ClientCtor<unknown>);
}

/** Build grpc.Metadata from the configured AuthHook. */
export function buildMetadata(authHook?: AuthHook): grpc.Metadata {
  const metadata = new grpc.Metadata();
  const hook = authHook ?? getConfig().authHook;
  if (hook) {
    for (const [key, value] of hook.getMetadata()) {
      metadata.set(key, value);
    }
  }
  return metadata;
}

/**
 * Promisify a gRPC unary call. Pass a closure that receives the Node
 * callback and invokes the stub method with it:
 *
 *   unaryCall<Resp>((cb) => client.method(req, buildMetadata(), cb))
 */
export function unaryCall<R>(
  invoke: (cb: (err: grpc.ServiceError | null, resp?: R) => void) => unknown,
): Promise<R> {
  return new Promise<R>((resolve, reject) => {
    invoke((err, resp) => {
      if (err) reject(err);
      else resolve(resp as R);
    });
  });
}
