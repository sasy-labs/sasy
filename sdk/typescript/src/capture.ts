/** Narrow credential hygiene for telemetry copies. This is not a secret-shape
 * scanner and must never be applied to actual HTTP/auth/policy inputs. */
export const CAPTURE_REDACTED = "[redacted]";
const QUERY_NAMES = new Set([
  "key", "api_key", "apikey", "access_token", "token", "auth_token", "id_token",
  "refresh_token", "client_secret", "password", "passwd", "secret", "sig",
  "signature", "x-amz-signature", "x-goog-signature",
]);
const HEADER_NAMES = new Set([
  "authorization", "proxy-authorization", "x-api-key", "api-key", "x-goog-api-key",
  "x-auth-token", "cookie", "set-cookie",
]);
function headerKey(value: string): boolean {
  let name = value.toLowerCase();
  for (const prefix of ["http.request.header.", "http.response.header."])
    if (name.startsWith(prefix)) { name = name.slice(prefix.length).replaceAll("_", "-"); break; }
  return HEADER_NAMES.has(name);
}

/** Find a JSON array boundary without treating quoted brackets as delimiters. */
function arrayEnd(text: string, start: number, budget: { remaining: number }): number | undefined {
  let depth = 0, quoted = false, escaped = false;
  for (let i = start; i < text.length; i++) {
    if (--budget.remaining < 0) throw new Error("telemetry capture JSON nesting/work limit exceeded");
    const c = text[i];
    if (quoted) {
      if (escaped) escaped = false;
      else if (c === "\\") escaped = true;
      else if (c === '"') quoted = false;
    } else if (c === '"') quoted = true;
    else if (c === "[" || c === "{") {
      if (++depth > MAX_CAPTURE_DEPTH) throw new Error("telemetry capture JSON nesting/work limit exceeded");
    } else if ((c === "]" || c === "}") && --depth === 0) {
      // Charge the subsequent parse too, even when this is not a string array.
      budget.remaining -= i + 1 - start;
      if (budget.remaining < 0) throw new Error("telemetry capture JSON nesting/work limit exceeded");
      return i + 1;
    }
  }
}

const secret = (value: string): boolean => value.length > 0 && value !== CAPTURE_REDACTED;
function decodeName(name: string): string {
  // Query parameter names are ASCII. Decode each escape separately so an
  // unrelated malformed UTF-8 escape cannot conceal a recognized ASCII name.
  return name.replace(/%([0-9a-f]{2})/gi, (_, hex) => String.fromCharCode(parseInt(hex, 16))).toLowerCase();
}

export function captureUrl(input: string): string {
  let url = input;
  const start = url.indexOf("://") + 3;
  const ends = [url.indexOf("/", start), url.indexOf("?", start), url.indexOf("#", start)].filter(p => p >= 0);
  const end = ends.length ? Math.min(...ends) : url.length;
  const authority = url.slice(start, end);
  const at = authority.lastIndexOf("@");
  if (at >= 0) {
    const info = authority.slice(0, at);
    const colon = info.indexOf(":");
    if (colon >= 0 && secret(info.slice(colon + 1)))
      url = url.slice(0, start) + info.slice(0, colon + 1) + CAPTURE_REDACTED + authority.slice(at) + url.slice(end);
  }
  const queryStart = url.indexOf("?");
  if (queryStart < 0) return url;
  const fragment = url.indexOf("#", queryStart);
  const queryEnd = fragment >= 0 ? fragment : url.length;
  const query = url.slice(queryStart + 1, queryEnd);
  const field = (raw: string): string => {
    const eq = raw.indexOf("=");
    if (eq < 0 || !QUERY_NAMES.has(decodeName(raw.slice(0, eq))) || !secret(raw.slice(eq + 1))) return raw;
    return raw.slice(0, eq + 1) + CAPTURE_REDACTED;
  };
  const parts: string[] = [];
  let position = 0;
  for (const match of query.matchAll(/&amp;|[&;]/gi)) {
    parts.push(field(query.slice(position, match.index)), match[0]);
    position = match.index! + match[0].length;
  }
  parts.push(field(query.slice(position)));
  return url.slice(0, queryStart + 1) + parts.join("") + url.slice(queryEnd);
}

function plain(text: string): string {
  return text.replace(/\b[a-zA-Z][a-zA-Z0-9+.-]*:\/\/[^\s<>"'`]+/g, captureUrl);
}

function rawHeaders(text: string): string {
  const tokens = text.matchAll(/"(?:[^"\\]|\\.)*"/g);
  let token = tokens.next().value;
  const header = new RegExp(String.raw`(?<![^\s"'])((?:${[...HEADER_NAMES].sort().join("|")})[ \t]*:[ \t]*)`, "gi");
  const parts: string[] = [];
  let start = 0;
  for (const match of text.matchAll(header)) {
    if (match.index! < start) continue;
    while (token && token.index! + token[0].length <= match.index!) token = tokens.next().value;
    if (token && token.index! <= match.index! && match.index! < token.index! + token[0].length) continue;
    let end = text.indexOf("\n", match.index! + match[0].length);
    if (end < 0) end = text.length;
    if (end > 0 && text[end - 1] === "\r") end--;
    if (match.index! > 0 && text[match.index! - 1] === "'") {
      const quote = text.indexOf("'", match.index!);
      if (quote >= 0) end = quote;
    }
    const value = text.slice(match.index! + match[1].length, end);
    parts.push(text.slice(start, match.index), match[1], secret(value) ? CAPTURE_REDACTED : value);
    start = end;
  }
  parts.push(text.slice(start));
  return parts.join("");
}

export const MAX_CAPTURE_LENGTH = 16 * 1024 * 1024;
export const MAX_CAPTURE_DEPTH = 16;

/** JSON tokens keep their original spelling unless content changes. Keys and
 * bytes outside changed tokens stay untouched; escaped string values are read. */
/** Return `text` unchanged, refusing it if it exceeds the capture limit. */
export function captureLength(text: string): string {
  if (text.length > MAX_CAPTURE_LENGTH) throw new Error("telemetry capture text exceeds 16 MiB character limit");
  return text;
}

/**
 * Remove transport credentials from text that carries transport syntax.
 *
 * For diagnostics and transport metadata — a URL, a log line, span
 * attributes. Not for a recorded message: see `captureEvent`.
 */
export function captureText(text: string): string {
  captureLength(text);
  return capture(text, 0, { remaining: Math.max(65536, text.length * 4) });
}

function capture(text: string, depth: number, budget: { remaining: number }): string {
  budget.remaining -= text.length;
  if (depth > MAX_CAPTURE_DEPTH || budget.remaining < 0) throw new Error("telemetry capture JSON nesting/work limit exceeded");
  const singleHeader = new RegExp(String.raw`('(?:${[...HEADER_NAMES].sort().join("|")})'[ \t]*:[ \t]*)('(?:[^'\\]|\\.)*')`, "gi");
  text = text.replace(singleHeader, (whole, name: string, raw: string) => secret(raw.slice(1, -1)) ? name + "'" + CAPTURE_REDACTED + "'" : whole);
  text = rawHeaders(text);
  const tokens = [...text.matchAll(/"(?:[^"\\]|\\.)*"/g)];
  const pieces: string[] = [];
  let start = 0;
  for (let index = 0; index < tokens.length; index++) {
    const token = tokens[index];
    if (token.index! < start) continue;
    pieces.push(plain(text.slice(start, token.index)));
    const raw = token[0];
    let value: string;
    try { value = JSON.parse(raw); }
    catch { pieces.push(plain(raw)); start = token.index! + raw.length; continue; }
    const end = token.index! + raw.length;
    const keyColon = text.slice(end).match(/^\s*:\s*/);
    const isKey = keyColon !== null;
    const valueStart = end + (keyColon?.[0].length ?? 0);
    if (isKey && headerKey(value) && text[valueStart] === "[") {
      const finish = arrayEnd(text, valueStart, budget);
      if (finish !== undefined) {
        try {
          const values = JSON.parse(text.slice(valueStart, finish));
          if (Array.isArray(values) && values.every(v => typeof v === "string")) {
            const clean = values.map(v => secret(v) ? CAPTURE_REDACTED : v);
            pieces.push(raw + text.slice(end, valueStart) + (values.every((v, i) => v === clean[i]) ? text.slice(valueStart, finish) : JSON.stringify(clean)));
            start = finish;
            continue;
          }
        } catch { /* malformed array remains text */ }
      }
    }
    let headerValue = false;
    if (index > 0) {
      const previous = tokens[index - 1];
      if (/^\s*:\s*$/.test(text.slice(previous.index! + previous[0].length, token.index))) {
        try { headerValue = headerKey(JSON.parse(previous[0])); } catch { /* not JSON */ }
      }
    }
    const clean = headerValue && secret(value) ? CAPTURE_REDACTED : isKey ? value : capture(value, depth + 1, budget);
    pieces.push(clean === value ? raw : JSON.stringify(clean));
    start = end;
  }
  pieces.push(plain(text.slice(start)));
  return pieces.join("");
}
