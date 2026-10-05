"""Narrow transport-credential hygiene for telemetry copies, not policy/HTTP input.

Only full URLs and authentication-header syntax are recognized. This is not a
secret-shape scanner. It never rewrites identities, graph keys or policy source.
"""
from __future__ import annotations

import json
import logging
import re
import traceback

REDACTED = "[redacted]"
QUERY_NAMES = frozenset({
    "key", "api_key", "apikey", "access_token", "token", "auth_token", "id_token",
    "refresh_token", "client_secret", "password", "passwd", "secret", "sig",
    "signature", "x-amz-signature", "x-goog-signature",
})
HEADER_NAMES = frozenset({
    "authorization", "proxy-authorization", "x-api-key", "api-key", "x-goog-api-key",
    "x-auth-token", "cookie", "set-cookie",
})
_CAPTURE_MARKERS = HEADER_NAMES | frozenset(name.replace("-", "_") for name in HEADER_NAMES)
_URL = re.compile(r"\b[a-zA-Z][a-zA-Z0-9+.-]*://[^\s<>\"'`]+")
_JSON_STRING = re.compile(r'"(?:[^"\\]|\\.)*"')
_SEPARATOR = re.compile(r"&amp;|[&;]", re.IGNORECASE)


def _header_key(value: str) -> bool:
    name = value.lower()
    for prefix in ("http.request.header.", "http.response.header."):
        if name.startswith(prefix):
            name = name[len(prefix):].replace("_", "-")
            break
    return name in HEADER_NAMES


def _query_name(value: str) -> str:
    return re.sub(r"%([0-9a-fA-F]{2})", lambda m: chr(int(m.group(1), 16)), value).lower()


def _secret(value: str) -> bool:
    return bool(value) and value != REDACTED


def capture_url(url: str) -> str:
    """Preserve URL spelling; replace only credential parameter/password values."""
    authority_start = url.find("://") + 3
    authority_end = min((p for p in (url.find("/", authority_start), url.find("?", authority_start), url.find("#", authority_start)) if p >= 0), default=len(url))
    authority = url[authority_start:authority_end]
    if "@" in authority:
        userinfo, host = authority.rsplit("@", 1)
        if ":" in userinfo:
            user, password = userinfo.split(":", 1)
            if _secret(password):
                url = url[:authority_start] + user + ":" + REDACTED + "@" + host + url[authority_end:]
    query_start = url.find("?")
    if query_start < 0:
        return url
    fragment = url.find("#", query_start)
    query_end = fragment if fragment >= 0 else len(url)
    query = url[query_start + 1:query_end]
    def field(raw: str) -> str:
        name, sep, value = raw.partition("=")
        if sep and _query_name(name) in QUERY_NAMES and _secret(value):
            return name + "=" + REDACTED
        return raw
    parts: list[str] = []
    start = 0
    for match in _SEPARATOR.finditer(query):
        parts.extend((field(query[start:match.start()]), match.group()))
        start = match.end()
    parts.append(field(query[start:]))
    return url[:query_start + 1] + "".join(parts) + url[query_end:]


def _plain(text: str) -> str:
    return _URL.sub(lambda m: capture_url(m.group()), text)


_RAW_HEADER = re.compile(
    r"(?i)(?<![^\s\"'])((?:" + "|".join(sorted(HEADER_NAMES)) + r")[ \t]*:[ \t]*)"
)


def _raw_headers(text: str) -> str:
    # Headers outside JSON strings may themselves contain quoted Digest fields.
    # Handle the entire line before tokenizing those fields. Inside a JSON string,
    # decode first; this avoids damaging JSON escapes or adjacent object fields.
    tokens = iter(_JSON_STRING.finditer(text))
    token = next(tokens, None)
    parts: list[str] = []
    start = 0
    for match in _RAW_HEADER.finditer(text):
        if match.start() < start:
            continue
        while token and token.end() <= match.start():
            token = next(tokens, None)
        if token and token.start() <= match.start() < token.end():
            continue
        end = text.find("\n", match.end())
        if end < 0:
            end = len(text)
        if end and text[end - 1] == "\r":
            end -= 1
        if match.start() and text[match.start() - 1] == "'":
            quote = text.find("'", match.start())
            if quote >= 0:
                end = quote
        value = text[match.end():end]
        parts.extend((text[start:match.start()], match.group(1), REDACTED if _secret(value) else value))
        start = end
    parts.append(text[start:])
    return "".join(parts)


MAX_CAPTURE_LENGTH = 16 * 1024 * 1024
MAX_CAPTURE_DEPTH = 16


def _array_end(text: str, start: int, budget: list[int]) -> int | None:
    """Bound scanning and parsing before handing a header array to json.loads."""
    depth = 0
    quoted = escaped = False
    for index in range(start, len(text)):
        budget[0] -= 1
        if budget[0] < 0:
            raise ValueError("telemetry capture JSON nesting/work limit exceeded")
        char = text[index]
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char in "[{":
            depth += 1
            if depth > MAX_CAPTURE_DEPTH:
                raise ValueError("telemetry capture JSON nesting/work limit exceeded")
        elif char in "]}":
            depth -= 1
            if depth == 0:
                # Parsing this slice is additional work, including when its
                # structure is not a string array and another key follows.
                budget[0] -= index + 1 - start
                if budget[0] < 0:
                    raise ValueError("telemetry capture JSON nesting/work limit exceeded")
                return index + 1
    return None


def capture_length(text: str) -> str:
    """Return *text* unchanged, refusing it if it exceeds the capture limit.

    The limit is what bounds one recorded value on the wire and in the store.
    It applies wherever text is captured, including where the text itself is
    kept exactly as it was.
    """
    if len(text) > MAX_CAPTURE_LENGTH:
        raise ValueError("telemetry capture text exceeds 16 MiB character limit")
    return text


def capture_text(text: str) -> str:
    """Remove transport credentials from text that carries transport syntax.

    For diagnostics and transport metadata — span attributes, log records, a
    URL, the feedback text handed back to a model. Not for a recorded
    message: see :func:`capture_events`.
    """
    capture_length(text)
    # Most message text contains no transport credentials. Escapes must take
    # the full path: decoding JSON can reveal hidden URL/header syntax.
    if text.isascii() and "\\" not in text:
        if ":" not in text:
            return text
        if "://" not in text:
            lower = text.lower()
            if not any(name in lower for name in _CAPTURE_MARKERS):
                return text
    return _capture_text(text, 0, [max(65536, len(text) * 4)])


def _capture_text(text: str, depth: int, budget: list[int]) -> str:
    """Sanitize transport syntax in text, including JSON-escaped string values.

    JSON tokens retain their original spelling unless their content changes.
    JSON keys and every byte outside changed string tokens remain untouched.
    """
    budget[0] -= len(text)
    if depth > MAX_CAPTURE_DEPTH or budget[0] < 0:
        raise ValueError("telemetry capture JSON nesting/work limit exceeded")
    text = re.sub(
        r"(?i)('(?:" + "|".join(sorted(HEADER_NAMES)) + r")'[ \t]*:[ \t]*)('(?:[^'\\]|\\.)*')",
        lambda m: m.group(1) + "'" + REDACTED + "'" if _secret(m.group(2)[1:-1]) else m.group(), text,
    )
    text = _raw_headers(text)
    tokens = list(_JSON_STRING.finditer(text))
    pieces: list[str] = []
    start = 0
    for index, token in enumerate(tokens):
        if token.start() < start:
            continue
        pieces.append(_plain(text[start:token.start()]))
        raw = token.group()
        try:
            value = json.loads(raw)
        except ValueError:
            pieces.append(_plain(raw))
            start = token.end()
            continue
        key_colon = re.compile(r"\s*:\s*").match(text, token.end())
        is_key = key_colon is not None
        if key_colon and _header_key(value) and text[key_colon.end():key_colon.end() + 1] == "[":
            end = _array_end(text, key_colon.end(), budget)
            if end is not None:
                try:
                    values = json.loads(text[key_colon.end():end])
                    if isinstance(values, list) and all(isinstance(v, str) for v in values):
                        clean_values = [REDACTED if _secret(v) else v for v in values]
                        replacement = text[key_colon.end():end] if values == clean_values else json.dumps(clean_values, ensure_ascii=False, separators=(",", ":"))
                        pieces.append(raw + text[token.end():key_colon.end()] + replacement)
                        start = end
                        continue
                except ValueError:
                    pass
        header_value = False
        if index:
            previous = tokens[index - 1]
            if re.fullmatch(r"\s*:\s*", text[previous.end():token.start()]):
                try:
                    header_value = _header_key(json.loads(previous.group()))
                except ValueError:
                    pass
        clean = REDACTED if header_value and _secret(value) else value if is_key else _capture_text(value, depth + 1, budget)
        pieces.append(raw if clean == value else json.dumps(clean, ensure_ascii=False))
        start = token.end()
    pieces.append(_plain(text[start:]))
    return "".join(pieces)


def capture_events(events):
    """Clone protobuf events, checking the size of the text they carry.

    A message's text, an adapter's metadata record of it and a tool call's
    arguments are recorded as they are.
    They are what the policy engine reasons over, and the reference monitor
    is given the same bytes when it decides, so rewriting them here would
    leave a policy reading text that neither the model nor the monitor saw —
    and a rule that reads an ancestor's contents would decide on something
    that never happened. Removing credentials from a message's own text is
    not this layer's job either: the graph holds the conversation, so it is
    sensitive whatever this function does, and it is protected as a store
    rather than by rewriting the record. Transport metadata is different and
    is still scrubbed where it appears — see :func:`capture_computations`,
    :func:`capture_logger` and the HTTP hooks.
    """
    from sasy.proto.observability_pb2 import Event

    copies = []
    for event in events:
        copy = Event(**event) if isinstance(event, dict) else Event()
        if not isinstance(event, dict):
            copy.CopyFrom(event)
        if copy.HasField("text"):
            capture_length(copy.text)
        if copy.HasField("metadata"):
            capture_length(copy.metadata)
        for tool in list(copy.tools) + ([copy.derived_from] if copy.HasField("derived_from") else []):
            if tool.HasField("arguments"):
                capture_length(tool.arguments)
        copies.append(copy)
    return copies


def capture_computations(computations):
    """Clone spans; leave IDs, names, identity, links and timestamps intact."""
    from sasy.proto.observability_pb2 import Computation

    copies = []
    for computation in computations:
        copy = Computation(**computation) if isinstance(computation, dict) else Computation()
        if not isinstance(computation, dict):
            copy.CopyFrom(computation)
        for field in ("status_message", "attributes_json", "events_json"):
            value = getattr(copy, field)
            if value:
                setattr(copy, field, capture_text(value))
        copies.append(copy)
    return copies


class CaptureLogFilter(logging.Filter):
    """Sanitize messages emitted by an SDK-owned logger, not application logs."""
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            record.msg = capture_text(record.getMessage())
            record.args = ()
            if record.exc_info:
                record.exc_text = capture_text("".join(traceback.format_exception(*record.exc_info)))
                record.exc_info = None
            elif record.exc_text:
                record.exc_text = capture_text(record.exc_text)
            if record.stack_info:
                record.stack_info = capture_text(record.stack_info)
        except ValueError:
            record.msg = "[telemetry diagnostic omitted: capture limit exceeded]"
            record.args = ()
            record.exc_info = None
            record.exc_text = None
            record.stack_info = None
        return True


def capture_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not any(isinstance(f, CaptureLogFilter) for f in logger.filters):
        logger.addFilter(CaptureLogFilter())
    return logger
