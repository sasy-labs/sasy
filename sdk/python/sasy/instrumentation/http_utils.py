"""
HTTP request/response conversion utilities for instrumentation.
"""

import io
from collections.abc import Iterator

import httpx
import requests
import urllib3
from requests.cookies import MockRequest
from requests.structures import CaseInsensitiveDict

from sasy.proto.reference_monitor_pb2 import (
    BaseRequest,
    HttpHeader,
    HTTPResponse,
    Message,
)


def decode_bytes(s: str | bytes | None) -> str | None:
    """Decode bytes into a string."""
    if s is None:
        return None
    if isinstance(s, str):
        return s
    return s.decode("UTF-8")


def encode_bytes(s: str | bytes | None) -> bytes | None:
    """Encode a string into bytes."""
    if s is None:
        return None
    if isinstance(s, bytes):
        return s
    return s.encode("UTF-8")


def from_httpx_request(request: httpx.Request) -> BaseRequest:
    """Converts an `httpx.Request` a protobuf request message"""
    return BaseRequest(
        method=request.method,
        url=str(request.url),
        message=Message(
            headers=[HttpHeader(key=k, value=v) for (k, v) in request.headers.raw],
            content=request.read(),
        ),
    )


def to_httpx_request(request: BaseRequest) -> httpx.Request:
    """Converts a protobuf request message to an `httpx.Request`"""
    return httpx.Request(
        method=request.method,
        url=request.url,
        content=request.message.content,
        headers={h.key: h.value for h in request.message.headers},
    )


def to_httpx_response(
    response: HTTPResponse,
    stream: httpx.SyncByteStream | httpx.AsyncByteStream,
    base_request: httpx.Request | None = None,
) -> httpx.Response:
    """
    Translate a response message and content stream into an `httpx.Response`.
    """
    if response.message.HasField("content") and response.message.content is not None:
        result = httpx.Response(
            status_code=response.status,
            headers={h.key: h.value for h in response.message.headers},
            request=base_request,
        )
        result._content = response.message.content
        result.stream = httpx.ByteStream(result._content)
        return result

    return httpx.Response(
        status_code=response.status,
        headers={h.key: h.value for h in response.message.headers},
        request=base_request,
        stream=stream,
    )


def from_requests_request(request: requests.PreparedRequest) -> BaseRequest:
    """Translate a request from `requests` to protobuf form."""
    return BaseRequest(
        method=request.method,
        url=str(request.url),
        message=Message(
            headers=[
                HttpHeader(key=encode_bytes(k), value=encode_bytes(v))
                for k, v in request.headers.items()
            ],
            content=encode_bytes(request.body),  # type: ignore
        ),
    )


def to_requests_request(request: BaseRequest) -> requests.PreparedRequest:
    """Translate a protobuf request to `requests` form."""
    r = requests.PreparedRequest()
    r.method = request.method
    r.url = request.url
    r.headers = CaseInsensitiveDict(
        {k: v for h in request.message.headers
         if (k := decode_bytes(h.key)) is not None and (v := decode_bytes(h.value)) is not None}
    )
    r.body = request.message.content
    return r


class RawStream:
    """Wraps an httpx response stream in the form used by urllib3."""

    def __init__(self, httpx_response: httpx.Response):
        self.httpx_response = httpx_response

    def stream(
        self, chunk_size: int = 2**16, decode_content: bool = True
    ) -> Iterator[bytes]:
        if decode_content:
            try:
                yield from self.httpx_response.iter_bytes(chunk_size)
            except httpx.DecodingError as e:
                raise urllib3.exceptions.DecodeError(e)
        else:
            yield from self.httpx_response.iter_raw(chunk_size)


def to_requests_response(
    response: HTTPResponse,
    requests_request: requests.PreparedRequest,
    stream: httpx.SyncByteStream,
) -> requests.Response:
    """
    Convert the base response and content stream from the reference monitor
    to a `requests` response.
    """
    r = requests.Response()
    r.url = requests_request.url or ""
    r.status_code = response.status
    r.reason = requests.codes.get(r.status_code, None)
    r.headers = CaseInsensitiveDict(
        {k: v for h in response.message.headers
         if (k := decode_bytes(h.key)) is not None and (v := decode_bytes(h.value)) is not None}
    )
    r.encoding = requests.utils.get_encoding_from_headers(r.headers)
    r.request = requests_request

    httpx_response = to_httpx_response(response, stream)
    r.cookies.extract_cookies(
        httpx.Cookies._CookieCompatResponse(httpx_response),  # type: ignore
        MockRequest(requests_request),  # type: ignore
    )

    if response.message.HasField("content") and response.message.content is not None:
        r._content = response.message.content
        r.raw = io.BytesIO(r._content)
    else:
        r.raw = RawStream(httpx_response)
    return r
