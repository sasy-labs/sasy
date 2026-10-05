from . import policy_engine_pb2 as _policy_engine_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class HttpHeader(_message.Message):
    __slots__ = ("key", "value")
    KEY_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    key: bytes
    value: bytes
    def __init__(self, key: _Optional[bytes] = ..., value: _Optional[bytes] = ...) -> None: ...

class Message(_message.Message):
    __slots__ = ("headers", "content")
    HEADERS_FIELD_NUMBER: _ClassVar[int]
    CONTENT_FIELD_NUMBER: _ClassVar[int]
    headers: _containers.RepeatedCompositeFieldContainer[HttpHeader]
    content: bytes
    def __init__(self, headers: _Optional[_Iterable[_Union[HttpHeader, _Mapping]]] = ..., content: _Optional[bytes] = ...) -> None: ...

class BaseRequest(_message.Message):
    __slots__ = ("url", "method", "message")
    URL_FIELD_NUMBER: _ClassVar[int]
    METHOD_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    url: str
    method: str
    message: Message
    def __init__(self, url: _Optional[str] = ..., method: _Optional[str] = ..., message: _Optional[_Union[Message, _Mapping]] = ...) -> None: ...

class HTTPRequest(_message.Message):
    __slots__ = ("request", "input_node_ids", "session_id", "entity")
    REQUEST_FIELD_NUMBER: _ClassVar[int]
    INPUT_NODE_IDS_FIELD_NUMBER: _ClassVar[int]
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    ENTITY_FIELD_NUMBER: _ClassVar[int]
    request: BaseRequest
    input_node_ids: _containers.RepeatedScalarFieldContainer[str]
    session_id: str
    entity: str
    def __init__(self, request: _Optional[_Union[BaseRequest, _Mapping]] = ..., input_node_ids: _Optional[_Iterable[str]] = ..., session_id: _Optional[str] = ..., entity: _Optional[str] = ...) -> None: ...

class HTTPResponse(_message.Message):
    __slots__ = ("status", "message")
    STATUS_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    status: int
    message: Message
    def __init__(self, status: _Optional[int] = ..., message: _Optional[_Union[Message, _Mapping]] = ...) -> None: ...

class ToolCallRequest(_message.Message):
    __slots__ = ("fn_name", "args", "input_node_ids", "session_id", "entity", "metadata")
    FN_NAME_FIELD_NUMBER: _ClassVar[int]
    ARGS_FIELD_NUMBER: _ClassVar[int]
    INPUT_NODE_IDS_FIELD_NUMBER: _ClassVar[int]
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    ENTITY_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    fn_name: str
    args: str
    input_node_ids: _containers.RepeatedScalarFieldContainer[str]
    session_id: str
    entity: str
    metadata: _containers.RepeatedCompositeFieldContainer[_policy_engine_pb2.PolicyMetadataFact]
    def __init__(self, fn_name: _Optional[str] = ..., args: _Optional[str] = ..., input_node_ids: _Optional[_Iterable[str]] = ..., session_id: _Optional[str] = ..., entity: _Optional[str] = ..., metadata: _Optional[_Iterable[_Union[_policy_engine_pb2.PolicyMetadataFact, _Mapping]]] = ...) -> None: ...

class ToolCallResponse(_message.Message):
    __slots__ = ("authorized", "denial_trace", "transform_ids")
    AUTHORIZED_FIELD_NUMBER: _ClassVar[int]
    DENIAL_TRACE_FIELD_NUMBER: _ClassVar[int]
    TRANSFORM_IDS_FIELD_NUMBER: _ClassVar[int]
    authorized: bool
    denial_trace: _policy_engine_pb2.DenialTrace
    transform_ids: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, authorized: bool = ..., denial_trace: _Optional[_Union[_policy_engine_pb2.DenialTrace, _Mapping]] = ..., transform_ids: _Optional[_Iterable[str]] = ...) -> None: ...
