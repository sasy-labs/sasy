from . import policy_engine_pb2 as _policy_engine_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class CheckAuthRequest(_message.Message):
    __slots__ = ("current_node_ids", "actions", "entity", "roles", "session_id", "tenant", "principal")
    CURRENT_NODE_IDS_FIELD_NUMBER: _ClassVar[int]
    ACTIONS_FIELD_NUMBER: _ClassVar[int]
    ENTITY_FIELD_NUMBER: _ClassVar[int]
    ROLES_FIELD_NUMBER: _ClassVar[int]
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    TENANT_FIELD_NUMBER: _ClassVar[int]
    PRINCIPAL_FIELD_NUMBER: _ClassVar[int]
    current_node_ids: _containers.RepeatedScalarFieldContainer[str]
    actions: _containers.RepeatedCompositeFieldContainer[_policy_engine_pb2.Action]
    entity: str
    roles: _containers.RepeatedScalarFieldContainer[str]
    session_id: str
    tenant: str
    principal: str
    def __init__(self, current_node_ids: _Optional[_Iterable[str]] = ..., actions: _Optional[_Iterable[_Union[_policy_engine_pb2.Action, _Mapping]]] = ..., entity: _Optional[str] = ..., roles: _Optional[_Iterable[str]] = ..., session_id: _Optional[str] = ..., tenant: _Optional[str] = ..., principal: _Optional[str] = ...) -> None: ...

class GraphUpdateBatch(_message.Message):
    __slots__ = ("updates",)
    UPDATES_FIELD_NUMBER: _ClassVar[int]
    updates: _containers.RepeatedCompositeFieldContainer[GraphUpdateEntry]
    def __init__(self, updates: _Optional[_Iterable[_Union[GraphUpdateEntry, _Mapping]]] = ...) -> None: ...

class GraphUpdateEntry(_message.Message):
    __slots__ = ("node_created", "node_deleted", "edge_created", "edge_deleted")
    NODE_CREATED_FIELD_NUMBER: _ClassVar[int]
    NODE_DELETED_FIELD_NUMBER: _ClassVar[int]
    EDGE_CREATED_FIELD_NUMBER: _ClassVar[int]
    EDGE_DELETED_FIELD_NUMBER: _ClassVar[int]
    node_created: NodeCreated
    node_deleted: str
    edge_created: EdgeUpdate
    edge_deleted: EdgeUpdate
    def __init__(self, node_created: _Optional[_Union[NodeCreated, _Mapping]] = ..., node_deleted: _Optional[str] = ..., edge_created: _Optional[_Union[EdgeUpdate, _Mapping]] = ..., edge_deleted: _Optional[_Union[EdgeUpdate, _Mapping]] = ...) -> None: ...

class NodeCreated(_message.Message):
    __slots__ = ("id", "content", "role", "agent", "tools", "entity", "derived_from", "session_id")
    ID_FIELD_NUMBER: _ClassVar[int]
    CONTENT_FIELD_NUMBER: _ClassVar[int]
    ROLE_FIELD_NUMBER: _ClassVar[int]
    AGENT_FIELD_NUMBER: _ClassVar[int]
    TOOLS_FIELD_NUMBER: _ClassVar[int]
    ENTITY_FIELD_NUMBER: _ClassVar[int]
    DERIVED_FROM_FIELD_NUMBER: _ClassVar[int]
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    id: str
    content: str
    role: str
    agent: str
    tools: _containers.RepeatedCompositeFieldContainer[ToolInfo]
    entity: str
    derived_from: ToolInfo
    session_id: str
    def __init__(self, id: _Optional[str] = ..., content: _Optional[str] = ..., role: _Optional[str] = ..., agent: _Optional[str] = ..., tools: _Optional[_Iterable[_Union[ToolInfo, _Mapping]]] = ..., entity: _Optional[str] = ..., derived_from: _Optional[_Union[ToolInfo, _Mapping]] = ..., session_id: _Optional[str] = ...) -> None: ...

class ToolInfo(_message.Message):
    __slots__ = ("name", "arguments")
    NAME_FIELD_NUMBER: _ClassVar[int]
    ARGUMENTS_FIELD_NUMBER: _ClassVar[int]
    name: str
    arguments: str
    def __init__(self, name: _Optional[str] = ..., arguments: _Optional[str] = ...) -> None: ...

class EdgeUpdate(_message.Message):
    __slots__ = ("source", "destination", "session_id", "principal", "entity")
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    DESTINATION_FIELD_NUMBER: _ClassVar[int]
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    PRINCIPAL_FIELD_NUMBER: _ClassVar[int]
    ENTITY_FIELD_NUMBER: _ClassVar[int]
    source: str
    destination: str
    session_id: str
    principal: str
    entity: str
    def __init__(self, source: _Optional[str] = ..., destination: _Optional[str] = ..., session_id: _Optional[str] = ..., principal: _Optional[str] = ..., entity: _Optional[str] = ...) -> None: ...

class SyncStatusProto(_message.Message):
    __slots__ = ("current_sequence", "node_count", "edge_count", "connected")
    CURRENT_SEQUENCE_FIELD_NUMBER: _ClassVar[int]
    NODE_COUNT_FIELD_NUMBER: _ClassVar[int]
    EDGE_COUNT_FIELD_NUMBER: _ClassVar[int]
    CONNECTED_FIELD_NUMBER: _ClassVar[int]
    current_sequence: int
    node_count: int
    edge_count: int
    connected: bool
    def __init__(self, current_sequence: _Optional[int] = ..., node_count: _Optional[int] = ..., edge_count: _Optional[int] = ..., connected: bool = ...) -> None: ...
