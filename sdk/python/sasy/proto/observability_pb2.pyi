from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Role(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    SYSTEM: _ClassVar[Role]
    USER: _ClassVar[Role]
    LLM: _ClassVar[Role]
    AGENT: _ClassVar[Role]

class UpdateType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    HEARTBEAT: _ClassVar[UpdateType]
    NODE_CREATED: _ClassVar[UpdateType]
    NODE_DELETED: _ClassVar[UpdateType]
    EDGE_CREATED: _ClassVar[UpdateType]
    EDGE_DELETED: _ClassVar[UpdateType]
    FULL_STATE: _ClassVar[UpdateType]
    COMPUTATION_CREATED: _ClassVar[UpdateType]
    COMPUTATION_DELETED: _ClassVar[UpdateType]
    COMPUTATION_EDGE_CREATED: _ClassVar[UpdateType]
    COMPUTATION_MESSAGE_LINK_CREATED: _ClassVar[UpdateType]
    FULL_STATE_START: _ClassVar[UpdateType]
    FULL_STATE_CHUNK: _ClassVar[UpdateType]
    FULL_STATE_COMMIT: _ClassVar[UpdateType]

class SpanStatusCode(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    STATUS_UNSET: _ClassVar[SpanStatusCode]
    STATUS_OK: _ClassVar[SpanStatusCode]
    STATUS_ERROR: _ClassVar[SpanStatusCode]

class ComputationMessageEdgeType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    PRODUCES: _ClassVar[ComputationMessageEdgeType]
    CONSUMES: _ClassVar[ComputationMessageEdgeType]
SYSTEM: Role
USER: Role
LLM: Role
AGENT: Role
HEARTBEAT: UpdateType
NODE_CREATED: UpdateType
NODE_DELETED: UpdateType
EDGE_CREATED: UpdateType
EDGE_DELETED: UpdateType
FULL_STATE: UpdateType
COMPUTATION_CREATED: UpdateType
COMPUTATION_DELETED: UpdateType
COMPUTATION_EDGE_CREATED: UpdateType
COMPUTATION_MESSAGE_LINK_CREATED: UpdateType
FULL_STATE_START: UpdateType
FULL_STATE_CHUNK: UpdateType
FULL_STATE_COMMIT: UpdateType
STATUS_UNSET: SpanStatusCode
STATUS_OK: SpanStatusCode
STATUS_ERROR: SpanStatusCode
PRODUCES: ComputationMessageEdgeType
CONSUMES: ComputationMessageEdgeType

class GraphUpdates(_message.Message):
    __slots__ = ("sequence", "updates")
    SEQUENCE_FIELD_NUMBER: _ClassVar[int]
    UPDATES_FIELD_NUMBER: _ClassVar[int]
    sequence: int
    updates: _containers.RepeatedCompositeFieldContainer[GraphUpdate]
    def __init__(self, sequence: _Optional[int] = ..., updates: _Optional[_Iterable[_Union[GraphUpdate, _Mapping]]] = ...) -> None: ...

class Events(_message.Message):
    __slots__ = ("events", "session_id")
    EVENTS_FIELD_NUMBER: _ClassVar[int]
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    events: _containers.RepeatedCompositeFieldContainer[Event]
    session_id: str
    def __init__(self, events: _Optional[_Iterable[_Union[Event, _Mapping]]] = ..., session_id: _Optional[str] = ...) -> None: ...

class Event(_message.Message):
    __slots__ = ("text", "agent", "role", "id", "tools", "derived_from", "principal", "entity", "metadata")
    TEXT_FIELD_NUMBER: _ClassVar[int]
    AGENT_FIELD_NUMBER: _ClassVar[int]
    ROLE_FIELD_NUMBER: _ClassVar[int]
    ID_FIELD_NUMBER: _ClassVar[int]
    TOOLS_FIELD_NUMBER: _ClassVar[int]
    DERIVED_FROM_FIELD_NUMBER: _ClassVar[int]
    PRINCIPAL_FIELD_NUMBER: _ClassVar[int]
    ENTITY_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    text: str
    agent: str
    role: Role
    id: str
    tools: _containers.RepeatedCompositeFieldContainer[Tool]
    derived_from: Tool
    principal: str
    entity: str
    metadata: str
    def __init__(self, text: _Optional[str] = ..., agent: _Optional[str] = ..., role: _Optional[_Union[Role, str]] = ..., id: _Optional[str] = ..., tools: _Optional[_Iterable[_Union[Tool, _Mapping]]] = ..., derived_from: _Optional[_Union[Tool, _Mapping]] = ..., principal: _Optional[str] = ..., entity: _Optional[str] = ..., metadata: _Optional[str] = ...) -> None: ...

class Tool(_message.Message):
    __slots__ = ("name", "arguments")
    NAME_FIELD_NUMBER: _ClassVar[int]
    ARGUMENTS_FIELD_NUMBER: _ClassVar[int]
    name: str
    arguments: str
    def __init__(self, name: _Optional[str] = ..., arguments: _Optional[str] = ...) -> None: ...

class IDs(_message.Message):
    __slots__ = ("ids",)
    IDS_FIELD_NUMBER: _ClassVar[int]
    ids: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, ids: _Optional[_Iterable[str]] = ...) -> None: ...

class Dependencies(_message.Message):
    __slots__ = ("edges", "session_id")
    EDGES_FIELD_NUMBER: _ClassVar[int]
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    edges: _containers.RepeatedCompositeFieldContainer[Edge]
    session_id: str
    def __init__(self, edges: _Optional[_Iterable[_Union[Edge, _Mapping]]] = ..., session_id: _Optional[str] = ...) -> None: ...

class EventsWithDependencies(_message.Message):
    __slots__ = ("events", "edges", "session_id")
    EVENTS_FIELD_NUMBER: _ClassVar[int]
    EDGES_FIELD_NUMBER: _ClassVar[int]
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    events: _containers.RepeatedCompositeFieldContainer[Event]
    edges: _containers.RepeatedCompositeFieldContainer[Edge]
    session_id: str
    def __init__(self, events: _Optional[_Iterable[_Union[Event, _Mapping]]] = ..., edges: _Optional[_Iterable[_Union[Edge, _Mapping]]] = ..., session_id: _Optional[str] = ...) -> None: ...

class EventSnapshot(_message.Message):
    __slots__ = ("event", "base_id", "dependencies", "reuse_dependencies", "content_hash")
    EVENT_FIELD_NUMBER: _ClassVar[int]
    BASE_ID_FIELD_NUMBER: _ClassVar[int]
    DEPENDENCIES_FIELD_NUMBER: _ClassVar[int]
    REUSE_DEPENDENCIES_FIELD_NUMBER: _ClassVar[int]
    CONTENT_HASH_FIELD_NUMBER: _ClassVar[int]
    event: Event
    base_id: str
    dependencies: _containers.RepeatedCompositeFieldContainer[Edge]
    reuse_dependencies: bool
    content_hash: bytes
    def __init__(self, event: _Optional[_Union[Event, _Mapping]] = ..., base_id: _Optional[str] = ..., dependencies: _Optional[_Iterable[_Union[Edge, _Mapping]]] = ..., reuse_dependencies: bool = ..., content_hash: _Optional[bytes] = ...) -> None: ...

class EventSnapshots(_message.Message):
    __slots__ = ("snapshots", "session_id")
    SNAPSHOTS_FIELD_NUMBER: _ClassVar[int]
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    snapshots: _containers.RepeatedCompositeFieldContainer[EventSnapshot]
    session_id: str
    def __init__(self, snapshots: _Optional[_Iterable[_Union[EventSnapshot, _Mapping]]] = ..., session_id: _Optional[str] = ...) -> None: ...

class Edge(_message.Message):
    __slots__ = ("source", "destination", "message_index", "proximal", "principal", "entity")
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    DESTINATION_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_INDEX_FIELD_NUMBER: _ClassVar[int]
    PROXIMAL_FIELD_NUMBER: _ClassVar[int]
    PRINCIPAL_FIELD_NUMBER: _ClassVar[int]
    ENTITY_FIELD_NUMBER: _ClassVar[int]
    source: str
    destination: str
    message_index: int
    proximal: bool
    principal: str
    entity: str
    def __init__(self, source: _Optional[str] = ..., destination: _Optional[str] = ..., message_index: _Optional[int] = ..., proximal: bool = ..., principal: _Optional[str] = ..., entity: _Optional[str] = ...) -> None: ...

class Response(_message.Message):
    __slots__ = ("response",)
    RESPONSE_FIELD_NUMBER: _ClassVar[int]
    response: str
    def __init__(self, response: _Optional[str] = ...) -> None: ...

class SliceRequest(_message.Message):
    __slots__ = ("event_id", "max_depth", "session_id")
    EVENT_ID_FIELD_NUMBER: _ClassVar[int]
    MAX_DEPTH_FIELD_NUMBER: _ClassVar[int]
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    event_id: str
    max_depth: int
    session_id: str
    def __init__(self, event_id: _Optional[str] = ..., max_depth: _Optional[int] = ..., session_id: _Optional[str] = ...) -> None: ...

class Graph(_message.Message):
    __slots__ = ("nodes", "edges")
    NODES_FIELD_NUMBER: _ClassVar[int]
    EDGES_FIELD_NUMBER: _ClassVar[int]
    nodes: _containers.RepeatedCompositeFieldContainer[Event]
    edges: _containers.RepeatedCompositeFieldContainer[Edge]
    def __init__(self, nodes: _Optional[_Iterable[_Union[Event, _Mapping]]] = ..., edges: _Optional[_Iterable[_Union[Edge, _Mapping]]] = ...) -> None: ...

class StateRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class GraphState(_message.Message):
    __slots__ = ("events", "edges", "sequence")
    EVENTS_FIELD_NUMBER: _ClassVar[int]
    EDGES_FIELD_NUMBER: _ClassVar[int]
    SEQUENCE_FIELD_NUMBER: _ClassVar[int]
    events: _containers.RepeatedCompositeFieldContainer[Event]
    edges: _containers.RepeatedCompositeFieldContainer[Edge]
    sequence: int
    def __init__(self, events: _Optional[_Iterable[_Union[Event, _Mapping]]] = ..., edges: _Optional[_Iterable[_Union[Edge, _Mapping]]] = ..., sequence: _Optional[int] = ...) -> None: ...

class StreamMessage(_message.Message):
    __slots__ = ("subscribe", "ack")
    SUBSCRIBE_FIELD_NUMBER: _ClassVar[int]
    ACK_FIELD_NUMBER: _ClassVar[int]
    subscribe: StreamRequest
    ack: Ack
    def __init__(self, subscribe: _Optional[_Union[StreamRequest, _Mapping]] = ..., ack: _Optional[_Union[Ack, _Mapping]] = ...) -> None: ...

class StreamRequest(_message.Message):
    __slots__ = ("from_sequence",)
    FROM_SEQUENCE_FIELD_NUMBER: _ClassVar[int]
    from_sequence: int
    def __init__(self, from_sequence: _Optional[int] = ...) -> None: ...

class Ack(_message.Message):
    __slots__ = ("position",)
    POSITION_FIELD_NUMBER: _ClassVar[int]
    position: int
    def __init__(self, position: _Optional[int] = ...) -> None: ...

class GraphUpdate(_message.Message):
    __slots__ = ("type", "event", "edge", "node_id", "source_id", "destination_id", "full_state", "timestamp", "computation", "computation_edge", "computation_message_edge", "state_chunk", "session_id")
    TYPE_FIELD_NUMBER: _ClassVar[int]
    EVENT_FIELD_NUMBER: _ClassVar[int]
    EDGE_FIELD_NUMBER: _ClassVar[int]
    NODE_ID_FIELD_NUMBER: _ClassVar[int]
    SOURCE_ID_FIELD_NUMBER: _ClassVar[int]
    DESTINATION_ID_FIELD_NUMBER: _ClassVar[int]
    FULL_STATE_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    COMPUTATION_FIELD_NUMBER: _ClassVar[int]
    COMPUTATION_EDGE_FIELD_NUMBER: _ClassVar[int]
    COMPUTATION_MESSAGE_EDGE_FIELD_NUMBER: _ClassVar[int]
    STATE_CHUNK_FIELD_NUMBER: _ClassVar[int]
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    type: UpdateType
    event: Event
    edge: Edge
    node_id: str
    source_id: str
    destination_id: str
    full_state: GraphState
    timestamp: str
    computation: Computation
    computation_edge: ComputationEdge
    computation_message_edge: ComputationMessageEdge
    state_chunk: GraphStateChunk
    session_id: str
    def __init__(self, type: _Optional[_Union[UpdateType, str]] = ..., event: _Optional[_Union[Event, _Mapping]] = ..., edge: _Optional[_Union[Edge, _Mapping]] = ..., node_id: _Optional[str] = ..., source_id: _Optional[str] = ..., destination_id: _Optional[str] = ..., full_state: _Optional[_Union[GraphState, _Mapping]] = ..., timestamp: _Optional[str] = ..., computation: _Optional[_Union[Computation, _Mapping]] = ..., computation_edge: _Optional[_Union[ComputationEdge, _Mapping]] = ..., computation_message_edge: _Optional[_Union[ComputationMessageEdge, _Mapping]] = ..., state_chunk: _Optional[_Union[GraphStateChunk, _Mapping]] = ..., session_id: _Optional[str] = ...) -> None: ...

class GraphStateChunk(_message.Message):
    __slots__ = ("events", "edges")
    EVENTS_FIELD_NUMBER: _ClassVar[int]
    EDGES_FIELD_NUMBER: _ClassVar[int]
    events: _containers.RepeatedCompositeFieldContainer[Event]
    edges: _containers.RepeatedCompositeFieldContainer[Edge]
    def __init__(self, events: _Optional[_Iterable[_Union[Event, _Mapping]]] = ..., edges: _Optional[_Iterable[_Union[Edge, _Mapping]]] = ...) -> None: ...

class SpanEvent(_message.Message):
    __slots__ = ("name", "timestamp_ns", "attributes")
    class AttributesEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    NAME_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_NS_FIELD_NUMBER: _ClassVar[int]
    ATTRIBUTES_FIELD_NUMBER: _ClassVar[int]
    name: str
    timestamp_ns: int
    attributes: _containers.ScalarMap[str, str]
    def __init__(self, name: _Optional[str] = ..., timestamp_ns: _Optional[int] = ..., attributes: _Optional[_Mapping[str, str]] = ...) -> None: ...

class SpanAttribute(_message.Message):
    __slots__ = ("key", "string_value", "int_value", "double_value", "bool_value", "json_value")
    KEY_FIELD_NUMBER: _ClassVar[int]
    STRING_VALUE_FIELD_NUMBER: _ClassVar[int]
    INT_VALUE_FIELD_NUMBER: _ClassVar[int]
    DOUBLE_VALUE_FIELD_NUMBER: _ClassVar[int]
    BOOL_VALUE_FIELD_NUMBER: _ClassVar[int]
    JSON_VALUE_FIELD_NUMBER: _ClassVar[int]
    key: str
    string_value: str
    int_value: int
    double_value: float
    bool_value: bool
    json_value: str
    def __init__(self, key: _Optional[str] = ..., string_value: _Optional[str] = ..., int_value: _Optional[int] = ..., double_value: _Optional[float] = ..., bool_value: bool = ..., json_value: _Optional[str] = ...) -> None: ...

class Computation(_message.Message):
    __slots__ = ("trace_id", "span_id", "parent_span_id", "name", "start_time_ns", "end_time_ns", "duration_ns", "status_code", "status_message", "attributes_json", "events_json", "service_name", "service_version", "input_message_ids", "output_message_id", "linked_span_ids", "principal", "entity")
    TRACE_ID_FIELD_NUMBER: _ClassVar[int]
    SPAN_ID_FIELD_NUMBER: _ClassVar[int]
    PARENT_SPAN_ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    START_TIME_NS_FIELD_NUMBER: _ClassVar[int]
    END_TIME_NS_FIELD_NUMBER: _ClassVar[int]
    DURATION_NS_FIELD_NUMBER: _ClassVar[int]
    STATUS_CODE_FIELD_NUMBER: _ClassVar[int]
    STATUS_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    ATTRIBUTES_JSON_FIELD_NUMBER: _ClassVar[int]
    EVENTS_JSON_FIELD_NUMBER: _ClassVar[int]
    SERVICE_NAME_FIELD_NUMBER: _ClassVar[int]
    SERVICE_VERSION_FIELD_NUMBER: _ClassVar[int]
    INPUT_MESSAGE_IDS_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_MESSAGE_ID_FIELD_NUMBER: _ClassVar[int]
    LINKED_SPAN_IDS_FIELD_NUMBER: _ClassVar[int]
    PRINCIPAL_FIELD_NUMBER: _ClassVar[int]
    ENTITY_FIELD_NUMBER: _ClassVar[int]
    trace_id: str
    span_id: str
    parent_span_id: str
    name: str
    start_time_ns: int
    end_time_ns: int
    duration_ns: int
    status_code: SpanStatusCode
    status_message: str
    attributes_json: str
    events_json: str
    service_name: str
    service_version: str
    input_message_ids: _containers.RepeatedScalarFieldContainer[str]
    output_message_id: str
    linked_span_ids: _containers.RepeatedScalarFieldContainer[str]
    principal: str
    entity: str
    def __init__(self, trace_id: _Optional[str] = ..., span_id: _Optional[str] = ..., parent_span_id: _Optional[str] = ..., name: _Optional[str] = ..., start_time_ns: _Optional[int] = ..., end_time_ns: _Optional[int] = ..., duration_ns: _Optional[int] = ..., status_code: _Optional[_Union[SpanStatusCode, str]] = ..., status_message: _Optional[str] = ..., attributes_json: _Optional[str] = ..., events_json: _Optional[str] = ..., service_name: _Optional[str] = ..., service_version: _Optional[str] = ..., input_message_ids: _Optional[_Iterable[str]] = ..., output_message_id: _Optional[str] = ..., linked_span_ids: _Optional[_Iterable[str]] = ..., principal: _Optional[str] = ..., entity: _Optional[str] = ...) -> None: ...

class Computations(_message.Message):
    __slots__ = ("computations", "session_id")
    COMPUTATIONS_FIELD_NUMBER: _ClassVar[int]
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    computations: _containers.RepeatedCompositeFieldContainer[Computation]
    session_id: str
    def __init__(self, computations: _Optional[_Iterable[_Union[Computation, _Mapping]]] = ..., session_id: _Optional[str] = ...) -> None: ...

class ComputationEdge(_message.Message):
    __slots__ = ("parent_span_id", "child_span_id")
    PARENT_SPAN_ID_FIELD_NUMBER: _ClassVar[int]
    CHILD_SPAN_ID_FIELD_NUMBER: _ClassVar[int]
    parent_span_id: str
    child_span_id: str
    def __init__(self, parent_span_id: _Optional[str] = ..., child_span_id: _Optional[str] = ...) -> None: ...

class ComputationMessageEdge(_message.Message):
    __slots__ = ("span_id", "message_id", "edge_type", "message_index")
    SPAN_ID_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_ID_FIELD_NUMBER: _ClassVar[int]
    EDGE_TYPE_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_INDEX_FIELD_NUMBER: _ClassVar[int]
    span_id: str
    message_id: str
    edge_type: ComputationMessageEdgeType
    message_index: int
    def __init__(self, span_id: _Optional[str] = ..., message_id: _Optional[str] = ..., edge_type: _Optional[_Union[ComputationMessageEdgeType, str]] = ..., message_index: _Optional[int] = ...) -> None: ...

class TraceRequest(_message.Message):
    __slots__ = ("trace_id", "start_time_ns", "end_time_ns")
    TRACE_ID_FIELD_NUMBER: _ClassVar[int]
    START_TIME_NS_FIELD_NUMBER: _ClassVar[int]
    END_TIME_NS_FIELD_NUMBER: _ClassVar[int]
    trace_id: str
    start_time_ns: int
    end_time_ns: int
    def __init__(self, trace_id: _Optional[str] = ..., start_time_ns: _Optional[int] = ..., end_time_ns: _Optional[int] = ...) -> None: ...

class SpanRequest(_message.Message):
    __slots__ = ("span_id", "include_children", "include_messages", "session_id")
    SPAN_ID_FIELD_NUMBER: _ClassVar[int]
    INCLUDE_CHILDREN_FIELD_NUMBER: _ClassVar[int]
    INCLUDE_MESSAGES_FIELD_NUMBER: _ClassVar[int]
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    span_id: str
    include_children: bool
    include_messages: bool
    session_id: str
    def __init__(self, span_id: _Optional[str] = ..., include_children: bool = ..., include_messages: bool = ..., session_id: _Optional[str] = ...) -> None: ...

class TraceGraph(_message.Message):
    __slots__ = ("computations", "messages", "computation_edges", "message_edges")
    COMPUTATIONS_FIELD_NUMBER: _ClassVar[int]
    MESSAGES_FIELD_NUMBER: _ClassVar[int]
    COMPUTATION_EDGES_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_EDGES_FIELD_NUMBER: _ClassVar[int]
    computations: _containers.RepeatedCompositeFieldContainer[Computation]
    messages: _containers.RepeatedCompositeFieldContainer[Event]
    computation_edges: _containers.RepeatedCompositeFieldContainer[ComputationEdge]
    message_edges: _containers.RepeatedCompositeFieldContainer[ComputationMessageEdge]
    def __init__(self, computations: _Optional[_Iterable[_Union[Computation, _Mapping]]] = ..., messages: _Optional[_Iterable[_Union[Event, _Mapping]]] = ..., computation_edges: _Optional[_Iterable[_Union[ComputationEdge, _Mapping]]] = ..., message_edges: _Optional[_Iterable[_Union[ComputationMessageEdge, _Mapping]]] = ...) -> None: ...
