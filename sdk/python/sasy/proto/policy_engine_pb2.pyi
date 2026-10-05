from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class DenialReasonType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    DENIAL_REASON_UNSPECIFIED: _ClassVar[DenialReasonType]
    NOT_AUTHENTICATED: _ClassVar[DenialReasonType]
    DENYLISTED: _ClassVar[DenialReasonType]
    NOT_ALLOWLISTED: _ClassVar[DenialReasonType]
    SYNC_TIMEOUT: _ClassVar[DenialReasonType]
    ASK: _ClassVar[DenialReasonType]
DENIAL_REASON_UNSPECIFIED: DenialReasonType
NOT_AUTHENTICATED: DenialReasonType
DENYLISTED: DenialReasonType
NOT_ALLOWLISTED: DenialReasonType
SYNC_TIMEOUT: DenialReasonType
ASK: DenialReasonType

class AuthorizationRequest(_message.Message):
    __slots__ = ("current_node_ids", "actions", "entity", "roles", "session_id", "principal")
    CURRENT_NODE_IDS_FIELD_NUMBER: _ClassVar[int]
    ACTIONS_FIELD_NUMBER: _ClassVar[int]
    ENTITY_FIELD_NUMBER: _ClassVar[int]
    ROLES_FIELD_NUMBER: _ClassVar[int]
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    PRINCIPAL_FIELD_NUMBER: _ClassVar[int]
    current_node_ids: _containers.RepeatedScalarFieldContainer[str]
    actions: _containers.RepeatedCompositeFieldContainer[Action]
    entity: str
    roles: _containers.RepeatedScalarFieldContainer[str]
    session_id: str
    principal: str
    def __init__(self, current_node_ids: _Optional[_Iterable[str]] = ..., actions: _Optional[_Iterable[_Union[Action, _Mapping]]] = ..., entity: _Optional[str] = ..., roles: _Optional[_Iterable[str]] = ..., session_id: _Optional[str] = ..., principal: _Optional[str] = ...) -> None: ...

class Action(_message.Message):
    __slots__ = ("http_request", "tool_call", "send_message", "metadata")
    HTTP_REQUEST_FIELD_NUMBER: _ClassVar[int]
    TOOL_CALL_FIELD_NUMBER: _ClassVar[int]
    SEND_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    http_request: HttpRequestAction
    tool_call: ToolCallAction
    send_message: SendMessageAction
    metadata: _containers.RepeatedCompositeFieldContainer[PolicyMetadataFact]
    def __init__(self, http_request: _Optional[_Union[HttpRequestAction, _Mapping]] = ..., tool_call: _Optional[_Union[ToolCallAction, _Mapping]] = ..., send_message: _Optional[_Union[SendMessageAction, _Mapping]] = ..., metadata: _Optional[_Iterable[_Union[PolicyMetadataFact, _Mapping]]] = ...) -> None: ...

class HttpRequestAction(_message.Message):
    __slots__ = ("url", "body", "headers")
    URL_FIELD_NUMBER: _ClassVar[int]
    BODY_FIELD_NUMBER: _ClassVar[int]
    HEADERS_FIELD_NUMBER: _ClassVar[int]
    url: str
    body: str
    headers: _containers.RepeatedCompositeFieldContainer[Header]
    def __init__(self, url: _Optional[str] = ..., body: _Optional[str] = ..., headers: _Optional[_Iterable[_Union[Header, _Mapping]]] = ...) -> None: ...

class Header(_message.Message):
    __slots__ = ("key", "value")
    KEY_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    key: str
    value: str
    def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...

class ToolCallAction(_message.Message):
    __slots__ = ("fn_name", "args")
    FN_NAME_FIELD_NUMBER: _ClassVar[int]
    ARGS_FIELD_NUMBER: _ClassVar[int]
    fn_name: str
    args: str
    def __init__(self, fn_name: _Optional[str] = ..., args: _Optional[str] = ...) -> None: ...

class SendMessageAction(_message.Message):
    __slots__ = ("content", "agent", "agent_role", "tool_calls", "entity")
    CONTENT_FIELD_NUMBER: _ClassVar[int]
    AGENT_FIELD_NUMBER: _ClassVar[int]
    AGENT_ROLE_FIELD_NUMBER: _ClassVar[int]
    TOOL_CALLS_FIELD_NUMBER: _ClassVar[int]
    ENTITY_FIELD_NUMBER: _ClassVar[int]
    content: str
    agent: str
    agent_role: str
    tool_calls: _containers.RepeatedCompositeFieldContainer[ToolCall]
    entity: str
    def __init__(self, content: _Optional[str] = ..., agent: _Optional[str] = ..., agent_role: _Optional[str] = ..., tool_calls: _Optional[_Iterable[_Union[ToolCall, _Mapping]]] = ..., entity: _Optional[str] = ...) -> None: ...

class ToolCall(_message.Message):
    __slots__ = ("fn_name", "args")
    FN_NAME_FIELD_NUMBER: _ClassVar[int]
    ARGS_FIELD_NUMBER: _ClassVar[int]
    fn_name: str
    args: str
    def __init__(self, fn_name: _Optional[str] = ..., args: _Optional[str] = ...) -> None: ...

class AuthorizationResponse(_message.Message):
    __slots__ = ("results", "timing")
    RESULTS_FIELD_NUMBER: _ClassVar[int]
    TIMING_FIELD_NUMBER: _ClassVar[int]
    results: _containers.RepeatedCompositeFieldContainer[ActionResult]
    timing: PerformanceTiming
    def __init__(self, results: _Optional[_Iterable[_Union[ActionResult, _Mapping]]] = ..., timing: _Optional[_Union[PerformanceTiming, _Mapping]] = ...) -> None: ...

class PerformanceTiming(_message.Message):
    __slots__ = ("total_us", "sync_wait_us", "eval_us", "backend", "graph_nodes", "graph_edges")
    TOTAL_US_FIELD_NUMBER: _ClassVar[int]
    SYNC_WAIT_US_FIELD_NUMBER: _ClassVar[int]
    EVAL_US_FIELD_NUMBER: _ClassVar[int]
    BACKEND_FIELD_NUMBER: _ClassVar[int]
    GRAPH_NODES_FIELD_NUMBER: _ClassVar[int]
    GRAPH_EDGES_FIELD_NUMBER: _ClassVar[int]
    total_us: int
    sync_wait_us: int
    eval_us: int
    backend: str
    graph_nodes: int
    graph_edges: int
    def __init__(self, total_us: _Optional[int] = ..., sync_wait_us: _Optional[int] = ..., eval_us: _Optional[int] = ..., backend: _Optional[str] = ..., graph_nodes: _Optional[int] = ..., graph_edges: _Optional[int] = ...) -> None: ...

class ActionResult(_message.Message):
    __slots__ = ("index", "authorized", "trace", "transform_ids", "deny_if_unauthorized")
    INDEX_FIELD_NUMBER: _ClassVar[int]
    AUTHORIZED_FIELD_NUMBER: _ClassVar[int]
    TRACE_FIELD_NUMBER: _ClassVar[int]
    TRANSFORM_IDS_FIELD_NUMBER: _ClassVar[int]
    DENY_IF_UNAUTHORIZED_FIELD_NUMBER: _ClassVar[int]
    index: int
    authorized: bool
    trace: DenialTrace
    transform_ids: _containers.RepeatedScalarFieldContainer[str]
    deny_if_unauthorized: bool
    def __init__(self, index: _Optional[int] = ..., authorized: bool = ..., trace: _Optional[_Union[DenialTrace, _Mapping]] = ..., transform_ids: _Optional[_Iterable[str]] = ..., deny_if_unauthorized: bool = ...) -> None: ...

class DenialTrace(_message.Message):
    __slots__ = ("action_description", "reasons", "suggested_fixes", "allow_routes")
    ACTION_DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    REASONS_FIELD_NUMBER: _ClassVar[int]
    SUGGESTED_FIXES_FIELD_NUMBER: _ClassVar[int]
    ALLOW_ROUTES_FIELD_NUMBER: _ClassVar[int]
    action_description: str
    reasons: _containers.RepeatedCompositeFieldContainer[DenialReason]
    suggested_fixes: _containers.RepeatedScalarFieldContainer[str]
    allow_routes: _containers.RepeatedCompositeFieldContainer[AllowRouteDiagnostic]
    def __init__(self, action_description: _Optional[str] = ..., reasons: _Optional[_Iterable[_Union[DenialReason, _Mapping]]] = ..., suggested_fixes: _Optional[_Iterable[str]] = ..., allow_routes: _Optional[_Iterable[_Union[AllowRouteDiagnostic, _Mapping]]] = ...) -> None: ...

class AllowRouteDiagnostic(_message.Message):
    __slots__ = ("rule_id", "status", "details", "suggestions", "source_location")
    RULE_ID_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    DETAILS_FIELD_NUMBER: _ClassVar[int]
    SUGGESTIONS_FIELD_NUMBER: _ClassVar[int]
    SOURCE_LOCATION_FIELD_NUMBER: _ClassVar[int]
    rule_id: str
    status: str
    details: str
    suggestions: _containers.RepeatedScalarFieldContainer[str]
    source_location: str
    def __init__(self, rule_id: _Optional[str] = ..., status: _Optional[str] = ..., details: _Optional[str] = ..., suggestions: _Optional[_Iterable[str]] = ..., source_location: _Optional[str] = ...) -> None: ...

class DenialReason(_message.Message):
    __slots__ = ("reason_type", "details", "source_location")
    REASON_TYPE_FIELD_NUMBER: _ClassVar[int]
    DETAILS_FIELD_NUMBER: _ClassVar[int]
    SOURCE_LOCATION_FIELD_NUMBER: _ClassVar[int]
    reason_type: DenialReasonType
    details: str
    source_location: str
    def __init__(self, reason_type: _Optional[_Union[DenialReasonType, str]] = ..., details: _Optional[str] = ..., source_location: _Optional[str] = ...) -> None: ...

class SyncStatusRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class SyncStatusResponse(_message.Message):
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

class HealthRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class HealthResponse(_message.Message):
    __slots__ = ("healthy", "message")
    HEALTHY_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    healthy: bool
    message: str
    def __init__(self, healthy: bool = ..., message: _Optional[str] = ...) -> None: ...

class SetPolicyRequest(_message.Message):
    __slots__ = ("policy_source", "functor_source", "backend", "scope", "policy_metadata", "bind_content_hash", "bind_profile_name")
    POLICY_SOURCE_FIELD_NUMBER: _ClassVar[int]
    FUNCTOR_SOURCE_FIELD_NUMBER: _ClassVar[int]
    BACKEND_FIELD_NUMBER: _ClassVar[int]
    SCOPE_FIELD_NUMBER: _ClassVar[int]
    POLICY_METADATA_FIELD_NUMBER: _ClassVar[int]
    BIND_CONTENT_HASH_FIELD_NUMBER: _ClassVar[int]
    BIND_PROFILE_NAME_FIELD_NUMBER: _ClassVar[int]
    policy_source: str
    functor_source: str
    backend: str
    scope: PolicyScope
    policy_metadata: _containers.RepeatedCompositeFieldContainer[PolicyMetadataFact]
    bind_content_hash: str
    bind_profile_name: str
    def __init__(self, policy_source: _Optional[str] = ..., functor_source: _Optional[str] = ..., backend: _Optional[str] = ..., scope: _Optional[_Union[PolicyScope, _Mapping]] = ..., policy_metadata: _Optional[_Iterable[_Union[PolicyMetadataFact, _Mapping]]] = ..., bind_content_hash: _Optional[str] = ..., bind_profile_name: _Optional[str] = ...) -> None: ...

class PolicyMetadataFact(_message.Message):
    __slots__ = ("rel", "a", "b")
    REL_FIELD_NUMBER: _ClassVar[int]
    A_FIELD_NUMBER: _ClassVar[int]
    B_FIELD_NUMBER: _ClassVar[int]
    rel: str
    a: str
    b: str
    def __init__(self, rel: _Optional[str] = ..., a: _Optional[str] = ..., b: _Optional[str] = ...) -> None: ...

class UpdatePolicyMetadataRequest(_message.Message):
    __slots__ = ("session_id", "facts")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    FACTS_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    facts: _containers.RepeatedCompositeFieldContainer[PolicyMetadataFact]
    def __init__(self, session_id: _Optional[str] = ..., facts: _Optional[_Iterable[_Union[PolicyMetadataFact, _Mapping]]] = ...) -> None: ...

class UpdatePolicyMetadataResponse(_message.Message):
    __slots__ = ("accepted",)
    ACCEPTED_FIELD_NUMBER: _ClassVar[int]
    accepted: bool
    def __init__(self, accepted: bool = ...) -> None: ...

class PolicyScope(_message.Message):
    __slots__ = ("session", "default", "force")
    SESSION_FIELD_NUMBER: _ClassVar[int]
    DEFAULT_FIELD_NUMBER: _ClassVar[int]
    FORCE_FIELD_NUMBER: _ClassVar[int]
    session: SessionTarget
    default: DefaultTarget
    force: ForceTarget
    def __init__(self, session: _Optional[_Union[SessionTarget, _Mapping]] = ..., default: _Optional[_Union[DefaultTarget, _Mapping]] = ..., force: _Optional[_Union[ForceTarget, _Mapping]] = ...) -> None: ...

class SessionTarget(_message.Message):
    __slots__ = ("session_id",)
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    def __init__(self, session_id: _Optional[str] = ...) -> None: ...

class DefaultTarget(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class ForceTarget(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class SetPolicyResponse(_message.Message):
    __slots__ = ("accepted", "message", "error_output", "policy_id")
    ACCEPTED_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    ERROR_OUTPUT_FIELD_NUMBER: _ClassVar[int]
    POLICY_ID_FIELD_NUMBER: _ClassVar[int]
    accepted: bool
    message: str
    error_output: str
    policy_id: str
    def __init__(self, accepted: bool = ..., message: _Optional[str] = ..., error_output: _Optional[str] = ..., policy_id: _Optional[str] = ...) -> None: ...

class EvaluatorStatusRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class EvaluatorStatusResponse(_message.Message):
    __slots__ = ("backend", "alive", "policy_path")
    BACKEND_FIELD_NUMBER: _ClassVar[int]
    ALIVE_FIELD_NUMBER: _ClassVar[int]
    POLICY_PATH_FIELD_NUMBER: _ClassVar[int]
    backend: str
    alive: bool
    policy_path: str
    def __init__(self, backend: _Optional[str] = ..., alive: bool = ..., policy_path: _Optional[str] = ...) -> None: ...

class ValidatePolicyRequest(_message.Message):
    __slots__ = ("policy_source", "run_analyses", "extra_reach_targets")
    POLICY_SOURCE_FIELD_NUMBER: _ClassVar[int]
    RUN_ANALYSES_FIELD_NUMBER: _ClassVar[int]
    EXTRA_REACH_TARGETS_FIELD_NUMBER: _ClassVar[int]
    policy_source: str
    run_analyses: bool
    extra_reach_targets: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, policy_source: _Optional[str] = ..., run_analyses: bool = ..., extra_reach_targets: _Optional[_Iterable[str]] = ...) -> None: ...

class ValidatePolicyResponse(_message.Message):
    __slots__ = ("valid", "error_output", "desugared_source", "analyses")
    VALID_FIELD_NUMBER: _ClassVar[int]
    ERROR_OUTPUT_FIELD_NUMBER: _ClassVar[int]
    DESUGARED_SOURCE_FIELD_NUMBER: _ClassVar[int]
    ANALYSES_FIELD_NUMBER: _ClassVar[int]
    valid: bool
    error_output: str
    desugared_source: str
    analyses: AnalysisReport
    def __init__(self, valid: bool = ..., error_output: _Optional[str] = ..., desugared_source: _Optional[str] = ..., analyses: _Optional[_Union[AnalysisReport, _Mapping]] = ...) -> None: ...

class AnalysisReport(_message.Message):
    __slots__ = ("contradictions", "redundancies", "reachability", "broad_rules")
    CONTRADICTIONS_FIELD_NUMBER: _ClassVar[int]
    REDUNDANCIES_FIELD_NUMBER: _ClassVar[int]
    REACHABILITY_FIELD_NUMBER: _ClassVar[int]
    BROAD_RULES_FIELD_NUMBER: _ClassVar[int]
    contradictions: _containers.RepeatedCompositeFieldContainer[ContradictionFinding]
    redundancies: _containers.RepeatedCompositeFieldContainer[RedundancyFinding]
    reachability: _containers.RepeatedCompositeFieldContainer[ReachabilityFinding]
    broad_rules: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, contradictions: _Optional[_Iterable[_Union[ContradictionFinding, _Mapping]]] = ..., redundancies: _Optional[_Iterable[_Union[RedundancyFinding, _Mapping]]] = ..., reachability: _Optional[_Iterable[_Union[ReachabilityFinding, _Mapping]]] = ..., broad_rules: _Optional[_Iterable[str]] = ...) -> None: ...

class ContradictionFinding(_message.Message):
    __slots__ = ("category", "allow_location", "deny_location", "message", "allow_body", "deny_body")
    CATEGORY_FIELD_NUMBER: _ClassVar[int]
    ALLOW_LOCATION_FIELD_NUMBER: _ClassVar[int]
    DENY_LOCATION_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    ALLOW_BODY_FIELD_NUMBER: _ClassVar[int]
    DENY_BODY_FIELD_NUMBER: _ClassVar[int]
    category: str
    allow_location: str
    deny_location: str
    message: str
    allow_body: str
    deny_body: str
    def __init__(self, category: _Optional[str] = ..., allow_location: _Optional[str] = ..., deny_location: _Optional[str] = ..., message: _Optional[str] = ..., allow_body: _Optional[str] = ..., deny_body: _Optional[str] = ...) -> None: ...

class RedundancyFinding(_message.Message):
    __slots__ = ("head_relation", "redundant_location", "covered_by_location", "redundant_body", "covered_by_body")
    HEAD_RELATION_FIELD_NUMBER: _ClassVar[int]
    REDUNDANT_LOCATION_FIELD_NUMBER: _ClassVar[int]
    COVERED_BY_LOCATION_FIELD_NUMBER: _ClassVar[int]
    REDUNDANT_BODY_FIELD_NUMBER: _ClassVar[int]
    COVERED_BY_BODY_FIELD_NUMBER: _ClassVar[int]
    head_relation: str
    redundant_location: str
    covered_by_location: str
    redundant_body: str
    covered_by_body: str
    def __init__(self, head_relation: _Optional[str] = ..., redundant_location: _Optional[str] = ..., covered_by_location: _Optional[str] = ..., redundant_body: _Optional[str] = ..., covered_by_body: _Optional[str] = ...) -> None: ...

class EndSessionRequest(_message.Message):
    __slots__ = ("session_id",)
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    def __init__(self, session_id: _Optional[str] = ...) -> None: ...

class EndSessionResponse(_message.Message):
    __slots__ = ("was_active",)
    WAS_ACTIVE_FIELD_NUMBER: _ClassVar[int]
    was_active: bool
    def __init__(self, was_active: bool = ...) -> None: ...

class ReachabilityFinding(_message.Message):
    __slots__ = ("target", "disjuncts", "opaque", "pruned")
    TARGET_FIELD_NUMBER: _ClassVar[int]
    DISJUNCTS_FIELD_NUMBER: _ClassVar[int]
    OPAQUE_FIELD_NUMBER: _ClassVar[int]
    PRUNED_FIELD_NUMBER: _ClassVar[int]
    target: str
    disjuncts: _containers.RepeatedScalarFieldContainer[str]
    opaque: _containers.RepeatedScalarFieldContainer[str]
    pruned: int
    def __init__(self, target: _Optional[str] = ..., disjuncts: _Optional[_Iterable[str]] = ..., opaque: _Optional[_Iterable[str]] = ..., pruned: _Optional[int] = ...) -> None: ...
