from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class SetCredentialsRequest(_message.Message):
    __slots__ = ("entity", "service", "credentials")
    ENTITY_FIELD_NUMBER: _ClassVar[int]
    SERVICE_FIELD_NUMBER: _ClassVar[int]
    CREDENTIALS_FIELD_NUMBER: _ClassVar[int]
    entity: str
    service: str
    credentials: _containers.RepeatedCompositeFieldContainer[Credential]
    def __init__(self, entity: _Optional[str] = ..., service: _Optional[str] = ..., credentials: _Optional[_Iterable[_Union[Credential, _Mapping]]] = ...) -> None: ...

class CredentialsRequest(_message.Message):
    __slots__ = ("entity", "service")
    ENTITY_FIELD_NUMBER: _ClassVar[int]
    SERVICE_FIELD_NUMBER: _ClassVar[int]
    entity: str
    service: str
    def __init__(self, entity: _Optional[str] = ..., service: _Optional[str] = ...) -> None: ...

class Credential(_message.Message):
    __slots__ = ("key", "value")
    KEY_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    key: str
    value: str
    def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...

class Credentials(_message.Message):
    __slots__ = ("credentials",)
    CREDENTIALS_FIELD_NUMBER: _ClassVar[int]
    credentials: _containers.RepeatedCompositeFieldContainer[Credential]
    def __init__(self, credentials: _Optional[_Iterable[_Union[Credential, _Mapping]]] = ...) -> None: ...

class CredentialResponse(_message.Message):
    __slots__ = ("response",)
    RESPONSE_FIELD_NUMBER: _ClassVar[int]
    response: str
    def __init__(self, response: _Optional[str] = ...) -> None: ...
