"""Qualified text artifacts on ADK artifact services as versioned message transport."""
from __future__ import annotations

import copy
import hashlib
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
from threading import RLock

import grpc

from sasy.observability._snapshots import VERSION_ID_PREFIX

from . import adk_state as resources
from .session import is_session_active

_installed = False
_lock = RLock()
_instrumented: dict[type, dict] = {}
_presentations: ContextVar[dict | None] = ContextVar("sasy_adk_artifact_presentations", default=None)
# Producers this session recorded, stored with the artifact by ADK itself and
# re-verified here before a load in another process may depend on them.
_METADATA_KEY = "sasy_provenance"
_METADATA_FORMAT = "sasy.adk.artifact.provenance/1"
# The one gRPC status that is a decision about a claimed reference: the server
# holds no such version of that origin in this session. Every other status
# means the question was not answered and stays fatal, as in langchain.py.
_REFUSAL = grpc.StatusCode.INVALID_ARGUMENT
_OBSERVED = ("save_artifact", "load_artifact", "list_artifact_keys")
_RESTRICTED = ("delete_artifact", "list_versions", "list_artifact_versions", "get_artifact_version")


@dataclass
class Store:
    layouts: dict[tuple, dict[str, tuple]] = field(default_factory=dict)
    poisoned: set[tuple] = field(default_factory=set)
    busy: bool = False
    publications: dict[tuple, tuple[str, list[str]]] = field(default_factory=dict)
    claims: dict[tuple, dict] = field(default_factory=dict)


def _sdk():
    from . import adk
    return adk


def _part(value):
    from google.genai import types
    part = value if isinstance(value, types.Part) else types.Part.model_validate(value)
    if set(part.model_dump(exclude_none=True)) != {"text"}:
        resources._fail("Only plain-text ADK artifacts are qualified")
    return part.model_copy(deep=True)


def _store(ledger):
    store = getattr(ledger, "artifacts", None)
    if store is None:
        store = Store()
        ledger.artifacts = store
    return store


def _scopes(app_name, user_id, session_id):
    return [(app_name, user_id, "user"), (app_name, user_id, session_id)]


def _originals(service):
    entry = _instrumented.get(type(service))
    if entry is None:
        resources._fail("Artifact services must be qualified before use")
    return entry


async def _layout(service, app_name, user_id, session_id):
    """Stored versions per scope, read through the service interface alone."""
    originals = _originals(service)
    keys, listed = originals["list_artifact_keys"], originals["list_versions"]
    shared = set(await keys(service, app_name=app_name, user_id=user_id, session_id=None))
    names = {(app_name, user_id, "user"): sorted(shared),
             (app_name, user_id, session_id): [name for name in
                await keys(service, app_name=app_name, user_id=user_id, session_id=session_id)
                if name not in shared]}
    layout = {}
    for scope, scoped in names.items():
        layout[scope] = {name: tuple(await listed(service, app_name=app_name, user_id=user_id,
            filename=name, session_id=session_id)) for name in scoped}
    return layout


def _unfinished_save(store, scope, previous, current):
    """Whether a save this session began but never finished recording explains the change.

    The version stays poisoned either way, so the run still stops; naming the
    interrupted save keeps the reason accurate.
    """
    added = {(name, version) for name, versions in current.items()
             for version in versions if version not in previous.get(name, ())}
    lost = any(name not in current or set(versions) - set(current[name])
               for name, versions in previous.items())
    return bool(added) and not lost and all(("artifact", *scope, name, version) in store.poisoned
                                            for name, version in added)


async def _reconcile(service, app_name, user_id, session_id, store):
    """Fail unless the stored versions are still the ones this operation captured.

    A writer outside this session can store a version while the call is in
    flight, so a result read after that point describes versions the recorded
    dependencies never named.
    """
    current = await _layout(service, app_name, user_id, session_id)
    if any(current[scope] != store.layouts[scope] for scope in current):
        resources._fail("Artifact versions changed without an observed save")


@asynccontextmanager
async def _operation(service, app_name, user_id, session_id, *, concurrent=False):
    from .adk_callbacks import in_callback
    frame = resources._frame.get()
    if frame is None:
        resources._fail("Artifact access lacks an observed consumer")
    owner = frame.resources
    owner.check()
    if in_callback():
        resources._fail("Artifact access in application callbacks requires additional instrumentation")
    if service is not owner.runner.artifact_service:
        resources._fail("Only the runner's own artifact service is qualified")
    if (app_name, user_id, session_id) != (owner.runner.app_name, owner.user_id, owner.session_id):
        resources._fail("Artifact access must use the current ADK app/user/session scope")
    ledger = owner.claim(service, artifact=concurrent)
    store = _store(ledger)
    if concurrent:
        yield frame, owner, ledger, store
        owner.check()
        return
    if store.busy:
        resources._fail("Concurrent artifact operations require additional qualification")
    store.busy = True
    try:
        layout = await _layout(service, app_name, user_id, session_id)
        owner.check()
        for scope, scoped in layout.items():
            previous = store.layouts.setdefault(scope, scoped)
            if scoped != previous:
                if _unfinished_save(store, scope, previous, scoped):
                    resources._fail("Artifact save did not complete observation")
                resources._fail("Artifact versions changed without an observed save")
        yield frame, owner, ledger, store
        owner.check()
    finally:
        store.busy = False


def _scope(app_name, user_id, session_id, filename):
    return (app_name, user_id, "user" if filename.startswith("user:") else session_id)


def _key(app_name, user_id, session_id, filename, version):
    scope = "user" if filename.startswith("user:") else session_id
    return ("artifact", app_name, user_id, scope, filename, version)


def _record(key, value):
    """The exact text the producing snapshot recorded for this artifact version."""
    return _sdk()._json({"adk_resource": list(key), "provenance": "observed production", "value": value})


def validate_service(service):
    from google.adk.artifacts.base_artifact_service import BaseArtifactService
    if service is None:
        return
    if not isinstance(service, BaseArtifactService):
        resources._fail("Artifact services must implement ADK's BaseArtifactService")
    instrument_service(type(service))


def validate_tool(tool):
    from google.adk.tools.load_artifacts_tool import LoadArtifactsTool
    if type(tool) is not LoadArtifactsTool:
        return False
    for name in ("run_async", "process_llm_request", "_append_artifacts_to_llm_request", "_get_declaration"):
        if getattr(getattr(tool, name), "__func__", None) is not getattr(LoadArtifactsTool, name):
            resources._fail("LoadArtifactsTool overrides require explicit instrumentation")
    if tool._process_artifact is not None or tool._enable_spreadsheet_parsing:
        resources._fail("Artifact transformations require explicit instrumentation")
    return True


async def _observed_load(original, self, *, app_name, user_id, filename, session_id=None, version=None):
    sdk = _sdk()
    if (not is_session_active() or sdk._active.get() is None):
        return await original(self, app_name=app_name, user_id=user_id, filename=filename, session_id=session_id, version=version)
    await resources.flush_reads_async()
    if concurrent_file_service(self):
        return await _file_load(original, self, app_name=app_name, user_id=user_id, filename=filename,
            session_id=session_id, version=version)
    async with _operation(self, app_name, user_id, session_id) as (frame, owner, ledger, store):
        stored = store.layouts[_scope(app_name, user_id, session_id, filename)].get(filename, ())
        index = (stored[-1] if stored else -1) if version is None else version
        if type(index) is not int or (version is not None and index < 0):
            resources._fail("Artifact version must identify an existing nonnegative version")
        key = _key(app_name, user_id, session_id, filename, index)
        if key in store.poisoned:
            resources._fail("Artifact save did not complete observation")
        part = await original(self, app_name=app_name, user_id=user_id, filename=filename, session_id=session_id, version=version)
        await _reconcile(self, app_name, user_id, session_id, store)
        frozen = _part(part) if part is not None else None
        value = {"present": frozen is not None, "text": frozen.text if frozen else None}
        if frozen is not None and key not in ledger.versions:
            await _adopt(owner, ledger, key, value, self, app_name, user_id, filename, session_id, index)
        ids = await owner.snapshot(ledger, key, value)
        presentation = _presentations.get()
        if presentation is None:
            resources._add_reads(ids)
        elif frozen is None:
            presentation["missing"] = (filename, ids)
        else:
            missing = presentation.pop("missing", None)
            parents = [*missing[1], *ids] if missing and filename == "user:" + missing[0] else ids
            presentation[id(frozen)] = (frozen, parents)
        return frozen


async def _observed_save(original, self, *, app_name, user_id, filename, artifact, session_id=None, custom_metadata=None):
    sdk = _sdk()
    if (not is_session_active() or sdk._active.get() is None):
        return await original(self, app_name=app_name, user_id=user_id, filename=filename,
            artifact=artifact, session_id=session_id, custom_metadata=custom_metadata)
    await resources.flush_reads_async()
    if concurrent_file_service(self):
        return await _file_save(original, self, app_name=app_name, user_id=user_id, filename=filename,
            artifact=artifact, session_id=session_id, custom_metadata=custom_metadata)
    async with _operation(self, app_name, user_id, session_id) as (frame, owner, ledger, store):
        if frame.kind != "tool" or not sdk._current_input_ids.get():
            resources._fail("Artifact saves require an observed tool producer")
        if custom_metadata:
            resources._fail("Artifact custom metadata requires additional instrumentation")
        frozen = _part(artifact)
        catalogs = {}
        for scope in _scopes(app_name, user_id, session_id):
            names = sorted(store.layouts[scope])
            catalogs[scope] = (names, await owner.snapshot(ledger, ("artifact-list", *scope), names))
        stored = store.layouts[_scope(app_name, user_id, session_id, filename)].get(filename, ())
        index = stored[-1] + 1 if stored else 0
        key = _key(app_name, user_id, session_id, filename, index)
        store.poisoned.add(key)
        value = {"present": bool(frozen.text), "text": frozen.text if frozen.text else None}
        # The producing nodes travel with the artifact, so they must exist
        # before ADK stores the version they describe.
        ids = await owner.snapshot(ledger, key, value,
            inputs=sdk._current_input_ids.get(), agent=frame.invocation.agent.name)
        # A version is bound to the alias its event was first recorded under,
        # so the alias travels too.
        saved = await original(self, app_name=app_name, user_id=user_id, filename=filename,
            artifact=frozen, session_id=session_id,
            custom_metadata={_METADATA_KEY: {"format": _METADATA_FORMAT, "nodes": list(ids),
                "origin": owner.state.snapshots[ids[0]].id,
                "entity": resources.get_current_entity() or "",
                "agent": frame.invocation.agent.name,
                "digest": hashlib.sha256(_record(key, value).encode("utf-8")).hexdigest()}})
        owner.check()
        if saved != index:
            resources._fail("Artifact storage changed during save")
        store.layouts.update(await _layout(self, app_name, user_id, session_id))
        owner.check()
        frame.artifact_writes.append((ledger, key, ledger.versions[key]))
        for scope, (previous, previous_ids) in catalogs.items():
            names = sorted(store.layouts[scope])
            if names != previous:
                await owner.snapshot(ledger, ("artifact-list", *scope), names,
                    inputs=[*sdk._current_input_ids.get(), *previous_ids], agent=frame.invocation.agent.name)
                catalog_key = ("artifact-list", *scope)
                frame.artifact_writes.append((ledger, catalog_key, ledger.versions[catalog_key]))
        store.poisoned.remove(key)
        return saved


async def _observed_list(original, self, *, app_name, user_id, session_id=None):
    sdk = _sdk()
    if (not is_session_active() or sdk._active.get() is None):
        return await original(self, app_name=app_name, user_id=user_id, session_id=session_id)
    await resources.flush_reads_async()
    if concurrent_file_service(self):
        return await _file_list(original, self, app_name=app_name, user_id=user_id, session_id=session_id)
    async with _operation(self, app_name, user_id, session_id) as (frame, owner, ledger, store):
        names = await original(self, app_name=app_name, user_id=user_id, session_id=session_id)
        owner.check()
        scopes = _scopes(app_name, user_id, session_id)
        if sorted(names) != sorted(name for scope in scopes for name in store.layouts[scope]):
            resources._fail("Artifact listing changed without an observed save")
        # Listing reveals names, never content from unread files.
        for scope in scopes:
            ids = await owner.snapshot(ledger, ("artifact-list", *scope), sorted(store.layouts[scope]))
            resources._add_reads(ids)
        return list(names)


async def _restricted(original, self, *args, **kwargs):
    if (is_session_active() and _sdk()._active.get() is not None):
        resources._fail("Artifact deletion and version metadata require additional instrumentation")
    return await original(self, *args, **kwargs)


async def _adopt(owner, ledger, key, value, service, app_name, user_id, filename, session_id, index):
    """Depend on a producer recorded elsewhere only when its own claim verifies here.

    The artifact carries the producing node IDs and a digest of the exact text
    that producer recorded. Both are untrusted: the digest must match the text
    rebuilt from what this load actually returned, and the observation service
    must still hold that text under those IDs in this SASY session. Anything
    else leaves the content an unattributed external input.
    """
    from sasy.proto.observability_pb2 import Event, EventSnapshot
    sdk = _sdk()
    metadata = _originals(service).get("get_artifact_version")
    if metadata is None:
        return
    # A storage failure answers nothing about the claim, so it stays fatal, as
    # for the observation service below. A service without the method, or one
    # that holds no metadata for this version, is an answer: no claim, so the
    # content enters as an unattributed external input.
    version = await metadata(service, app_name=app_name, user_id=user_id,
        filename=filename, session_id=session_id, version=index)
    claim = (getattr(version, "custom_metadata", None) or {}).get(_METADATA_KEY)
    if not isinstance(claim, dict) or claim.get("format") != _METADATA_FORMAT:
        return
    nodes, agent, digest, origin = claim.get("nodes"), claim.get("agent"), claim.get("digest"), claim.get("origin")
    if (not isinstance(nodes, list) or len(nodes) != 1 or not isinstance(nodes[0], str)
            or not nodes[0].startswith(VERSION_ID_PREFIX) or not isinstance(agent, str) or not isinstance(digest, str)
            or not isinstance(origin, str) or not origin or not isinstance(claim.get("entity"), str)):
        return
    # The entity label is part of a version's content, and the SDK stamps the
    # current one on an event that has none, so a version recorded without an
    # entity can only be replayed where none is set.
    entity = claim["entity"]
    if not entity and resources.get_current_entity():
        return
    recorded = _record(key, value)
    if digest != hashlib.sha256(recorded.encode("utf-8")).hexdigest():
        return
    # The rebuilt event must be the whole event the producer recorded, metadata
    # included: the engine decides on its content hash.
    event = Event(id=origin, agent=agent, role=sdk.Role.AGENT, text=recorded,
                  metadata=sdk._text_metadata(recorded))
    if entity:
        event.entity = entity
    try:
        resolved = await sdk.observation.resolve_events_async(
            [EventSnapshot(event=event, base_id=nodes[0], reuse_dependencies=True)])
    except grpc.RpcError as error:
        # Only the server's refusal means the claim does not hold. An outage
        # decides nothing, and dropping a good claim would lose real ancestry.
        if error.code() != _REFUSAL:
            raise
        return
    if resolved != nodes:
        return
    owner.check()
    ledger.versions[key] = resources.Version(sdk._record_json(value), nodes={owner.state.session: list(nodes)},
        producer_graph=owner.state.session, snapshots={nodes[0]: event}, value=value)


def _wrap(original, behavior):
    @wraps(original)
    async def method(self, *args, **kwargs):
        return await behavior(original, self, *args, **kwargs)
    setattr(method, "__sasy_original__", original)  # noqa: B010
    return method


def instrument_service(cls):
    """Observe one artifact service implementation through its ADK interface."""
    with _lock:
        if cls in _instrumented:
            return
        originals = {}
        for name in (*_OBSERVED, *_RESTRICTED):
            method = getattr(cls, name, None)
            if method is not None:
                originals[name] = getattr(method, "__sasy_original__", method)
        if any(name not in originals for name in (*_OBSERVED, "list_versions")):
            resources._fail("Artifact services must implement ADK's BaseArtifactService")
        _instrumented[cls] = originals
        for name, behavior in (("save_artifact", _observed_save), ("load_artifact", _observed_load),
                               ("list_artifact_keys", _observed_list)):
            setattr(cls, name, _wrap(originals[name], behavior))
        for name in _RESTRICTED:
            if name in originals:
                setattr(cls, name, _wrap(originals[name], _restricted))


def install():
    global _installed
    if _installed:
        return
    from google.adk.artifacts.file_artifact_service import FileArtifactService
    from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
    from google.adk.tools.load_artifacts_tool import LoadArtifactsTool

    from . import adk_artifact_files, adk_otel
    adk_artifact_files.install()
    for backend in (InMemoryArtifactService, FileArtifactService):
        instrument_service(backend)

    run = LoadArtifactsTool.run_async
    @wraps(run)
    async def guarded_load(self, *, args, tool_context):
        if not is_session_active():
            return await run(self, args=args, tool_context=tool_context)
        sdk = _sdk()
        state, agent, origin = sdk._origin(tool_context)
        validate_tool(self)
        sdk._validate_actions(tool_context.actions)
        if tool_context.actions.transfer_to_agent:
            resources._fail("Transfer action was set before artifact dispatch")
        arguments = copy.deepcopy(args)
        arguments.setdefault("artifact_names", [])
        serialized = sdk._json(arguments)
        with adk_otel.operation("tool", agent, resources.tool_inputs(origin.ids),
                call_id=tool_context.function_call_id, tool=self.name) as telemetry:
            verdict = await sdk.monitor.check_tool_call_async(self.name, serialized, resources.tool_inputs(origin.ids),
                metadata=[("adk_agent", agent, ""), ("framework", "adk", "")])
            telemetry.decision(verdict)
            if verdict.authorized and not verdict.transform_ids:
                result = await run(self, args=arguments, tool_context=tool_context)
            else:
                result = sdk._blocked(verdict, self.name)
            state.artifact_loads[(agent, tool_context.function_call_id)] = sdk._record_json(result)
            successful = verdict.authorized and not verdict.transform_ids and "error" not in result
            await sdk._record_tool(self, tool_context, result,
                sdk.Tool(name=self.name, arguments=serialized) if successful else None)
            telemetry.consumed(resources.tool_inputs(origin.ids))
            telemetry.produced(state.results[(agent, tool_context.function_call_id)])
            if not successful:
                telemetry.failed_result()
            return result
    setattr(LoadArtifactsTool, "run_async", guarded_load)

    process = LoadArtifactsTool.process_llm_request
    @wraps(process)
    async def observed_process(self, *, tool_context, llm_request):
        if not is_session_active():
            return await process(self, tool_context=tool_context, llm_request=llm_request)
        validate_tool(self)
        validate_service(tool_context._invocation_context.artifact_service)
        from google.adk.features import FeatureName, is_feature_enabled
        if is_feature_enabled(FeatureName.DYNAMIC_INSTRUCTION_ROUTING):
            resources._fail("Artifact dynamic-instruction routing requires additional qualification")
        frame = resources._frame.get()
        if frame is None:
            resources._fail("Artifact presentation lacks an observed consumer")
        start = len(llm_request.contents)
        sdk = _sdk()
        owner = frame.resources
        selected = []
        if llm_request.contents:
            key = sdk._content_key(llm_request.contents[-1])
            matching = [record for record in owner.state.records.values() if record.key == key]
            if len(matching) == 1:
                selected = list(matching[0].ids)
        presentations: dict = {}
        token = _presentations.set(presentations)
        try:
            await process(self, tool_context=tool_context, llm_request=llm_request)
        finally:
            _presentations.reset(token)
        for index in range(start, len(llm_request.contents)):
            content = llm_request.contents[index]
            parents = list(selected)
            matched = False
            for part in content.parts or []:
                source = presentations.get(id(part))
                if source is not None and source[0] is part:
                    parents.extend(source[1])
                    matched = True
            if not matched or not selected:
                resources._fail("Artifact presentation lacks its exact load and selection provenance")
            ids = await owner.state.record(content, frame.invocation.agent.name, parents, role=sdk.Role.USER)
            resources.register_rendering(owner.state, frame.invocation.agent.name, content, ids)
    setattr(LoadArtifactsTool, "process_llm_request", observed_process)
    _installed = True


async def complete_tool(frame, result_ids):
    """Attach actual completion only to versions written by this computation."""
    owner = frame.resources
    owner.check()
    for ledger, key, version in frame.artifact_writes:
        # A later save may have replaced a catalog; never promote stale metadata.
        if ledger.versions.get(key) is not version:
            continue
        await owner.snapshot(ledger, key, version.value,
            inputs=[*version.nodes[owner.state.session], *result_ids],
            agent=frame.invocation.agent.name)
    frame.artifact_writes.clear()


def concurrent_file_service(service):
    """Only the unchanged native file backend supplies atomic version publication."""
    from . import adk_artifact_files
    return adk_artifact_files.qualified(service)


def _present(frozen, filename, ids):
    presentation = _presentations.get()
    if presentation is None:
        resources._add_reads(ids)
    elif frozen is None:
        presentation["missing"] = (filename, ids)
    else:
        missing = presentation.pop("missing", None)
        parents = [*missing[1], *ids] if missing and filename == "user:" + missing[0] else ids
        presentation[id(frozen)] = (frozen, parents)
    return frozen


async def _file_load(original, service, *, app_name, user_id, filename, session_id, version):
    from . import adk_artifact_files as files
    files.validate_name(service, app_name, user_id, session_id, filename)
    if version is not None and (type(version) is not int or version < 0):
        resources._fail("Artifact version must identify an existing nonnegative version")
    async with _operation(service, app_name, user_id, session_id, concurrent=True) as (_, owner, ledger, store):
        originals = _originals(service)
        published = await originals["list_versions"](service, app_name=app_name, user_id=user_id,
            filename=filename, session_id=session_id)
        owner.check()
        index = max(published) if version is None and published else version
        if index is None or index not in published:
            # An unpublished reservation can already have a producing snapshot;
            # an absent read must not reuse or replace that write's identity.
            key = ("artifact-missing", *_key(app_name, user_id, session_id, filename, index)[1:])
            ids = await owner.snapshot(ledger, key, {"present": False, "text": None})
            return _present(None, filename, ids)
        key = _key(app_name, user_id, session_id, filename, index)
        part = await original(service, app_name=app_name, user_id=user_id, filename=filename,
            session_id=session_id, version=index)
        owner.check()
        if part is None:
            resources._fail("A published artifact version disappeared during its read")
        frozen = _part(part)
        value = {"present": True, "text": frozen.text}
        publication = store.publications.get(key)
        if publication is not None and publication[0] == owner.state.session:
            await _verify_file_publication(service, key, store.claims[key], session_id)
        if key not in ledger.versions:
            await _adopt(owner, ledger, key, value, service, app_name, user_id, filename, session_id, index)
        ids = await owner.snapshot(ledger, key, value)
        return _present(frozen, filename, ids)


async def _file_save(original, service, *, app_name, user_id, filename, artifact, session_id, custom_metadata):
    from . import adk_artifact_files as files
    sdk = _sdk()
    async with _operation(service, app_name, user_id, session_id, concurrent=True) as (frame, owner, ledger, store):
        if frame.kind != "tool" or not sdk._current_input_ids.get():
            resources._fail("Artifact saves require an observed tool producer")
        if custom_metadata:
            resources._fail("Artifact custom metadata requires additional instrumentation")
        frozen = _part(artifact)
        reservation = files.allocate(service, app_name, user_id, session_id, filename)
        token = None
        key = _key(app_name, user_id, session_id, filename, reservation.version)
        try:
            value = {"present": True, "text": frozen.text}
            ids = await owner.snapshot(ledger, key, value,
                inputs=sdk._current_input_ids.get(), agent=frame.invocation.agent.name)
            # Keep publication ancestry distinct from eventual tool completion.
            # Publication may finish in the worker even if this task is cancelled.
            store.publications[key] = (owner.state.session, list(ids))
            metadata = {_METADATA_KEY: {"format": _METADATA_FORMAT, "nodes": list(ids),
                "origin": owner.state.snapshots[ids[0]].id,
                "entity": resources.get_current_entity() or "",
                "agent": frame.invocation.agent.name,
                "digest": hashlib.sha256(_record(key, value).encode("utf-8")).hexdigest()}}
            store.claims[key] = copy.deepcopy(metadata[_METADATA_KEY])
            token = files._reservation.set(reservation)
            saved = await original(service, app_name=app_name, user_id=user_id, filename=filename,
                artifact=frozen, session_id=session_id, custom_metadata=metadata)
            owner.check()
            if not reservation.consumed or saved != reservation.version:
                resources._fail("Artifact storage changed its reserved version")
            frame.artifact_writes.append((ledger, key, ledger.versions[key]))
            return saved
        finally:
            if token is not None:
                files._reservation.reset(token)
            reservation.finish()


async def _file_list(original, service, *, app_name, user_id, session_id):
    # Listing is still serial-only: a parallel branch may consume exact versions,
    # but cannot turn a moving inventory into a purported atomic catalog.
    frame = resources._frame.get()
    if frame is None:
        resources._fail("Artifact access lacks an observed consumer")
    frame.resources.claim(service)
    async with _operation(service, app_name, user_id, session_id, concurrent=True) as (_, owner, ledger, store):
        names = await original(service, app_name=app_name, user_id=user_id, session_id=session_id)
        owner.check()
        layout = await _layout(service, app_name, user_id, session_id)
        if sorted(names) != sorted(name for scoped in layout.values() for name in scoped):
            resources._fail("Artifact listing changed during its observation")
        if any(not versions for scoped in layout.values() for versions in scoped.values()):
            # Native reservations expose directory names before publication,
            # even when observation or storage later fails. Those names cannot
            # be treated as unrelated external inputs after losing the writer.
            resources._fail("Artifact listing contains unpublished reservations")
        for scope, scoped in layout.items():
            key = ("artifact-list", *scope)
            parents = []
            previous = ledger.versions.get(key)
            if previous is not None:
                parents.extend(previous.nodes.get(owner.state.session, []))
            observed_names = set()
            for artifact_key, (session, ids) in list(store.publications.items()):
                if (session == owner.state.session and artifact_key[1:4] == scope
                        and artifact_key[-1] in scoped.get(artifact_key[-2], ())):
                    await _verify_file_publication(service, artifact_key, store.claims[artifact_key], session_id)
                    parents.extend(ids)
                    observed_names.add(artifact_key[-2])
            external = sorted(set(scoped) - observed_names)
            if external:
                # A catalog can mix observed publications and outside inputs;
                # retain the outside part as explicitly unattributed evidence.
                external_key = ("artifact-list-external", *scope, tuple(external))
                parents.extend(await owner.snapshot(ledger, external_key, external))
            ids = await owner.snapshot(ledger, key, sorted(scoped), inputs=parents if parents else None)
            resources._add_reads(ids)
        store.layouts.update(layout)
        return list(names)


async def _verify_file_publication(service, key, claim, session_id):
    metadata = await _originals(service)["get_artifact_version"](service,
        app_name=key[1], user_id=key[2], session_id=session_id, filename=key[-2], version=key[-1])
    actual = (getattr(metadata, "custom_metadata", None) or {}).get(_METADATA_KEY)
    if actual != claim:
        resources._fail("Artifact publication no longer matches its observed producer")
