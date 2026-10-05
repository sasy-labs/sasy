"""Synthetic credential corpus and capture boundary regressions; no engine needed."""
import asyncio
import io
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from sasy.capture import MAX_CAPTURE_DEPTH, capture_logger, capture_text
from sasy.instrumentation.session import session
from sasy.observability import api
from sasy.proto.observability_pb2 import Computation, Edge, Event, Tool

FIXTURES = json.loads((Path(__file__).parents[1] / 'sdk/typescript/test/fixtures/capture-credentials.json').read_text())
RAW = 'https://api.test/?key=fixture-secret&n=1'
CLEAN = 'https://api.test/?key=[redacted]&n=1'


@pytest.mark.parametrize('fixture', FIXTURES, ids=lambda f: f['name'])
def test_shared_corpus(fixture):
    assert capture_text(fixture['input']) == fixture['expected']
    assert capture_text(fixture['expected']) == fixture['expected']


@pytest.mark.parametrize('text', [
    'An ordinary message: no credential syntax.',
    '{"question":"summarize","scope":"internal"}',
    r'{"\u0061uthorization":"fixture-secret"}',
    r'{"url":"https:\/\/example.test\/?key=fixture-secret"}',
    'Authorızatıon: Bearer fixture-secret',
    'AUTHORIZATION: Bearer fixture-secret',
    'https://example.test/?token=fixture-secret',
])
def test_capture_fast_path_matches_full_scanner(text):
    from sasy.capture import _capture_text
    assert capture_text(text) == _capture_text(text, 0, [max(65536, len(text) * 4)])


@pytest.mark.parametrize('method', ['record_events', 'record_events_async', 'record_events_with_dependencies', 'record_events_with_dependencies_async', 'register_computations', 'register_computations_async'])
def test_wire_copies_all_capture_paths(monkeypatch, method):
    calls = []
    def collect(request, metadata):
        calls.append((request, metadata))
        return SimpleNamespace(ids=['fixture-id'])
    async def async_collect(request, metadata):
        return collect(request, metadata)
    stub = SimpleNamespace(**{name: async_collect if method.endswith('_async') else collect for name in ['RegisterEvents', 'RegisterEventsWithDependencies', 'RegisterComputations']})
    monkeypatch.setattr(api, 'get_stub', lambda _: stub)
    monkeypatch.setattr(api, 'get_async_stub', lambda _: stub)
    monkeypatch.setattr(api, '_metadata', lambda: [('x-api-key', 'fixture-transport-secret')])
    event = Event(id='fixture-id', text=RAW, agent=RAW, principal=RAW, tools=[Tool(name=RAW, arguments=json.dumps({'Authorization':'fixture-secret'}))], derived_from=Tool(name='tool', arguments=RAW))
    edge = Edge(source='fixture-input', destination='fixture-id', principal=RAW)
    span = Computation(trace_id='fixture-trace', span_id='fixture-span', name=RAW, status_message=RAW, attributes_json=json.dumps({'http.request.header.authorization':['fixture-secret']}), events_json=json.dumps([{'exception.message': RAW}]), linked_span_ids=['fixture-link'])
    before = [x.SerializeToString() for x in (event, edge, span)]
    args = ([span],) if method.startswith('register_computations') else ([event], [edge]) if 'with_dependencies' in method else ([event],)
    with session('fixture-session', entity='fixture-actor', end_on_exit=False):
        result = getattr(api, method)(*args)
        if method.endswith('_async'):
            result = asyncio.run(result)
    assert result == ['fixture-id']
    assert [x.SerializeToString() for x in (event, edge, span)] == before
    request, metadata = calls[0]
    assert metadata == [('x-api-key', 'fixture-transport-secret')]
    assert request.session_id == 'fixture-session'
    if method.startswith('register_computations'):
        captured = request.computations[0]
        assert captured.status_message == CLEAN
        assert 'fixture-secret' not in captured.attributes_json + captured.events_json
        assert captured.name == RAW and captured.linked_span_ids == ['fixture-link']
    else:
        # A recorded message goes to the graph as it is. The reference monitor
        # is given the same bytes when it decides, so a rule that reads an
        # ancestor's contents reads what the action actually carried.
        captured = request.events[0]
        assert captured.text == RAW and captured.agent == RAW and captured.principal == RAW
        assert captured.entity == 'fixture-actor'
        assert captured.tools[0].name == RAW
        assert captured.tools[0].arguments == json.dumps({'Authorization': 'fixture-secret'})
        assert captured.derived_from.arguments == RAW
        if 'with_dependencies' in method:
            assert request.edges[0].source == edge.source and request.edges[0].principal == RAW
    if method.startswith('register_computations'):
        # Transport metadata has no policy value and is still scrubbed.
        assert captured.status_message != RAW


def test_real_request_and_direct_authorization_unchanged(monkeypatch):
    from sasy.proto.reference_monitor_pb2 import ToolCallResponse
    from sasy.reference_monitor import api as rm
    request = httpx.Request('GET', RAW, headers={'Authorization':'Bearer fixture-secret'})
    before = (str(request.url), list(request.headers.raw))
    args = json.dumps({'url':str(request.url), 'headers':dict(request.headers)})
    assert 'fixture-secret' not in capture_text(args)
    calls = []
    def check(req, metadata):
        calls.append(req)
        return ToolCallResponse(authorized=True)
    monkeypatch.setattr(rm, 'get_stub', lambda _: SimpleNamespace(CheckToolCall=check))
    monkeypatch.setattr(rm, '_metadata', lambda: [])
    with session('fixture-session', end_on_exit=False):
        rm.check_tool_call('http', args, ['fixture-input'])
    assert calls[0].args == args
    assert calls[0].input_node_ids == ['fixture-input']
    assert (str(request.url), list(request.headers.raw)) == before


def test_sdk_logger_sanitizes_exception_without_mutating_exception():
    stream = io.StringIO()
    logger = capture_logger('sasy.capture.synthetic-test')
    handler = logging.StreamHandler(stream)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    error = ValueError(RAW)
    try:
        try:
            raise error
        except ValueError:
            logger.exception('failed %s', RAW)
        assert 'fixture-secret' not in stream.getvalue()
        assert CLEAN in stream.getvalue() and str(error) == RAW
    finally:
        logger.removeHandler(handler)


def test_depth_limit_refuses_capture_and_omits_diagnostic(monkeypatch):
    # Lowering the same configured internal limit avoids exponential test data.
    import sasy.capture as capture
    monkeypatch.setattr(capture, 'MAX_CAPTURE_DEPTH', 2)
    nested = RAW
    for _ in range(4):
        nested = json.dumps(nested)
    with pytest.raises(ValueError, match='limit'):
        capture_text(nested)
    record = logging.LogRecord('sasy', logging.ERROR, '', 1, nested, (), None)
    assert capture.CaptureLogFilter().filter(record)
    assert record.getMessage() == '[telemetry diagnostic omitted: capture limit exceeded]'
    assert MAX_CAPTURE_DEPTH == 16


def test_mapping_computation_constructor_compatibility():
    from sasy.capture import capture_computations
    mapping = {'span_id':'fixture-span', 'status_message':RAW}
    assert capture_computations([mapping])[0].status_message == CLEAN
    assert mapping['status_message'] == RAW


@pytest.mark.parametrize('depth,padding', [(10000, 0), (7, 20000)])
def test_nested_header_arrays_reject_before_repeated_parsing(monkeypatch, depth, padding):
    import sasy.capture as capture
    payload = '{"Authorization":[' * depth + json.dumps('fixture-secret' + 'x' * padding) + ']}' * depth
    original_loads = json.loads
    parse_calls = []
    def count_loads(value):
        parse_calls.append(len(value))
        return original_loads(value)
    monkeypatch.setattr(capture.json, 'loads', count_loads)
    with pytest.raises(ValueError, match='nesting/work limit') as error:
        capture_text(payload)
    assert 'fixture-secret' not in str(error.value)
    # Deep input is rejected before parsing an array. Shallow repeated suffix
    # parsing spends its budget after at most one array, independent of payload.
    assert len(parse_calls) <= 3
    assert sum(length > 20 for length in parse_calls) <= 1


def test_a_recorded_message_is_what_the_monitor_was_given():
    """The graph and the decision must be about the same text.

    The reference monitor is handed a tool call's arguments raw. If the record
    were scrubbed, a rule reading an ancestor's contents would decide on text
    that neither the model nor the monitor ever saw.
    """
    from sasy.capture import capture_events

    # Prose that the transport-credential scrubber matches: it is an ordinary
    # sentence, and every word of it can matter to a policy.
    approval = 'The board gave authorization: proceed with the 1,000,000 EUR transfer to ACME'
    arguments = json.dumps({'note': 'authorization: approved by Jane', 'amount': 1000000})
    event = Event(text=approval, metadata=json.dumps({'s:content': approval}),
                  tools=[Tool(name='wire', arguments=arguments)],
                  derived_from=Tool(name='approve', arguments=arguments))

    recorded = capture_events([event])[0]
    assert recorded.text == approval
    assert recorded.metadata == json.dumps({'s:content': approval})
    assert recorded.tools[0].arguments == arguments
    assert recorded.derived_from.arguments == arguments

    # The same text as diagnostics is still scrubbed, and that is the
    # difference: a span attribute or a log line carries transport metadata,
    # which no policy reads.
    assert capture_text(approval) != approval


def test_an_oversized_message_is_still_refused():
    from sasy.capture import MAX_CAPTURE_LENGTH, capture_events

    with pytest.raises(ValueError, match='exceeds'):
        capture_events([Event(text='x' * (MAX_CAPTURE_LENGTH + 1))])
    with pytest.raises(ValueError, match='exceeds'):
        capture_events([Event(tools=[Tool(name='t', arguments='x' * (MAX_CAPTURE_LENGTH + 1))])])
    # An adapter's metadata record of the message is bounded the same way, and
    # like the text it is recorded verbatim rather than scrubbed.
    with pytest.raises(ValueError, match='exceeds'):
        capture_events([Event(metadata='x' * (MAX_CAPTURE_LENGTH + 1))])
