"""Import-triggered installation must precede framework references escaping."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def run(source):
    environment = {**os.environ, 'PYTHONPATH': str(ROOT / 'sdk/python')}
    result = subprocess.run([sys.executable, '-c', source], env=environment,
                            cwd=ROOT, text=True, capture_output=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr


def test_auto_registration_does_not_import_unused_frameworks():
    run('''
import sys
import sasy
roots = ('google.adk', 'langchain', 'langgraph', 'langroid')
assert not any(root in sys.modules for root in roots)
sasy.instrument()
sasy.instrument()
assert not any(root in sys.modules for root in roots)
assert 'sasy.instrumentation.adk' not in sys.modules
assert 'sasy.instrumentation.langchain' not in sys.modules
assert set(sasy.instrumentation._deferred) == {'adk', 'langchain', 'langroid'}
''')


@pytest.mark.parametrize('framework', ['adk', 'langchain', 'langroid'])
def test_framework_import_installs_only_its_adapter(framework):
    target = {'adk': 'google.adk', 'langchain': 'langchain', 'langroid': 'langroid'}[framework]
    run(f'''
import importlib, sys
import sasy
sasy.instrument()
importlib.import_module({target!r})
assert {framework!r} not in sasy.instrumentation._deferred
if {framework!r} == 'langchain':
    from langchain.agents import create_agent
    import sasy.instrumentation.langchain as adapter
    assert create_agent is adapter._langchain_create_agent
    assert adapter._installed
elif {framework!r} == 'adk':
    import sasy.instrumentation.adk as adapter
    assert adapter._installed
else:
    import sasy.instrumentation.langroid as adapter
    assert adapter.is_instrumented()
for other, root in [('adk', 'google.adk'), ('langchain', 'langchain'), ('langroid', 'langroid')]:
    if other != {framework!r}:
        assert root not in sys.modules, root
''')


@pytest.mark.parametrize('entry', [
    'from langgraph.prebuilt import ToolNode',
    'from langgraph.graph import StateGraph',
    'from langgraph.graph.state import StateGraph',
    'from langgraph.pregel import Pregel',
])
def test_langgraph_namespace_does_not_bypass_adapter_installation(entry):
    run(f'''
import sasy
sasy.instrument()
{entry}
from langgraph.prebuilt import ToolNode
import sasy.instrumentation.langchain as adapter
assert adapter._installed
assert hasattr(ToolNode._run_one, '__wrapped__')
assert 'langchain' not in sasy.instrumentation._deferred
''')


def test_framework_required_flag_is_eager():
    run('''
import sys, sasy
sasy.instrument(langchain=True, adk=False, langroid=False)
assert 'langchain' in sys.modules
import sasy.instrumentation.langchain as adapter
assert adapter._installed
''')


def test_false_cancels_pending_installation_and_auto_can_enable_it_again():
    run('''
import sys, sasy
sasy.instrument()
sasy.instrument(adk=False, langchain=False, langroid=False)
import langchain
assert 'sasy.instrumentation.langchain' not in sys.modules
sasy.instrument(adk=False, langchain=None, langroid=False)
from langchain.agents import create_agent
import sasy.instrumentation.langchain as adapter
assert create_agent is adapter._langchain_create_agent
''')


def test_worker_framework_imports_require_eager_startup_installation():
    run('''
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
import importlib, sasy
sasy.instrument()
barrier = Barrier(2)
def load(name):
    barrier.wait()
    try:
        module = importlib.import_module(name)
    except RuntimeError as error:
        assert 'instrumentation startup thread' in str(error), str(error)
        return False
    import sasy.instrumentation.langchain as adapter
    assert adapter._installed
    from langchain.agents import create_agent
    assert create_agent is adapter._langchain_create_agent
    return True
with ThreadPoolExecutor(max_workers=2) as pool:
    results = list(pool.map(load, ['langchain', 'langgraph.prebuilt']))
assert results == [False, False]
''')


@pytest.mark.parametrize('problem', ['version', 'import'])
def test_deferred_unsupported_adapter_warns_on_first_use(problem):
    run(f'''
import warnings, sasy
from sasy.instrumentation import SasyInstrumentationWarning

def loader():
    if {problem!r} == 'import':
        raise ImportError('synthetic missing adapter dependency')
    def check():
        raise RuntimeError('synthetic unsupported framework version')
    def install():
        raise AssertionError('Unsupported adapter must not be installed')
    return check, install
sasy.instrumentation._langchain_adapter = loader
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always', SasyInstrumentationWarning)
    sasy.instrument(adk=False, langroid=False)
    from langchain.agents import create_agent
    assert not caught
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage
    model = FakeMessagesListChatModel(responses=[AIMessage(content='synthetic')])
    create_agent(model, tools=[])
    create_agent(model, tools=[])
assert len(caught) == 1
assert isinstance(caught[0].message, SasyInstrumentationWarning)
assert 'will NOT be checked' in str(caught[0].message)
assert caught[0].filename == '<string>'
''')


def test_eager_upgrade_cancels_pending_callback_before_loading_adapter():
    run('''
import warnings, sasy
from sasy.instrumentation import SasyInstrumentationWarning
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always', SasyInstrumentationWarning)
    sasy.instrument(adk=False, langroid=False)
    sasy.instrument(adk=False, langroid=False, langchain=True)
    from langchain.agents import create_agent
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage
    model = FakeMessagesListChatModel(responses=[AIMessage(content='synthetic')])
    create_agent(model, tools=[])
    assert not [w for w in caught if isinstance(w.message, SasyInstrumentationWarning)]
import sasy.instrumentation.langchain as adapter
assert adapter._installed
assert 'langchain' not in sasy.instrumentation._deferred
''')


@pytest.mark.parametrize('framework', ['adk', 'langchain'])
def test_direct_adapter_import_completes_pending_installation(framework):
    run(f'''
import importlib, warnings, sasy
from sasy.instrumentation import SasyInstrumentationWarning
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always', SasyInstrumentationWarning)
    sasy.instrument()
    adapter = importlib.import_module('sasy.instrumentation.' + {framework!r})
assert adapter._installed
assert {framework!r} not in sasy.instrumentation._deferred
assert not [w for w in caught if isinstance(w.message, SasyInstrumentationWarning)]
''')


def test_eager_startup_allows_framework_imports_in_workers():
    run('''
from concurrent.futures import ThreadPoolExecutor
import sasy
sasy.instrument(langchain=True, adk=False, langroid=False)
def use():
    from langchain.agents import create_agent
    import sasy.instrumentation.langchain as adapter
    assert adapter._installed
    assert create_agent is adapter._langchain_create_agent
with ThreadPoolExecutor(max_workers=2) as pool:
    list(pool.map(lambda _: use(), range(2)))
''')
