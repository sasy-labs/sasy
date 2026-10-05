"""Script provider responses to test the public message-flow demo without a provider call."""

import builtins
import copy
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from openai.types.chat import ChatCompletion

ROOT = next(parent for parent in Path(__file__).resolve().parents
            if (parent / "examples/message-flow/demo.py").is_file())
sys.path.insert(0, str(ROOT / "examples/message-flow"))
SPEC = importlib.util.spec_from_file_location("sasy_message_flow_demo", ROOT / "examples/message-flow/demo.py")
assert SPEC and SPEC.loader
demo = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(demo)


def completion(name=None, arguments=None, *, content=None):
    message = {"role": "assistant", "content": content}
    if name:
        message["tool_calls"] = [
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }
        ]
    return ChatCompletion.model_validate(
        {
            "id": "scripted",
            "created": 0,
            "model": "scripted",
            "object": "chat.completion",
            "choices": [{"index": 0, "finish_reason": "tool_calls" if name else "stop", "message": message}],
        }
    )


class ScriptedClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **request):
        self.requests.append(copy.deepcopy(request))
        return next(self.responses)


def scenario(untrusted):
    responses = []
    if untrusted:
        responses.append(completion("read_document", {"name": "external-note"}))
    responses.extend(
        [
            completion("read_document", {"name": "sensitive-report"}),
            completion("send_summary", {"to": "reviewer@example.net", "body": "The budget is 42."}),
            completion(content="Finished."),
        ]
    )
    return ScriptedClient(responses)


class LiveLoopTests(unittest.TestCase):
    def setUp(self):
        self.events = {}
        self.parents = {}
        self.requests = []
        self.deny_send = False

        def capture(events, edges):
            self.events.update((event.id, event) for event in events)
            for edge in edges:
                self.parents.setdefault(edge.destination, set()).add(edge.source)

        def check(name, arguments, input_node_ids):
            self.requests.append((name, json.loads(arguments), input_node_ids))
            allowed = not (self.deny_send and name == "send_summary")
            return SimpleNamespace(authorized=allowed, denial_reasons=[] if allowed else ["test denial"])

        for mocker in [
            patch.object(demo, "record_events_with_dependencies", side_effect=capture),
            patch.object(demo.sasy, "check_tool_call", side_effect=check),
            patch.object(demo.sasy, "session", return_value=nullcontext()),
        ]:
            mocker.start()
            self.addCleanup(mocker.stop)

    def test_denial_reaches_model_without_delivery_and_preserves_all_input_edges(self):
        self.deny_send = True
        client = scenario(True)
        sent, decisions = demo.run_live(client, "scripted", untrusted=True)
        self.assertEqual(sent, [])
        self.assertEqual(decisions[-1], ("send_summary", False))
        self.assertEqual(client.requests[-1]["messages"][-1]["content"], "[BLOCKED] test denial")
        name, arguments, request_ids = self.requests[-1]
        self.assertEqual(
            (name, arguments), ("send_summary", {"to": "reviewer@example.net", "body": "The budget is 42."})
        )
        (llm_id,) = self.parents[request_ids[0]]
        actual_inputs = client.requests[-2]["messages"]
        self.assertEqual(len(self.parents[llm_id]), len(actual_inputs))
        document_results = [event.id for event in self.events.values() if event.derived_from.name == "read_document"]
        self.assertTrue(set(document_results).issubset(self.parents[llm_id]))

    def test_allowed_send_runs_shared_simulated_handler(self):
        sent, decisions = demo.run_live(scenario(False), "scripted")
        self.assertEqual(sent, [{"to": "reviewer@example.net", "body": "The budget is 42."}])
        self.assertEqual(decisions, [("read_document", True), ("send_summary", True)])

    def test_invalid_model_arguments_never_dispatch(self):
        for name, args in [("shell", {"command": "anything"}), ("send_summary", {"to": []}), ("read_document", [])]:
            with self.subTest(name=name, args=args):
                client = ScriptedClient([completion(name, args)])
                with self.assertRaises(ValueError):
                    demo.run_live(client, "scripted")
        self.assertEqual(self.requests, [])

    def test_turn_bound(self):
        client = ScriptedClient([completion("read_document", {"name": "sensitive-report"})])
        with self.assertRaisesRegex(RuntimeError, "Stopped after 1"):
            demo.run_live(client, "scripted", max_turns=1)
        self.assertEqual(len(client.requests), 1)

    def test_reference_monitor_error_stops_before_side_effect(self):
        with patch.object(demo.sasy, "check_tool_call", side_effect=RuntimeError("engine unavailable")):
            client = ScriptedClient([completion("send_summary", {"to": "reviewer@example.net", "body": "synthetic"})])
            with self.assertRaisesRegex(RuntimeError, "engine unavailable"):
                demo.run_live(client, "scripted")
        self.assertEqual(len(client.requests), 1)


class CliTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="sasy-demo-cli-")
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        (self.root / ".env").write_text(
            "OPENAI_API_KEY=dotenv-test-key\nSASY_LIVE_MODEL=dotenv-model\nSASY_URL=localhost:12345\n"
        )
        for mocker in [
            patch.dict(os.environ, {}, clear=True),
            patch.object(demo, "__file__", str(self.root / "examples/message-flow/demo.py")),
        ]:
            mocker.start()
            self.addCleanup(mocker.stop)

    def test_scripted_cli_loads_dotenv_without_importing_openai(self):
        original_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name == "openai" or name.startswith("openai."):
                raise AssertionError("Scripted mode must not import OpenAI")
            return original_import(name, *args, **kwargs)

        configured = []
        with (
            patch("sys.argv", ["demo.py", "--quiet"]),
            patch("builtins.__import__", side_effect=guarded_import),
            patch.object(demo.sasy, "configure", side_effect=lambda **kwargs: configured.append(os.environ["SASY_URL"])),
            patch.object(demo, "run_scenario") as run,
        ):
            demo.main()
        self.assertEqual(configured, ["localhost:12345"])
        self.assertEqual([call.kwargs for call in run.call_args_list], [
            {"untrusted": False, "trace": False, "quiet": True},
            {"untrusted": True, "trace": False, "quiet": True},
        ])

    def test_live_cli_preserves_environment_and_selects_one_scenario(self):
        os.environ["OPENAI_API_KEY"] = "exported-test-key"
        for flags, expected_model, untrusted in [
            ([], "dotenv-model", True),
            (["--model", "explicit-model", "--sensitive-only"], "explicit-model", False),
            (["--untrusted"], "dotenv-model", True),
        ]:
            with (
                self.subTest(flags=flags),
                patch("sys.argv", ["demo.py", "--live", "--quiet", *flags]),
                patch.object(demo.sasy, "configure") as configure,
                patch("openai.OpenAI") as provider,
                patch.object(demo, "run_live") as run,
            ):
                demo.main()
            configure.assert_called_once()
            self.assertEqual(provider.call_args.kwargs["api_key"], "exported-test-key")
            run.assert_called_once_with(provider.return_value.__enter__.return_value, expected_model,
                                        untrusted=untrusted, max_turns=8, trace=False, quiet=True)

    def test_live_cli_uses_default_model_without_override(self):
        (self.root / ".env").write_text("OPENAI_API_KEY=dotenv-test-key\n")
        with (
            patch("sys.argv", ["demo.py", "--live"]),
            patch.object(demo.sasy, "configure"),
            patch("openai.OpenAI") as provider,
            patch.object(demo, "run_live") as run,
        ):
            demo.main()
        run.assert_called_once_with(provider.return_value.__enter__.return_value, "gpt-4.1-mini",
                                    untrusted=True, max_turns=8, trace=True, quiet=False)


if __name__ == "__main__":
    unittest.main()
