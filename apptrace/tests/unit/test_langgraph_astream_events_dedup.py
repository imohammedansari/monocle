"""Tests for the LangGraph "one agentic span per graph run" dedup and the
verify_tool_names_in_spans IndexError guard.

Background
----------
LangChain's ``Runnable.astream_events(version="v2")`` always drives the public
``astream`` on the SAME CompiledStateGraph instance. Because both methods are
instrumented, a single graph run used to emit a redundant *nested pair* of
``agentic.invocation`` spans (astream_events -> AGENT, inner astream -> AGENT_STREAM).

The fix makes ``astream_events`` skip its own span but stash the chosen
(turn-vs-invocation) processor keyed by ``id(instance)``; the inner
``astream``/``stream`` adopts it and becomes the single, correctly-typed agentic
span. These tests drive that path end-to-end (real wrapper machinery + real
AGENT / AGENT_STREAM processors) and assert exactly ONE agentic span per graph
run, with the inference child parented to it.
"""

import types
import unittest

from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from monocle_apptrace.instrumentation.common.constants import (
    ANY_AGENT,
    LAST_INFERENCE,
    SPAN_TYPES,
)
from monocle_apptrace.instrumentation.common.instrumentor import (
    get_monocle_instrumentor,
    setup_monocle_telemetry,
)
from monocle_apptrace.instrumentation.common.scope_wrapper import (
    start_scope,
    stop_scope,
)
from monocle_apptrace.instrumentation.common.utils import verify_tool_names_in_spans
from monocle_apptrace.instrumentation.common.wrapper import (
    atask_iter_wrapper,
    atask_wrapper,
)
from monocle_apptrace.instrumentation.common.wrapper_method import WrapperMethod
from monocle_apptrace.instrumentation.metamodel.langgraph import _helper
from monocle_apptrace.instrumentation.metamodel.langgraph.entities.inference import (
    AGENT,
    AGENT_STREAM,
)


class _FakeMessage:
    """Minimal LangChain-message stand-in for the stream processor."""

    def __init__(self, content):
        self.content = content
        self.response_metadata = {}


# A minimal inference-typed output processor so the child span shows as an
# inference span under the surviving agentic span.
_INFERENCE_PROC = {
    "type": SPAN_TYPES.INFERENCE_FRAMEWORK,
    "attributes": [],
    "events": [],
}


class FakeCompiledGraph:
    """A CompiledStateGraph-shaped stand-in.

    ``astream_events`` drives the public ``astream`` on the same instance, exactly
    like LangChain's Runnable.astream_events(v2). ``astream`` runs one inference
    (a child span) then yields a values-mode chunk.
    """

    def __init__(self, name="LangGraph", single_agent=True):
        self.name = name
        node = "agent" if single_agent else "supervisor"
        self.builder = types.SimpleNamespace(nodes={node: None})

    async def _infer(self, *args, **kwargs):
        return {"messages": [_FakeMessage("inference-done")]}

    async def astream(self, input=None, config=None, **kwargs):
        # One inference under this graph run.
        await self._infer()
        yield {"messages": [_FakeMessage("final answer")]}

    async def astream_events(self, input=None, config=None, version=None, **kwargs):
        # LangChain drives the wrapped public astream internally on the same instance.
        async for chunk in self.astream(input=input, config=config):
            yield {"event": "on_chain_stream", "data": {"chunk": chunk}}


def _langgraph_wrapper_methods():
    # NOTE: do NOT pass scope_name here -- WrapperMethod replaces wrapper_method with
    # scope_wrapper whenever scope_name is set without scope_values, which would bypass
    # the iter wrapper entirely. The agentic turn/invocation scope is derived from the
    # output_processor type (get_builtin_scope_names), not from scope_name.
    pkg = FakeCompiledGraph.__module__
    return [
        WrapperMethod(
            package=pkg,
            object_name="FakeCompiledGraph",
            method="astream_events",
            wrapper_method=atask_iter_wrapper,
            span_handler="langgraph_agent_handler",
            output_processor=AGENT,
        ),
        WrapperMethod(
            package=pkg,
            object_name="FakeCompiledGraph",
            method="astream",
            wrapper_method=atask_iter_wrapper,
            span_handler="langgraph_agent_handler",
            output_processor=AGENT_STREAM,
        ),
        WrapperMethod(
            package=pkg,
            object_name="FakeCompiledGraph",
            method="_infer",
            wrapper_method=atask_wrapper,
            span_handler="default",
            output_processor=_INFERENCE_PROC,
        ),
    ]


class TestAstreamEventsNoDoubling(unittest.IsolatedAsyncioTestCase):
    """A graph run driven via astream_events must emit ONE agentic span, not two."""

    def setUp(self):
        existing = get_monocle_instrumentor()
        if existing is not None:
            try:
                existing.uninstrument()
            except Exception:
                pass
        # Ensure no stale dedup markers leak across tests.
        _helper._PENDING_STREAM_PROCESSOR.clear()

        self.exporter = InMemorySpanExporter()
        self.instrumentor = setup_monocle_telemetry(
            workflow_name="astream_events_dedup_test",
            span_processors=[SimpleSpanProcessor(self.exporter)],
            wrapper_methods=_langgraph_wrapper_methods(),
            union_with_default_methods=False,
        )

    def tearDown(self):
        try:
            if self.instrumentor is not None:
                self.instrumentor.uninstrument()
        except Exception:
            pass
        _helper._PENDING_STREAM_PROCESSOR.clear()
        return super().tearDown()

    def _spans_by_type(self):
        by_type = {}
        for span in self.exporter.get_finished_spans():
            stype = span.attributes.get("span.type")
            by_type.setdefault(stype, []).append(span)
        return by_type

    async def _drain(self, graph):
        async for _ in graph.astream_events(input={"messages": [_FakeMessage("hi")]}):
            pass

    async def test_top_level_turn_emits_single_turn_and_no_stray_invocation(self):
        # Orchestrator (multi-agent) top-level graph run, no turn scope yet.
        graph = FakeCompiledGraph(name="orchestrator", single_agent=False)
        await self._drain(graph)

        by_type = self._spans_by_type()
        turns = by_type.get(SPAN_TYPES.AGENTIC_REQUEST, [])
        invocations = by_type.get(SPAN_TYPES.AGENTIC_INVOCATION, [])
        inferences = by_type.get(SPAN_TYPES.INFERENCE_FRAMEWORK, [])

        # Exactly one agentic.turn; the inner astream became it, astream_events emitted nothing.
        self.assertEqual(len(turns), 1, f"expected 1 turn, got {len(turns)}")
        # The redundant inner-astream invocation must be gone (this is the doubling bug).
        self.assertEqual(len(invocations), 0, f"expected 0 stray invocations, got {len(invocations)}")
        # The single inference is parented to the surviving turn span.
        self.assertEqual(len(inferences), 1)
        self.assertEqual(inferences[0].parent.span_id, turns[0].context.span_id)
        # No marker leaked.
        self.assertEqual(_helper._PENDING_STREAM_PROCESSOR, {})

    async def test_nested_invocation_emits_single_invocation(self):
        # Sub-agent graph run dispatched inside an already-established turn scope.
        turn_token = start_scope(SPAN_TYPES.AGENTIC_REQUEST, scope_value="orchestrator-turn")
        try:
            graph = FakeCompiledGraph(name="LangGraph", single_agent=True)
            await self._drain(graph)
        finally:
            stop_scope(turn_token)

        by_type = self._spans_by_type()
        invocations = by_type.get(SPAN_TYPES.AGENTIC_INVOCATION, [])
        turns = by_type.get(SPAN_TYPES.AGENTIC_REQUEST, [])
        inferences = by_type.get(SPAN_TYPES.INFERENCE_FRAMEWORK, [])

        # ONE agentic.invocation, not the old nested pair (astream_events + inner astream).
        self.assertEqual(len(invocations), 1, f"expected 1 invocation, got {len(invocations)}")
        self.assertEqual(len(turns), 0, "a nested sub-agent must not open a new turn")
        # A single inference span parented to that one invocation.
        self.assertEqual(len(inferences), 1)
        self.assertEqual(inferences[0].parent.span_id, invocations[0].context.span_id)
        self.assertEqual(_helper._PENDING_STREAM_PROCESSOR, {})


class _FakeSpan:
    """Span stand-in exposing only the .attributes dict verify_tool_names_in_spans reads."""

    def __init__(self, attributes):
        self.attributes = attributes


class TestVerifyToolNamesIndexGuard(unittest.TestCase):
    """verify_tool_names_in_spans must not raise IndexError on empty/colon-less LAST_INFERENCE."""

    def _child(self, entity_name, span_type=SPAN_TYPES.AGENTIC_TOOL_INVOCATION):
        return _FakeSpan({"entity.1.name": entity_name, "span.type": span_type})

    def _parent(self, last_inference_value):
        return _FakeSpan({LAST_INFERENCE: last_inference_value})

    def test_empty_last_inference_returns_false_without_raising(self):
        # LAST_INFERENCE is reset to "" after being consumed; the key still exists.
        result = verify_tool_names_in_spans(
            self._child("knowledge_base_search"), self._parent("")
        )
        self.assertFalse(result)

    def test_colonless_last_inference_returns_false_without_raising(self):
        result = verify_tool_names_in_spans(
            self._child("knowledge_base_search"), self._parent("no-colon-here")
        )
        self.assertFalse(result)

    def test_valid_last_inference_still_links_tool(self):
        result = verify_tool_names_in_spans(
            self._child("knowledge_base_search"),
            self._parent("0x00abc:knowledge_base_search"),
        )
        self.assertTrue(result)

    def test_valid_last_inference_non_matching_tool_returns_false(self):
        result = verify_tool_names_in_spans(
            self._child("some_other_tool"),
            self._parent("0x00abc:knowledge_base_search"),
        )
        self.assertFalse(result)

    def test_any_agent_wildcard_matches_agentic_invocation(self):
        result = verify_tool_names_in_spans(
            self._child("whatever", span_type=SPAN_TYPES.AGENTIC_INVOCATION),
            self._parent(f"0x00abc:{ANY_AGENT}"),
        )
        self.assertTrue(result)


if __name__ == "__main__":
    unittest.main()
