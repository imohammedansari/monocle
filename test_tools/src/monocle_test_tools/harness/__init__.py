"""Scenario test harness: drive a target agent with an LLM-backed test agent.

Everything here is optional: install with ``pip install "monocle_test_tools[test_agent]"``
to pull in the LangChain dependencies the test agent needs. Imports of LangChain are
deliberately lazy inside each module so importing ``monocle_test_tools`` stays cheap and
dependency-free.
"""
from monocle_test_tools.harness.scenario import ScenarioParam, ScenarioTestCase

__all__ = ["ScenarioParam", "ScenarioTestCase"]
