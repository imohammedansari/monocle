"""Scenario params as zero-arg LangChain tools.

One tool per param. The test agent calls a tool when the target agent asks for that
detail; the value of a withheld param never appears in the prompt, so the tool call is
the only way the test agent can learn it.
"""
from typing import Any

from monocle_test_tools.harness.scenario import ScenarioParam


def build_param_tools(params: list[ScenarioParam]) -> tuple[list[Any], list[str]]:
    """Build one zero-arg tool per param.

    Returns ``(tools, calls)``. ``calls`` is a live list that each tool appends its own
    name to when invoked, so the driver can record which details the test agent had to
    ask for on each turn. The driver clears it before a turn and snapshots it after.
    """
    calls: list[str] = []
    return [_build_param_tool(param, calls) for param in params], calls


def _build_param_tool(param: ScenarioParam, calls: list[str]) -> Any:
    """Build the tool for one param.

    Separate function rather than an inline closure in the loop: binding ``param`` as
    an argument is what keeps each tool returning its own value instead of every tool
    closing over the last loop variable.
    """
    from langchain_core.tools import StructuredTool

    def fetch_param() -> str:
        calls.append(param.name)
        return str(param.value)

    return StructuredTool.from_function(
        func=fetch_param,
        name=param.name,
        description=(
            f"Returns the {param.description}. Call this only when the target agent "
            f"asks you for the {param.description}."
        ),
    )
