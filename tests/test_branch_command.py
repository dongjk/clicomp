from __future__ import annotations

from pathlib import Path

import pytest

from clicomp.agent.loop import AgentLoop
from clicomp.bus.queue import MessageBus
from clicomp.providers.base import GenerationSettings, LLMProvider, LLMResponse


class DummyProvider(LLMProvider):
    def __init__(self):
        super().__init__()
        self.generation = GenerationSettings(max_tokens=128)

    async def chat(
        self,
        messages,
        tools=None,
        model=None,
        max_tokens=4096,
        temperature=0.7,
        reasoning_effort=None,
        tool_choice=None,
    ):
        user_messages = [m for m in messages if m.get("role") == "user"]
        content = user_messages[-1].get("content") if user_messages else ""
        if isinstance(content, list):
            content = " ".join(str(x) for x in content)
        return LLMResponse(content=f"echo:{content}")

    def get_default_model(self) -> str:
        return "dummy-model"


@pytest.fixture
def loop(tmp_path: Path) -> AgentLoop:
    bus = MessageBus()
    return AgentLoop(
        bus=bus,
        provider=DummyProvider(),
        workspace=tmp_path,
        context_window_tokens=100000,
    )


@pytest.mark.asyncio
async def test_branch_create_switch_and_isolate_history(loop: AgentLoop):
    base_key = "cli:direct"

    first = await loop.process_direct("hello", session_key=base_key, channel="cli", chat_id="direct")
    assert first is not None

    switched = await loop.process_direct("/branch abc", session_key=base_key, channel="cli", chat_id="direct")
    assert switched is not None
    assert "abc" in switched.content
    assert loop.sessions.get_current_branch(base_key) == "abc"

    branched = await loop.process_direct("on abc", session_key=base_key, channel="cli", chat_id="direct")
    assert branched is not None

    main_session = loop.sessions.get_or_create(base_key)
    abc_session = loop.sessions.get_or_create(loop.sessions.make_branch_key(base_key, "abc"))
    assert len(abc_session.messages) > len(main_session.messages)

    deleted = await loop.process_direct("/del 3-4", session_key=base_key, channel="cli", chat_id="direct")
    assert deleted is not None
    assert "Deleted 2 history line(s)." in deleted.content
    assert len(loop.sessions.get_or_create(loop.sessions.make_branch_key(base_key, "abc")).messages) == len(abc_session.messages) - 2
    assert len(loop.sessions.get_or_create(base_key).messages) == len(main_session.messages)

    back = await loop.process_direct("/branch main", session_key=base_key, channel="cli", chat_id="direct")
    assert back is not None
    assert loop.sessions.get_current_branch(base_key) == "main"

    status = await loop.process_direct("/status", session_key=base_key, channel="cli", chat_id="direct")
    assert status is not None
    assert "Session: cli:direct" in status.content
    assert "Branch: main" in status.content


@pytest.mark.asyncio
async def test_branch_list(loop: AgentLoop):
    base_key = "cli:direct"

    await loop.process_direct("seed", session_key=base_key, channel="cli", chat_id="direct")
    await loop.process_direct("/branch abc", session_key=base_key, channel="cli", chat_id="direct")
    await loop.process_direct("/branch", session_key=base_key, channel="cli", chat_id="direct")
    listed = await loop.process_direct("/branch", session_key=base_key, channel="cli", chat_id="direct")

    assert listed is not None
    assert "Branches:" in listed.content
    assert "- main" in listed.content
    assert "- abc (current)" in listed.content


@pytest.mark.asyncio
async def test_history_includes_system_entries_and_show_sx(loop: AgentLoop):
    base_key = "cli:direct"

    await loop.process_direct("hello", session_key=base_key, channel="cli", chat_id="direct")
    history = await loop.process_direct("/history", session_key=base_key, channel="cli", chat_id="direct")
    assert history is not None
    assert "S1. [S]" in history.content
    assert "Runtime context" in history.content

    shown = await loop.process_direct("/show S1", session_key=base_key, channel="cli", chat_id="direct")
    assert shown is not None
    assert "Built-in identity prompt" in shown.content

    runtime_id = next(
        line.split(".", 1)[0]
        for line in history.content.splitlines()
        if "Runtime context" in line and line.startswith("S")
    )
    runtime = await loop.process_direct(f"/show {runtime_id}", session_key=base_key, channel="cli", chat_id="direct")
    assert runtime is not None
    assert "Runtime context" in runtime.content or "Current Time:" in runtime.content


@pytest.mark.asyncio
async def test_tools_command_lists_registered_tools(loop: AgentLoop):
    listed = await loop.process_direct("/tools", session_key="cli:direct", channel="cli", chat_id="direct")
    assert listed is not None
    assert "Registered tools:" in listed.content
    assert "- read_file" in listed.content
