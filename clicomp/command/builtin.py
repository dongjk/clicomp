"""Built-in slash command handlers."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import datetime
from typing import Any

from clicomp import __version__
from clicomp.bus.events import OutboundMessage
from clicomp.command.router import CommandContext, CommandRouter
from clicomp.providers.base import GenerationSettings
from clicomp.utils.helpers import build_status_content


def _parse_history_ranges(spec: str) -> tuple[set[int], str | None]:
    """Parse 1-based line ranges like '1-3,7,10-12'."""
    text = (spec or "").strip()
    if not text:
        return set(), "Usage: /del 1-3,7,10-12"

    selected: set[int] = set()
    for raw_part in text.split(","):
        part = raw_part.strip()
        if not part:
            return set(), f"Invalid empty range in: {spec}"
        if "-" in part:
            left, right = part.split("-", 1)
            if not left.strip().isdigit() or not right.strip().isdigit():
                return set(), f"Invalid range: {part}"
            start = int(left)
            end = int(right)
            if start <= 0 or end <= 0:
                return set(), f"Line numbers must be >= 1: {part}"
            if start > end:
                return set(), f"Invalid descending range: {part}"
            selected.update(range(start, end + 1))
        else:
            if not part.isdigit():
                return set(), f"Invalid line number: {part}"
            line_no = int(part)
            if line_no <= 0:
                return set(), f"Line numbers must be >= 1: {part}"
            selected.add(line_no)

    return selected, None


def _history_view_indices(session) -> list[int]:
    """Map current /history lines back to indices in session.messages."""
    unconsolidated = session.messages[session.last_consolidated:]
    offset = session.last_consolidated
    sliced = unconsolidated

    for i, message in enumerate(sliced):
        if message.get("role") == "user":
            sliced = sliced[i:]
            offset += i
            break

    start = session._find_legal_start(sliced)
    if start:
        sliced = sliced[start:]
        offset += start

    return list(range(offset, offset + len(sliced)))


_HISTORY_ROLE_LABEL = {
    "user": "[U]",
    "assistant": "[A]",
    "tool": "[T]",
    "system": "[S]",
}


def _stringify_history_content(content: Any) -> str:
    """Flatten message content into one readable line."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                item_type = item.get("type")
                if item_type in {"text", "input_text", "output_text"}:
                    text = item.get("text")
                    if isinstance(text, str) and text:
                        parts.append(text)
                elif item_type == "image_url":
                    parts.append("[image]")
                else:
                    rendered = json.dumps(item, ensure_ascii=False)
                    if rendered:
                        parts.append(rendered)
            elif item is not None:
                parts.append(str(item))
        return " ".join(part for part in parts if part)
    if isinstance(content, dict):
        return json.dumps(content, ensure_ascii=False)
    return str(content)


def _history_preview(message: dict[str, Any], max_chars: int = 120) -> str:
    """Render one compact one-line preview for /history."""
    role = str(message.get("role") or "assistant")
    label = _HISTORY_ROLE_LABEL.get(role, "[?]")

    raw_content = _stringify_history_content(message.get("content"))
    char_len = len(raw_content)

    content = " ".join(raw_content.split())
    truncated = len(content) > max_chars
    if truncated:
        content = content[: max_chars - 3].rstrip() + "..."
    if not content:
        content = "(empty)"
    suffix = f" ({char_len} chars)" if truncated else ""
    return f"{label} {content}{suffix}"


def _parse_history_line_no(spec: str) -> tuple[int | None, str | None]:
    """Parse a single 1-based history line number."""
    text = (spec or "").strip()
    if not text:
        return None, "Usage: /show <history line number>"
    if not text.isdigit():
        return None, f"Invalid line number: {spec}"
    line_no = int(text)
    if line_no <= 0:
        return None, f"Line number must be >= 1: {spec}"
    return line_no, None


def _available_models(loop) -> list[str]:
    """Return configured model candidates for interactive model switching."""
    models: list[str] = []

    current = getattr(loop, "model", "")
    if current:
        models.append(current)

    provider = getattr(loop, "provider", None)
    default_model_getter = getattr(provider, "get_default_model", None)
    if callable(default_model_getter):
        default_model = default_model_getter()
        if default_model and default_model not in models:
            models.append(default_model)

    registry = getattr(provider, "spec", None)
    if registry is not None:
        for name, _ in getattr(registry, "model_overrides", ()):
            if name and name not in models:
                models.append(name)

    return models


async def cmd_stop(ctx: CommandContext) -> OutboundMessage:
    """Cancel all active tasks and subagents for the session."""
    loop = ctx.loop
    msg = ctx.msg
    tasks = loop._active_tasks.pop(msg.session_key, [])
    cancelled = sum(1 for t in tasks if not t.done() and t.cancel())
    for t in tasks:
        try:
            await t
        except (asyncio.CancelledError, Exception):
            pass
    sub_cancelled = await loop.subagents.cancel_by_session(msg.session_key)
    total = cancelled + sub_cancelled
    content = f"Stopped {total} task(s)." if total else "No active task to stop."
    return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=content)


async def cmd_restart(ctx: CommandContext) -> OutboundMessage:
    """Restart the process in-place via os.execv."""
    msg = ctx.msg

    async def _do_restart():
        await asyncio.sleep(1)
        os.execv(sys.executable, [sys.executable, "-m", "clicomp"] + sys.argv[1:])

    asyncio.create_task(_do_restart())
    return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content="Restarting...")


async def cmd_status(ctx: CommandContext) -> OutboundMessage:
    """Build an outbound status message for a session."""
    loop = ctx.loop
    session = ctx.session or loop.sessions.get_or_create(ctx.key)
    ctx_est = 0
    try:
        ctx_est, _ = loop.memory_consolidator.estimate_session_prompt_tokens(session)
    except Exception:
        pass
    if ctx_est <= 0:
        ctx_est = loop._last_usage.get("prompt_tokens", 0)
    return OutboundMessage(
        channel=ctx.msg.channel,
        chat_id=ctx.msg.chat_id,
        content=build_status_content(
            version=__version__, model=loop.model,
            start_time=loop._start_time, last_usage=loop._last_usage,
            context_window_tokens=loop.context_window_tokens,
            session_msg_count=len(session.get_history(max_messages=0)),
            context_tokens_estimate=ctx_est,
            reasoning_effort=loop.provider.generation.reasoning_effort,
        ),
        metadata={"render_as": "text"},
    )


async def cmd_new(ctx: CommandContext) -> OutboundMessage:
    """Start a fresh session."""
    loop = ctx.loop
    session = ctx.session or loop.sessions.get_or_create(ctx.key)
    snapshot = session.messages[session.last_consolidated:]
    had_history = loop.sessions.archive_and_reset(session.key)
    if snapshot:
        loop._schedule_background(loop.memory_consolidator.archive_messages(snapshot))
    return OutboundMessage(
        channel=ctx.msg.channel, chat_id=ctx.msg.chat_id,
        content="New session started. Previous session archived." if had_history else "New session started.",
    )


async def cmd_model(ctx: CommandContext) -> OutboundMessage:
    """List available models or switch to a selected model."""
    loop = ctx.loop
    msg = ctx.msg
    requested = (ctx.args or "").strip()

    if not requested:
        models = _available_models(loop)
        lines = [f"Current model: {loop.model}"]
        if models:
            lines.append("")
            lines.append("Available models:")
            lines.extend(
                f"- {name}{' (current)' if name == loop.model else ''}"
                for name in models
            )
        else:
            lines.append("")
            lines.append("No configured model list found. Use /model <name> to switch directly.")
        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content="\n".join(lines),
            metadata={"render_as": "text"},
        )

    loop.model = requested
    if hasattr(loop, "subagents") and loop.subagents:
        loop.subagents.model = requested
    if hasattr(loop, "memory_consolidator") and loop.memory_consolidator:
        loop.memory_consolidator.model = requested

    return OutboundMessage(
        channel=msg.channel,
        chat_id=msg.chat_id,
        content=f"Model switched to: {requested}",
        metadata={"render_as": "text"},
    )


async def cmd_think(ctx: CommandContext) -> OutboundMessage:
    """Show or update the provider reasoning effort."""
    loop = ctx.loop
    msg = ctx.msg
    requested = (ctx.args or "").strip().lower()
    current = loop.provider.generation.reasoning_effort
    allowed = {"none", "low", "medium", "high"}

    if not requested:
        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=(
                f"Current reasoning: {current or 'none'}\n"
                "Usage: /think <none|low|medium|high>"
            ),
            metadata={"render_as": "text"},
        )

    if requested not in allowed:
        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content="Invalid reasoning level. Use one of: none, low, medium, high.",
            metadata={"render_as": "text"},
        )

    loop.provider.generation = GenerationSettings(
        temperature=loop.provider.generation.temperature,
        max_tokens=loop.provider.generation.max_tokens,
        reasoning_effort=None if requested == "none" else requested,
    )

    return OutboundMessage(
        channel=msg.channel,
        chat_id=msg.chat_id,
        content=f"Reasoning set to: {requested}",
        metadata={"render_as": "text"},
    )


async def cmd_del(ctx: CommandContext) -> OutboundMessage:
    """Delete selected history lines from the current session."""
    loop = ctx.loop
    session = ctx.session or loop.sessions.get_or_create(ctx.key)
    selected, error = _parse_history_ranges(ctx.args)
    if error:
        return OutboundMessage(
            channel=ctx.msg.channel,
            chat_id=ctx.msg.chat_id,
            content=error,
            metadata={"render_as": "text"},
        )

    visible_indices = _history_view_indices(session)
    if not visible_indices:
        return OutboundMessage(
            channel=ctx.msg.channel,
            chat_id=ctx.msg.chat_id,
            content="No message history to delete.",
            metadata={"render_as": "text"},
        )

    max_line = len(visible_indices)
    out_of_range = sorted(n for n in selected if n > max_line)
    if out_of_range:
        preview = ", ".join(map(str, out_of_range[:10]))
        more = "..." if len(out_of_range) > 10 else ""
        return OutboundMessage(
            channel=ctx.msg.channel,
            chat_id=ctx.msg.chat_id,
            content=f"Line numbers out of range (1-{max_line}): {preview}{more}",
            metadata={"render_as": "text"},
        )

    delete_indices = {visible_indices[line_no - 1] for line_no in selected}
    if not delete_indices:
        return OutboundMessage(
            channel=ctx.msg.channel,
            chat_id=ctx.msg.chat_id,
            content="No matching history lines selected.",
            metadata={"render_as": "text"},
        )

    deleted_before_boundary = sum(1 for idx in delete_indices if idx < session.last_consolidated)
    session.messages = [msg for idx, msg in enumerate(session.messages) if idx not in delete_indices]
    if deleted_before_boundary:
        session.last_consolidated = max(0, session.last_consolidated - deleted_before_boundary)
    session.updated_at = datetime.now()
    loop.sessions.save(session)

    ctx_est = 0
    try:
        ctx_est, _ = loop.memory_consolidator.estimate_session_prompt_tokens(session)
    except Exception:
        pass

    return OutboundMessage(
        channel=ctx.msg.channel,
        chat_id=ctx.msg.chat_id,
        content=(
            f"Deleted {len(delete_indices)} history line(s). "
            f"Current visible history: {len(session.get_history(max_messages=0))} line(s). "
            f"Estimated context: {ctx_est} tokens."
        ),
        metadata={"render_as": "text"},
    )


async def cmd_history(ctx: CommandContext) -> OutboundMessage:
    """List the current session message history with numbering."""
    loop = ctx.loop
    session = ctx.session or loop.sessions.get_or_create(ctx.key)
    history = session.get_history(max_messages=0)

    if not history:
        content = "No message history yet."
    else:
        lines = [f"{idx}. {_history_preview(message)}" for idx, message in enumerate(history, start=1)]
        content = "\n".join(lines)

    return OutboundMessage(
        channel=ctx.msg.channel,
        chat_id=ctx.msg.chat_id,
        content=content,
        metadata={"render_as": "text"},
    )


async def cmd_show(ctx: CommandContext) -> OutboundMessage:
    """Show one full message from the current session history."""
    loop = ctx.loop
    session = ctx.session or loop.sessions.get_or_create(ctx.key)
    history = session.get_history(max_messages=0)
    line_no, error = _parse_history_line_no(ctx.args)
    if error:
        return OutboundMessage(
            channel=ctx.msg.channel,
            chat_id=ctx.msg.chat_id,
            content=error,
            metadata={"render_as": "text"},
        )

    if not history:
        return OutboundMessage(
            channel=ctx.msg.channel,
            chat_id=ctx.msg.chat_id,
            content="No message history yet.",
            metadata={"render_as": "text"},
        )

    assert line_no is not None
    if line_no > len(history):
        return OutboundMessage(
            channel=ctx.msg.channel,
            chat_id=ctx.msg.chat_id,
            content=f"Line number out of range (1-{len(history)}): {line_no}",
            metadata={"render_as": "text"},
        )

    message = history[line_no - 1]
    role = str(message.get("role") or "assistant")
    label = _HISTORY_ROLE_LABEL.get(role, "[?]")
    content = _stringify_history_content(message.get("content")) or "(empty)"

    return OutboundMessage(
        channel=ctx.msg.channel,
        chat_id=ctx.msg.chat_id,
        content=f"{line_no}. {label}\n\n{content}",
    )


async def cmd_help(ctx: CommandContext) -> OutboundMessage:
    """Return available slash commands."""
    lines = [
        "🐈 clicomp commands:",
        "/new — Start a new conversation",
        "/stop — Stop the current task",
        "/restart — Restart the bot",
        "/status — Show bot status",
        "/model — List models or switch model",
        "/model <name> — Switch to a model",
        "/think — Show current reasoning level",
        "/think <none|low|medium|high> — Set reasoning level",
        "/history — Show current session message history",
        "/show <n> — Show the full content of one history message",
        "/del 1-3,7,10-12 — Delete selected history lines from current session",
        "/help — Show available commands",
    ]
    return OutboundMessage(
        channel=ctx.msg.channel,
        chat_id=ctx.msg.chat_id,
        content="\n".join(lines),
        metadata={"render_as": "text"},
    )


def register_builtin_commands(router: CommandRouter) -> None:
    """Register the default set of slash commands."""
    router.priority("/stop", cmd_stop)
    router.priority("/restart", cmd_restart)
    router.priority("/status", cmd_status)
    router.exact("/new", cmd_new)
    router.exact("/status", cmd_status)
    router.exact("/help", cmd_help)
    router.exact("/history", cmd_history)
    router.prefix("/show ", cmd_show)
    router.prefix("/del ", cmd_del)
    router.exact("/model", cmd_model)
    router.prefix("/model ", cmd_model)
    router.exact("/think", cmd_think)
    router.prefix("/think ", cmd_think)
