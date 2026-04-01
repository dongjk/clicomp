"""Built-in slash command handlers."""

from __future__ import annotations

import asyncio
import os
import sys

from clicomp import __version__
from clicomp.bus.events import OutboundMessage
from clicomp.command.router import CommandContext, CommandRouter
from clicomp.providers.base import GenerationSettings
from clicomp.utils.helpers import build_status_content


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
    session.clear()
    loop.sessions.save(session)
    loop.sessions.invalidate(session.key)
    if snapshot:
        loop._schedule_background(loop.memory_consolidator.archive_messages(snapshot))
    return OutboundMessage(
        channel=ctx.msg.channel, chat_id=ctx.msg.chat_id,
        content="New session started.",
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
    router.exact("/model", cmd_model)
    router.prefix("/model ", cmd_model)
    router.exact("/think", cmd_think)
    router.prefix("/think ", cmd_think)
