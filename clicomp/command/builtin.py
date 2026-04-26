"""Built-in slash command handlers."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from clicomp import __version__
from clicomp.bus.events import OutboundMessage
from clicomp.command.router import CommandContext, CommandRouter
from clicomp.providers.base import GenerationSettings
from clicomp.session.manager import SessionManager
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


def _assistant_tool_call_ids(message: dict[str, Any]) -> set[str]:
    """Return tool call ids declared by an assistant message."""
    if message.get("role") != "assistant":
        return set()
    ids: set[str] = set()
    for tool_call in message.get("tool_calls") or []:
        if isinstance(tool_call, dict) and tool_call.get("id"):
            ids.add(str(tool_call["id"]))
    return ids


def _expand_delete_indices_for_tool_pairs(
    messages: list[dict[str, Any]],
    delete_indices: set[int],
) -> set[int]:
    """Expand deletions so assistant tool_calls and matching tool results stay paired.

    Tool call action lines shown in /history are derived from the assistant message
    and are not separately numbered.  If the assistant side is deleted, delete the
    corresponding tool result messages too.  Conversely, deleting a tool result must
    also delete the assistant message that declared it; otherwise providers may see
    orphaned/illegal tool-call history and /history may be forced to hide earlier
    messages at the next legal boundary.
    """
    expanded = {idx for idx in delete_indices if 0 <= idx < len(messages)}
    if not expanded:
        return expanded

    # call_id -> assistant index declaring it
    declaring_assistant: dict[str, int] = {}
    assistant_call_ids: dict[int, set[str]] = {}
    for idx, message in enumerate(messages):
        ids = _assistant_tool_call_ids(message)
        if not ids:
            continue
        assistant_call_ids[idx] = ids
        for call_id in ids:
            declaring_assistant[call_id] = idx

    call_ids_to_delete: set[str] = set()
    for idx in list(expanded):
        message = messages[idx]
        call_ids_to_delete.update(assistant_call_ids.get(idx, set()))
        if message.get("role") == "tool" and message.get("tool_call_id"):
            tool_call_id = str(message["tool_call_id"])
            assistant_idx = declaring_assistant.get(tool_call_id)
            if assistant_idx is not None:
                expanded.add(assistant_idx)
                call_ids_to_delete.update(assistant_call_ids.get(assistant_idx, {tool_call_id}))
            else:
                call_ids_to_delete.add(tool_call_id)

    if not call_ids_to_delete:
        return expanded

    for idx, message in enumerate(messages):
        if idx in assistant_call_ids and assistant_call_ids[idx] & call_ids_to_delete:
            expanded.add(idx)
        if message.get("role") == "tool" and str(message.get("tool_call_id") or "") in call_ids_to_delete:
            expanded.add(idx)

    return expanded


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


def _truncate_history_text(text: str, max_chars: int = 120) -> tuple[str, str]:
    """Normalize and truncate one-line history text; return text and suffix."""
    raw_len = len(text)
    content = " ".join(text.split())
    truncated = len(content) > max_chars
    if truncated:
        content = content[: max_chars - 3].rstrip() + "..."
    if not content:
        content = "(empty)"
    suffix = f" ({raw_len} chars)" if truncated else ""
    return content, suffix


def _history_preview(message: dict[str, Any], max_chars: int = 120) -> str:
    """Render one compact one-line preview for /history."""
    role = str(message.get("role") or "assistant")
    label = _HISTORY_ROLE_LABEL.get(role, "[?]")

    raw_content = _stringify_history_content(message.get("content"))
    content, suffix = _truncate_history_text(raw_content, max_chars=max_chars)
    return f"{label} {content}{suffix}"


def _iter_tool_call_parts(message: dict[str, Any]) -> list[dict[str, str]]:
    """Extract normalized tool-call display parts from an assistant message."""
    if message.get("role") != "assistant":
        return []

    parts: list[dict[str, str]] = []
    for tool_call in message.get("tool_calls") or []:
        if not isinstance(tool_call, dict):
            continue

        call_id = str(tool_call.get("id") or "").strip()
        function = tool_call.get("function") if isinstance(tool_call.get("function"), dict) else {}
        name = str(function.get("name") or tool_call.get("name") or "").strip()
        if not name:
            name = "(unknown tool)"

        arguments = function.get("arguments", tool_call.get("arguments", ""))
        if isinstance(arguments, str):
            args_text = arguments.strip()
        elif arguments is None:
            args_text = ""
        else:
            args_text = json.dumps(arguments, ensure_ascii=False)

        parts.append({"id": call_id, "name": name, "arguments": args_text})
    return parts


def _history_tool_call_previews(message: dict[str, Any], max_chars: int = 120) -> list[str]:
    """Render assistant tool calls as unnumbered /history action lines."""
    lines: list[str] = []
    for part in _iter_tool_call_parts(message):
        args_text = part["arguments"]
        call_text = f"{part['name']}({args_text})" if args_text else f"{part['name']}()"
        if part["id"]:
            call_text = f"{call_text} id={part['id']}"
        content, suffix = _truncate_history_text(call_text, max_chars=max_chars)
        lines.append(f"   [CALL] {content}{suffix}")
    return lines


def _show_tool_calls(message: dict[str, Any]) -> str:
    """Render full assistant tool-call details for /show."""
    blocks: list[str] = []
    for idx, part in enumerate(_iter_tool_call_parts(message), start=1):
        lines = [f"[CALL {idx}] {part['name']}"]
        if part["id"]:
            lines.append(f"id: {part['id']}")
        if part["arguments"]:
            args_text = part["arguments"]
            try:
                parsed = json.loads(args_text)
                args_text = json.dumps(parsed, ensure_ascii=False, indent=2)
            except Exception:
                pass
            lines.append("arguments:")
            lines.append(args_text)
        else:
            lines.append("arguments: {}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _parse_history_line_no(spec: str) -> tuple[int | str | None, str | None]:
    """Parse a single 1-based history line number or system id like S1."""
    text = (spec or "").strip()
    if not text:
        return None, "Usage: /show <history line number|Sx>"
    upper = text.upper()
    if upper.startswith("S") and upper[1:].isdigit():
        line_no = int(upper[1:])
        if line_no <= 0:
            return None, f"Line number must be >= 1: {spec}"
        return f"S{line_no}", None
    if not text.isdigit():
        return None, f"Invalid line number: {spec}"
    line_no = int(text)
    if line_no <= 0:
        return None, f"Line number must be >= 1: {spec}"
    return line_no, None


def _parse_repeat_args(spec: str) -> tuple[int | None, str | None, str | None]:
    """Parse /repeat arguments as <count> <message>."""
    text = (spec or "").strip()
    if not text:
        return None, None, "Usage: /repeat <count> <message>"
    parts = text.split(None, 1)
    if len(parts) != 2:
        return None, None, "Usage: /repeat <count> <message>"
    raw_count, message = parts[0], parts[1].strip()
    if not raw_count.isdigit():
        return None, None, f"Invalid repeat count: {raw_count}"
    count = int(raw_count)
    if count <= 0:
        return None, None, "Repeat count must be >= 1"
    if not message:
        return None, None, "Usage: /repeat <count> <message>"
    return count, message, None


def _read_text_if_exists(path: Path) -> str:
    """Read a UTF-8 file if it exists, else return empty text."""
    if not path.exists() or not path.is_file():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def _system_history_entries(ctx: CommandContext) -> list[dict[str, str]]:
    """Build visible system/injected context entries for /history and /show."""
    loop = ctx.loop
    workspace = loop.workspace
    builder = loop.context

    entries: list[dict[str, str]] = [
        {
            "id": "S1",
            "label": "Built-in identity prompt",
            "content": builder._get_identity(),
        }
    ]

    bootstrap_files = ["AGENTS.md", "SOUL.md", "USER.md", "TOOLS.md"]
    next_id = 2
    for filename in bootstrap_files:
        path = workspace / filename
        content = _read_text_if_exists(path)
        if content:
            entries.append({
                "id": f"S{next_id}",
                "label": filename,
                "content": content,
            })
            next_id += 1

    memory_content = _read_text_if_exists(workspace / "memory" / "MEMORY.md")
    if memory_content:
        entries.append({
            "id": f"S{next_id}",
            "label": "memory/MEMORY.md",
            "content": memory_content,
        })
        next_id += 1

    always_skills = builder.skills.get_always_skills()
    always_content = builder.skills.load_skills_for_context(always_skills) if always_skills else ""
    if always_content:
        entries.append({
            "id": f"S{next_id}",
            "label": "Always-loaded skills",
            "content": always_content,
        })
        next_id += 1

    skills_summary = builder.skills.build_skills_summary()
    if skills_summary:
        entries.append({
            "id": f"S{next_id}",
            "label": "Skills summary",
            "content": skills_summary,
        })
        next_id += 1

    entries.append({
        "id": f"S{next_id}",
        "label": "Runtime context",
        "content": builder._build_runtime_context(ctx.msg.channel, ctx.msg.chat_id, builder.timezone),
    })
    return entries


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
    base_key, _ = SessionManager.split_branch_key(ctx.key)
    active_key = loop.sessions.resolve_active_key(base_key)
    session = ctx.session if ctx.session and ctx.session.key == active_key else loop.sessions.get_or_create(active_key)
    snapshot: dict[str, int | str] | None = None
    ctx_est = 0
    try:
        snapshot = loop.memory_consolidator.estimate_effective_context_window_usage(session)
        ctx_est = int(snapshot["effective_tokens"])
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
            context_input_tokens=int(snapshot["input_tokens"]) if snapshot else None,
            context_output_reserve=int(snapshot["output_reserve"]) if snapshot else None,
            context_reasoning_reserve=int(snapshot["reasoning_reserve"]) if snapshot else None,
            context_safety_buffer=int(snapshot["safety_buffer"]) if snapshot else None,
            context_source=str(snapshot["source"]) if snapshot else None,
            session_key=base_key,
            branch=loop.sessions.branch_name_from_key(active_key),
        ),
        metadata={"render_as": "text"},
    )


async def cmd_new(ctx: CommandContext) -> OutboundMessage:
    """Start a fresh session."""
    loop = ctx.loop
    session = ctx.session or loop.sessions.get_or_create(ctx.key)
    snapshot = session.messages[session.last_consolidated:]
    had_history = loop.sessions.archive_and_reset(session.key)
    # Reset provider-side conversation chaining so the next turn is rebuilt
    # strictly from local session history.
    session = loop.sessions.get_or_create(ctx.key)
    session.metadata.pop("azure_previous_response_id", None)
    loop.sessions.save(session)
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

    requested_delete_indices = {visible_indices[line_no - 1] for line_no in selected}
    delete_indices = _expand_delete_indices_for_tool_pairs(session.messages, requested_delete_indices)
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
    # Deleting local history changes the intended prompt prefix, so clear any
    # provider-side chained response id to avoid stale remote context.
    session.metadata.pop("azure_previous_response_id", None)
    session.updated_at = datetime.now()
    loop.sessions.save(session)

    ctx_est = 0
    try:
        ctx_est, _ = loop.memory_consolidator.estimate_session_prompt_tokens(session)
    except Exception:
        pass

    extra_deleted = len(delete_indices) - len(requested_delete_indices)
    paired_note = f" ({extra_deleted} paired tool line(s) included)." if extra_deleted else "."
    return OutboundMessage(
        channel=ctx.msg.channel,
        chat_id=ctx.msg.chat_id,
        content=(
            f"Deleted {len(delete_indices)} history line(s){paired_note} "
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
    system_entries = _system_history_entries(ctx)

    lines: list[str] = []
    if system_entries:
        for entry in system_entries:
            preview = _history_preview({"role": "system", "content": entry["content"]})
            lines.append(f"{entry['id']}. {preview} — {entry['label']}")

    if history:
        if lines:
            lines.append("")
        for idx, message in enumerate(history, start=1):
            lines.append(f"{idx}. {_history_preview(message)}")
            lines.extend(_history_tool_call_previews(message))

    content = "\n".join(lines) if lines else "No message history yet."

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
    if isinstance(line_no, str) and line_no.upper().startswith("S"):
        system_entries = _system_history_entries(ctx)
        target = next((entry for entry in system_entries if entry["id"] == line_no.upper()), None)
        if target is None:
            return OutboundMessage(
                channel=ctx.msg.channel,
                chat_id=ctx.msg.chat_id,
                content=f"System line not found: {line_no}",
                metadata={"render_as": "text"},
            )
        content = target["content"] or "(empty)"
        return OutboundMessage(
            channel=ctx.msg.channel,
            chat_id=ctx.msg.chat_id,
            content=f"{target['id']}. [S] {target['label']}\n\n{content}",
            metadata={"render_as": "text"},
        )

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
    tool_calls = _show_tool_calls(message)
    if tool_calls:
        content = f"{content}\n\nTool calls:\n\n{tool_calls}"

    return OutboundMessage(
        channel=ctx.msg.channel,
        chat_id=ctx.msg.chat_id,
        content=f"{line_no}. {label}\n\n{content}",
    )


async def cmd_branch(ctx: CommandContext) -> OutboundMessage:
    """List branches for the current session or switch/create one."""
    loop = ctx.loop
    base_key, _ = SessionManager.split_branch_key(ctx.key)
    requested = (ctx.args or "").strip()

    if not requested:
        current = loop.sessions.get_current_branch(base_key)
        branches = loop.sessions.list_branches(base_key)
        lines = [f"Session: {base_key}", f"Current branch: {current}", "", "Branches:"]
        for item in branches:
            marker = " (current)" if item.get("current") else ""
            lines.append(f"- {item['branch']}{marker}")
        return OutboundMessage(
            channel=ctx.msg.channel,
            chat_id=ctx.msg.chat_id,
            content="\n".join(lines),
            metadata={"render_as": "text"},
        )

    session, created = loop.sessions.switch_branch(base_key, requested)
    branch = loop.sessions.branch_name_from_key(session.key)
    verb = "Created and switched to" if created else "Switched to"
    return OutboundMessage(
        channel=ctx.msg.channel,
        chat_id=ctx.msg.chat_id,
        content=f"{verb} branch: {branch}",
        metadata={"render_as": "text"},
    )


async def cmd_repeat(ctx: CommandContext) -> OutboundMessage:
    """Arm CLI auto-repeat for the next N turns."""
    count, message, error = _parse_repeat_args(ctx.args)
    if error:
        return OutboundMessage(
            channel=ctx.msg.channel,
            chat_id=ctx.msg.chat_id,
            content=error,
            metadata={"render_as": "text"},
        )

    assert count is not None and message is not None
    return OutboundMessage(
        channel=ctx.msg.channel,
        chat_id=ctx.msg.chat_id,
        content=f"Auto-repeat armed: {count} × {message}",
        metadata={
            "render_as": "text",
            "_repeat": {"remaining": count, "message": message},
        },
    )


async def cmd_tools(ctx: CommandContext) -> OutboundMessage:
    """List currently registered model tools."""
    loop = ctx.loop
    tool_defs = loop.tools.get_definitions() if getattr(loop, "tools", None) else []

    if not tool_defs:
        return OutboundMessage(
            channel=ctx.msg.channel,
            chat_id=ctx.msg.chat_id,
            content="No tools are currently registered.",
            metadata={"render_as": "text"},
        )

    lines = [f"Registered tools: {len(tool_defs)}", ""]
    for idx, tool_def in enumerate(tool_defs, start=1):
        if tool_def.get("type") == "function":
            fn = tool_def.get("function", {})
            name = str(fn.get("name") or f"tool_{idx}")
            desc = " ".join(str(fn.get("description") or "").split()) or "(no description)"
            props = ((fn.get("parameters") or {}).get("properties") or {})
            required = (fn.get("parameters") or {}).get("required") or []
            arg_names = ", ".join(props.keys()) if props else "(no args)"
            req_text = ", ".join(required) if required else "(none)"
            lines.append(f"- {name}")
            lines.append(f"  desc: {desc}")
            lines.append(f"  args: {arg_names}")
            lines.append(f"  required: {req_text}")
        else:
            lines.append(f"- {tool_def.get('type', 'unknown')}")
        if idx < len(tool_defs):
            lines.append("")

    return OutboundMessage(
        channel=ctx.msg.channel,
        chat_id=ctx.msg.chat_id,
        content="\n".join(lines),
        metadata={"render_as": "text"},
    )


async def cmd_help(ctx: CommandContext) -> OutboundMessage:
    """Return available slash commands."""
    lines = [
        "🐈 clicomp commands:",
        "/new — Start a new conversation",
        "/stop — Stop the current task",
        "/restart — Restart the bot",
        "/status — Show bot status, current session, and branch",
        "/model — List models or switch model",
        "/model <name> — Switch to a model",
        "/think — Show current reasoning level",
        "/think <none|low|medium|high> — Set reasoning level",
        "/history — Show current session message history",
        "/show <n|Sx> — Show the full content of one history or injected-system message",
        "/del 1-3,7,10-12 — Delete selected history lines from current session",
        "/branch — List branches for the current session",
        "/branch <name> — Create/switch to a branch; /branch main returns to main",
        "/repeat <n> <message> — CLI: auto-send <message> after each turn, n times",
        "/tools — List currently registered model tools",
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
    router.exact("/tools", cmd_tools)
    router.prefix("/repeat ", cmd_repeat)
    router.exact("/history", cmd_history)
    router.exact("/branch", cmd_branch)
    router.prefix("/branch ", cmd_branch)
    router.prefix("/show ", cmd_show)
    router.prefix("/del ", cmd_del)
    router.exact("/model", cmd_model)
    router.prefix("/model ", cmd_model)
    router.exact("/think", cmd_think)
    router.prefix("/think ", cmd_think)
