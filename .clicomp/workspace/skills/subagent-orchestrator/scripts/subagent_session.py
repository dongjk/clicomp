#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

VALID_THINK_LEVELS = ("none", "low", "medium", "high")
DEFAULT_CWD = Path("/home/admin01/clicomp")
BASE_COMMAND = [
    "uv",
    "run",
    "python",
    "-m",
    "clicomp.cli.commands",
    "agent",
    "--no-stream",
]


def find_project_root(start: Path) -> Path:
    if DEFAULT_CWD.exists():
        return DEFAULT_CWD
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    return Path.cwd()


def read_message(message: str | None, message_file: str | None) -> str:
    if message is not None:
        return message
    if message_file is None:
        raise ValueError("either --message or --message-file is required")
    if message_file == "-":
        return sys.stdin.read()
    return Path(message_file).read_text(encoding="utf-8")


def build_message(args: argparse.Namespace) -> str:
    if args.command == "send":
        return read_message(args.message, args.message_file)
    if args.command == "model":
        return f"/model {args.model}"
    if args.command == "think":
        return f"/think {args.effort}"
    raise ValueError(f"unsupported command: {args.command}")


def build_agent_command(
    task: str,
    message: str,
    *,
    model: str | None = None,
    think: str | None = None,
) -> list[str]:
    command = [*BASE_COMMAND, "-s", task, "-m", message]
    if model:
        command.extend(["--model", model])
    if think:
        command.extend(["--think", think])
    return command


def run_command(command: list[str], cwd: Path, timeout: int | None) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
        return {
            "ok": completed.returncode == 0,
            "exit_code": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    except FileNotFoundError as exc:
        return {
            "ok": False,
            "exit_code": 127,
            "stdout": "",
            "stderr": str(exc),
        }
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        return {
            "ok": False,
            "exit_code": 124,
            "stdout": stdout,
            "stderr": stderr + ("\n" if stderr else "") + f"Timed out after {timeout} seconds",
        }


def format_result(result: dict[str, Any], as_json: bool) -> str:
    if as_json:
        return json.dumps(result, ensure_ascii=False, indent=2)

    lines = [
        f"COMMAND: {result['command_display']}",
        f"WORKDIR: {result['cwd']}",
        f"TASK: {result['task']}",
        "MESSAGE:",
        result["message"],
        "--- STDOUT ---",
        result["stdout"] if result["stdout"] else "",
        "--- STDERR ---",
        result["stderr"] if result["stderr"] else "",
        f"EXIT_CODE: {result['exit_code']}",
        f"OK: {str(result['ok']).lower()}",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Send a message or control command to a persistent clicomp agent session "
            "and capture stdout/stderr for orchestration."
        )
    )
    parser.add_argument(
        "--cwd",
        help="Working directory for `uv run ...`. Defaults to /home/admin01/clicomp because that directory has the configured model environment.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Optional timeout in seconds.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit structured JSON instead of a human-readable report.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the derived command and payload without invoking the agent.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    send_parser = subparsers.add_parser("send", help="Send a normal message to the session.")
    send_parser.add_argument("--task", required=True, help="Session/task name.")
    send_group = send_parser.add_mutually_exclusive_group(required=True)
    send_group.add_argument("--message", help="Message text to send.")
    send_group.add_argument(
        "--message-file",
        help="Read message text from a UTF-8 file. Use '-' to read from stdin.",
    )
    send_parser.add_argument(
        "--model",
        help="Optional per-run model override for this invocation only.",
    )
    send_parser.add_argument(
        "--think",
        choices=VALID_THINK_LEVELS,
        help="Optional per-run reasoning effort override for this invocation only.",
    )

    model_parser = subparsers.add_parser("model", help="Send a /model control message.")
    model_parser.add_argument("--task", required=True, help="Session/task name.")
    model_parser.add_argument("--model", required=True, help="Model name, e.g. gpt-5.4-mini.")

    think_parser = subparsers.add_parser("think", help="Send a /think control message.")
    think_parser.add_argument("--task", required=True, help="Session/task name.")
    think_parser.add_argument(
        "--effort",
        required=True,
        choices=VALID_THINK_LEVELS,
        help="Reasoning effort: low, medium, or high.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    script_path = Path(__file__).resolve()
    detected_root = find_project_root(script_path.parent)
    cwd = Path(args.cwd).resolve() if args.cwd else detected_root

    message = build_message(args)
    command = build_agent_command(
        args.task,
        message,
        model=getattr(args, "model", None),
        think=getattr(args, "think", None),
    )

    result: dict[str, Any] = {
        "task": args.task,
        "message": message,
        "command": command,
        "command_display": subprocess.list2cmdline(command),
        "cwd": str(cwd),
    }

    if args.dry_run:
        result.update(
            {
                "ok": True,
                "exit_code": 0,
                "stdout": "",
                "stderr": "",
                "dry_run": True,
            }
        )
        print(format_result(result, args.json))
        return 0

    exec_result = run_command(command, cwd=cwd, timeout=args.timeout)
    result.update(exec_result)
    result["dry_run"] = False
    print(format_result(result, args.json))
    return int(result["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
