---
name: subagent-orchestrator
description: Coordinate persistent subagent sessions through `uv run python -m clicomp.cli.commands agent --no-stream -s SESSION -m MESSAGE`. Use when the agent should delegate work to a reusable helper session, inspect each reply, and keep steering the same worker across multiple turns instead of fire-and-forget delegation. Supports per-run overrides such as `--model ...` and `--think none|low|medium|high`, plus in-session control turns like `/model ...` and `/think ...`.
---

# Subagent Orchestrator

Use this skill to direct a persistent subagent session, read its latest output, and decide the next `-m` message based on that output.

A session name is the task identity. Reusing the same task name continues the existing conversation. Choose a new task name when you want a fresh worker with no prior context.

## Required Working Directory

The agent command must be executed from `/home/admin01/clicomp/`. That directory has the configured model environment for this workflow.

Treat this as a hard requirement, not a convenience.

- Preferred working directory: `/home/admin01/clicomp`
- If using the bundled script, let it default to `/home/admin01/clicomp`
- If running the raw command manually, `cd /home/admin01/clicomp` first

If the command is run from a different directory, model configuration may be wrong or incomplete.

## Core Rule

Do not treat the subagent as one-shot unless the task truly is one-shot.

Preferred loop:

1. Choose or reuse a task/session name.
2. Optionally set the worker model.
3. Optionally set reasoning effort.
4. Send a concrete work instruction.
5. Read the command output carefully.
6. Decide the next turn:
   - continue execution,
   - tighten constraints,
   - provide missing context,
   - ask for a different output format,
   - switch model,
   - switch think level,
   - or stop because the task is done.

## Use the Bundled Script

Prefer the wrapper script over hand-writing shell quoting:

```bash
python skills/subagent-orchestrator/scripts/subagent_session.py send --task <task-name> --message "<message>"
python skills/subagent-orchestrator/scripts/subagent_session.py send --task <task-name> --message "<message>" --model gpt-5.4-mini --think low
python skills/subagent-orchestrator/scripts/subagent_session.py model --task <task-name> --model gpt-5.4-mini
python skills/subagent-orchestrator/scripts/subagent_session.py think --task <task-name> --effort high
```

The script runs the canonical command from `/home/admin01/clicomp` by default:

```bash
cd /home/admin01/clicomp
uv run python -m clicomp.cli.commands agent --no-stream -s <task-name> -m <message>
uv run python -m clicomp.cli.commands agent --no-stream -s <task-name> -m <message> --model gpt-5.4-mini --think low
```

and prints the exact command, working directory, task, message, stdout, stderr, and exit code. Use that output to plan the next turn.

## Recommended Workflow

### 1. Pick a stable task name

Use short, specific names such as:

- `repo-audit`
- `pricing-research`
- `fix-auth-tests`
- `draft-release-notes`

Reuse the same name to continue the same worker. Fork into a new task name if the current session becomes muddled or if you need a separate workstream.

### 2. Start with an explicit assignment

First message should usually include:

- the role or responsibility,
- the concrete goal,
- the expected output,
- constraints,
- and whether the worker should execute immediately or first propose a plan.

Example:

```bash
python skills/subagent-orchestrator/scripts/subagent_session.py send \
  --task fix-auth-tests \
  --message "You are helping with auth test failures. First inspect the failure pattern, identify the smallest likely root cause, then propose a 3-step fix plan. Keep it concise."
```

### 3. Adjust model when useful

Default model is `gpt-5.4`.

Use a per-run override when you want to change model only for the next invocation:

```bash
python skills/subagent-orchestrator/scripts/subagent_session.py send \
  --task fix-auth-tests \
  --message "Continue from the last result and rewrite the summary as bullets." \
  --model gpt-5.4-mini
```

Switch model by sending a control message into the same session when you want the session itself to move to a different default:

```bash
python skills/subagent-orchestrator/scripts/subagent_session.py model --task fix-auth-tests --model gpt-5.4-mini
```

Use a smaller model for lighter, repetitive, or formatting-heavy follow-ups. Move back to the stronger model when quality or reasoning depth matters.

### 4. Adjust thinking effort when useful

Supported effort levels:

- `none`
- `low`
- `medium`
- `high`

Use a per-run override when you want to change reasoning effort only for the next invocation:

```bash
python skills/subagent-orchestrator/scripts/subagent_session.py send \
  --task fix-auth-tests \
  --message "Re-check the prior conclusion and list the two riskiest assumptions." \
  --think high
```

Change effort by sending a control message into the same session when you want to update the session setting:

```bash
python skills/subagent-orchestrator/scripts/subagent_session.py think --task fix-auth-tests --effort high
```

General heuristic:

- `low`: simple follow-ups, formatting, summarizing, obvious next steps
- `medium`: normal default for multi-step work
- `high`: ambiguous problems, tricky planning, synthesis, or debugging

### 5. Read output and choose the next turn deliberately

After every turn, inspect the returned stdout. Then decide what the next `-m` should do.

Common cases:

- **Worker gave only a plan but has not executed.**
  Reply with: execute step 1 now, report findings, and stop at the first blocker.

- **Worker asked for missing context.**
  Reply with only the needed facts plus a restated goal.

- **Worker drifted from scope.**
  Reply with a correction, the exact scope, and the required format.

- **Worker is doing fine but cost/speed matters more.**
  Use a per-run override like `--model gpt-5.4-mini --think low`, or send `/model gpt-5.4-mini` / `/think low` if you want the session itself to change.

- **Worker output is shallow or uncertain.**
  Use `--think high` for the next turn, or send `/think high` and ask for a revised answer; switch back to `gpt-5.4` if model quality is the real issue.

- **Worker is done.**
  Ask for a final concise deliverable if needed, then stop.

## Output-Shaping Patterns

Useful follow-up messages:

- "Execute the first step now and report only concrete findings."
- "Keep going; do not re-explain the plan unless it changed."
- "Restate the current blocker in one sentence, then propose the best next action."
- "Rewrite the result as a checklist with owner / action / risk."
- "Summarize only decisions and open questions."
- "You are drifting. Ignore previous optional ideas and focus only on <scope>."

## When to Change Session Names

Start a new session instead of continuing the old one when:

- the worker has accumulated too much irrelevant context,
- the task changed substantially,
- you need a clean comparison path,
- or you want two parallel worker threads.

## Raw Command Fallback

If you do not use the wrapper script, use the raw command directly from `/home/admin01/clicomp`:

```bash
cd /home/admin01/clicomp
uv run python -m clicomp.cli.commands agent --no-stream -s <task-name> -m "<message>"
uv run python -m clicomp.cli.commands agent --no-stream -s <task-name> -m "<message>" --model gpt-5.4-mini --think low
```

Use `--model` / `--think` for per-run overrides. Use control messages as ordinary `-m` values when you want to change the session itself:

```bash
uv run python -m clicomp.cli.commands agent --no-stream -s <task-name> -m "/model gpt-5.4-mini"
uv run python -m clicomp.cli.commands agent --no-stream -s <task-name> -m "/think low"
```

## Reference

For prompt patterns and turn-by-turn steering heuristics, read:

- `references/workflow.md`
