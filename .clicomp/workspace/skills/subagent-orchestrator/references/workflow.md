# Subagent Orchestration Workflow

## Goal

Use a persistent subagent session as a steerable worker. Inspect each reply and decide the next message intentionally.

## Required Working Directory

Run the underlying agent command from `/home/admin01/clicomp/`.

That repository directory has the configured model setup expected by this workflow. If you launch the command elsewhere, model selection/configuration may differ from what you expect.

## Decision Loop

### 1. Open or resume a worker

Choose a task name that matches the workstream.

- Reuse name = continue prior conversation
- New name = fresh worker

### 2. Configure the worker if needed

Choose between two control styles:

### Per-run overrides

Use these when you want to affect only the next invocation:

- `--model gpt-5.4`
- `--model gpt-5.4-mini`
- `--think none`
- `--think low`
- `--think medium`
- `--think high`

### In-session control turns

Use these when you want to change the session's ongoing setting:

- `/model gpt-5.4`
- `/model gpt-5.4-mini`
- `/think low`
- `/think medium`
- `/think high`

## Recommended Prompt Template

For the first working message, include:

1. role
2. goal
3. constraints
4. expected output format
5. immediate next action

Template:

```text
You are helping with <domain/task>.
Goal: <desired outcome>.
Constraints: <important limits>.
Output format: <format>.
Do this now: <specific next action>.
```

## Follow-up Prompt Patterns

### Continue execution

```text
Good. Continue with the next step. Do not repeat prior context unless something changed.
```

### Force execution instead of planning

```text
Do not expand the plan further. Execute step 1 now and report concrete findings only.
```

### Recover from drift

```text
You are drifting from scope. Ignore optional ideas. Focus only on <scope>. Output exactly: <format>.
```

### Provide missing context

```text
Here is the missing context:
- <fact 1>
- <fact 2>
Now continue from your last good state.
```

### Ask for better structure

```text
Rewrite your current answer as:
1. facts
2. decisions
3. open questions
4. recommended next action
```

### Ask for self-check

```text
Before continuing, list the top 2 assumptions you are making and the risk if either is wrong.
```

## Model / Effort Heuristics

### Prefer `gpt-5.4`

Use the stronger model when:

- task framing is ambiguous,
- quality matters more than speed,
- debugging or synthesis is non-trivial,
- or the worker has been making shallow mistakes.

### Prefer `gpt-5.4-mini`

Use the smaller model when:

- the task is repetitive,
- the structure is already clear,
- you mainly need formatting or straightforward follow-through,
- or you are doing cheap intermediate iterations.

### Prefer `none`

Use for:

- pure formatting,
- minimal-cost routing,
- or extremely obvious follow-ups.

### Prefer `low`

Use for:

- simple reformatting,
- short summaries,
- obvious next actions,
- lightweight status updates.

### Prefer `medium`

Use as the normal setting for routine multi-step work.

### Prefer `high`

Use for:

- ambiguous planning,
- tricky debugging,
- synthesis across multiple constraints,
- or when the worker's answer seems too shallow.

## Triage After Every Reply

### Reply is good and actionable

Send a narrow next step.

Example:

```text
Proceed with the fix. Stop after the first changed file and summarize what changed.
```

### Reply is too vague

Ask for evidence, not confidence.

Example:

```text
Be specific. What exact evidence supports that conclusion? Quote the relevant lines or observations.
```

### Reply is too long

Compress and reformat.

Example:

```text
Condense this to 5 bullets: findings, blocker, recommendation, risk, next step.
```

### Reply reveals confusion

Re-anchor the worker.

Example:

```text
Reset to this narrower goal: <goal>. Ignore any earlier side quests.
```

### Reply shows repeated weak reasoning

Increase effort or model quality.

Example per-run override:

```text
--think high
```

Or as an in-session control turn:

```text
/think high
```

then:

```text
Re-evaluate from scratch. State your assumptions explicitly before giving the answer.
```

## When to Stop

Stop when one of these is true:

- the worker delivered the requested artifact,
- the remaining work is not worth another turn,
- or the worker is blocked on information you do not have.

When stopping, it is often useful to request one final compact handoff:

```text
Give me the final result as:
- outcome
- evidence
- remaining risks
- recommended next action
```
