"""Session management for conversation history."""

import json
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from loguru import logger

from clicomp.utils.helpers import ensure_dir, safe_filename


@dataclass
class Session:
    """
    A conversation session.

    Stores messages in JSONL format for easy reading and persistence.

    Important: Messages are append-only for LLM cache efficiency.
    The consolidation process writes summaries to MEMORY.md/HISTORY.md
    but does NOT modify the messages list or get_history() output.
    """

    key: str  # channel:chat_id or channel:chat_id#branch=<name>
    messages: list[dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)
    last_consolidated: int = 0  # Number of messages already consolidated to files

    def add_message(self, role: str, content: str, **kwargs: Any) -> None:
        """Add a message to the session."""
        msg = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            **kwargs
        }
        self.messages.append(msg)
        self.updated_at = datetime.now()

    @staticmethod
    def _find_legal_start(messages: list[dict[str, Any]]) -> int:
        """Find first index where every tool result has a matching assistant tool_call."""
        declared: set[str] = set()
        start = 0
        for i, msg in enumerate(messages):
            role = msg.get("role")
            if role == "assistant":
                for tc in msg.get("tool_calls") or []:
                    if isinstance(tc, dict) and tc.get("id"):
                        declared.add(str(tc["id"]))
            elif role == "tool":
                tid = msg.get("tool_call_id")
                if tid and str(tid) not in declared:
                    start = i + 1
                    declared.clear()
                    for prev in messages[start:i + 1]:
                        if prev.get("role") == "assistant":
                            for tc in prev.get("tool_calls") or []:
                                if isinstance(tc, dict) and tc.get("id"):
                                    declared.add(str(tc["id"]))
        return start

    def get_history(self, max_messages: int = 500) -> list[dict[str, Any]]:
        """Return unconsolidated messages for LLM input, aligned to a legal tool-call boundary."""
        unconsolidated = self.messages[self.last_consolidated:]
        sliced = unconsolidated[-max_messages:]

        # Drop leading non-user messages to avoid starting mid-turn when possible.
        for i, message in enumerate(sliced):
            if message.get("role") == "user":
                sliced = sliced[i:]
                break

        # Some providers reject orphan tool results if the matching assistant
        # tool_calls message fell outside the fixed-size history window.
        start = self._find_legal_start(sliced)
        if start:
            sliced = sliced[start:]

        out: list[dict[str, Any]] = []
        for message in sliced:
            entry: dict[str, Any] = {"role": message["role"], "content": message.get("content", "")}
            for key in ("tool_calls", "tool_call_id", "name"):
                if key in message:
                    entry[key] = message[key]
            out.append(entry)
        return out

    def clear(self) -> None:
        """Clear all messages and reset session to initial state."""
        self.messages = []
        self.last_consolidated = 0
        self.metadata.pop("azure_previous_response_id", None)
        self.updated_at = datetime.now()

    def retain_recent_legal_suffix(self, max_messages: int) -> None:
        """Keep a legal recent suffix, mirroring get_history boundary rules."""
        if max_messages <= 0:
            self.clear()
            return
        if len(self.messages) <= max_messages:
            return

        start_idx = max(0, len(self.messages) - max_messages)

        # If the cutoff lands mid-turn, extend backward to the nearest user turn.
        while start_idx > 0 and self.messages[start_idx].get("role") != "user":
            start_idx -= 1

        retained = self.messages[start_idx:]

        # Mirror get_history(): avoid persisting orphan tool results at the front.
        start = self._find_legal_start(retained)
        if start:
            retained = retained[start:]

        dropped = len(self.messages) - len(retained)
        self.messages = retained
        self.last_consolidated = max(0, self.last_consolidated - dropped)
        self.updated_at = datetime.now()


class SessionManager:
    """
    Manages conversation sessions.

    Sessions are stored as JSONL files in the sessions directory.
    """

    BRANCH_MAIN = "main"

    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.sessions_dir = ensure_dir(self.workspace / "sessions")
        self.archive_dir = ensure_dir(self.sessions_dir / "archive")
        self._cache: dict[str, Session] = {}

    @staticmethod
    def split_branch_key(key: str) -> tuple[str, str]:
        """Split a persisted key into base key and branch name."""
        marker = "#branch="
        if marker in key:
            base, branch = key.split(marker, 1)
            branch = branch.strip() or SessionManager.BRANCH_MAIN
            return base, branch
        return key, SessionManager.BRANCH_MAIN

    @classmethod
    def make_branch_key(cls, base_key: str, branch: str | None) -> str:
        """Build the persisted session key for a branch."""
        branch_name = (branch or cls.BRANCH_MAIN).strip() or cls.BRANCH_MAIN
        if branch_name == cls.BRANCH_MAIN:
            return base_key
        return f"{base_key}#branch={branch_name}"

    @classmethod
    def branch_name_from_key(cls, key: str) -> str:
        """Return the branch portion from a session key."""
        return cls.split_branch_key(key)[1]

    def _get_session_path(self, key: str) -> Path:
        """Get the file path for a session."""
        safe_key = safe_filename(key.replace(":", "_"))
        return self.sessions_dir / f"{safe_key}.jsonl"

    def _get_branch_meta_path(self, base_key: str) -> Path:
        """Get the file path for current branch metadata."""
        safe_key = safe_filename(base_key.replace(":", "_"))
        return self.sessions_dir / f"{safe_key}.branches.json"

    def get_current_branch(self, base_key: str) -> str:
        """Return the currently selected branch for a base session."""
        path = self._get_branch_meta_path(base_key)
        if not path.exists():
            return self.BRANCH_MAIN
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            branch = str(data.get("current") or self.BRANCH_MAIN).strip() or self.BRANCH_MAIN
            return branch
        except Exception as e:
            logger.warning("Failed to load branch metadata for {}: {}", base_key, e)
            return self.BRANCH_MAIN

    def set_current_branch(self, base_key: str, branch: str) -> None:
        """Persist the currently selected branch for a base session."""
        branch_name = (branch or self.BRANCH_MAIN).strip() or self.BRANCH_MAIN
        path = self._get_branch_meta_path(base_key)
        payload = {
            "base_key": base_key,
            "current": branch_name,
            "updated_at": datetime.now().isoformat(),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def resolve_active_key(self, key: str) -> str:
        """Resolve a user-facing base session key to its active branch session key."""
        base_key, branch = self.split_branch_key(key)
        if branch != self.BRANCH_MAIN:
            return key
        current = self.get_current_branch(base_key)
        return self.make_branch_key(base_key, current)

    def get_or_create(self, key: str) -> Session:
        """
        Get an existing session or create a new one.

        Args:
            key: Session key (usually channel:chat_id or a branch session key).

        Returns:
            The session.
        """
        if key in self._cache:
            return self._cache[key]

        session = self._load(key)
        if session is None:
            session = Session(key=key)

        self._cache[key] = session
        return session

    def clone_session(self, source: Session, dest_key: str) -> Session:
        """Create a new session by copying messages/metadata from another session."""
        cloned = Session(
            key=dest_key,
            messages=deepcopy(source.messages),
            created_at=datetime.now(),
            updated_at=datetime.now(),
            metadata=deepcopy(source.metadata),
            last_consolidated=source.last_consolidated,
        )
        self.save(cloned)
        return cloned

    def switch_branch(self, base_key: str, branch: str) -> tuple[Session, bool]:
        """Switch the active branch, creating it from the current branch if needed."""
        branch_name = (branch or self.BRANCH_MAIN).strip() or self.BRANCH_MAIN
        target_key = self.make_branch_key(base_key, branch_name)
        target_path = self._get_session_path(target_key)
        created = False

        if target_key in self._cache:
            session = self._cache[target_key]
        elif target_path.exists():
            session = self.get_or_create(target_key)
        else:
            if branch_name == self.BRANCH_MAIN:
                session = self.get_or_create(base_key)
            else:
                source_key = self.resolve_active_key(base_key)
                source = self.get_or_create(source_key)
                session = self.clone_session(source, target_key)
                created = True

        self.set_current_branch(base_key, branch_name)
        return session, created

    def list_branches(self, base_key: str) -> list[dict[str, Any]]:
        """List all branches known for a base session."""
        current = self.get_current_branch(base_key)
        branch_map: dict[str, dict[str, Any]] = {}

        for path in self.sessions_dir.glob("*.jsonl"):
            try:
                with open(path, encoding="utf-8") as f:
                    first_line = f.readline().strip()
                    if not first_line:
                        continue
                    data = json.loads(first_line)
                    if data.get("_type") != "metadata":
                        continue
                    key = data.get("key") or path.stem.replace("_", ":", 1)
                    session_base, branch = self.split_branch_key(key)
                    if session_base != base_key:
                        continue
                    branch_map[branch] = {
                        "branch": branch,
                        "key": key,
                        "created_at": data.get("created_at"),
                        "updated_at": data.get("updated_at"),
                        "current": branch == current,
                    }
            except Exception:
                continue

        if self.BRANCH_MAIN not in branch_map:
            branch_map[self.BRANCH_MAIN] = {
                "branch": self.BRANCH_MAIN,
                "key": base_key,
                "created_at": None,
                "updated_at": None,
                "current": self.BRANCH_MAIN == current,
            }

        return sorted(
            branch_map.values(),
            key=lambda item: (0 if item["branch"] == self.BRANCH_MAIN else 1, item["branch"]),
        )

    def _load(self, key: str) -> Session | None:
        """Load a session from disk."""
        path = self._get_session_path(key)
        if not path.exists():
            return None

        try:
            messages = []
            metadata = {}
            created_at = None
            last_consolidated = 0

            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    data = json.loads(line)

                    if data.get("_type") == "metadata":
                        metadata = data.get("metadata", {})
                        created_at = datetime.fromisoformat(data["created_at"]) if data.get("created_at") else None
                        last_consolidated = data.get("last_consolidated", 0)
                    else:
                        messages.append(data)

            return Session(
                key=key,
                messages=messages,
                created_at=created_at or datetime.now(),
                metadata=metadata,
                last_consolidated=last_consolidated
            )
        except Exception as e:
            logger.warning("Failed to load session {}: {}", key, e)
            return None

    def save(self, session: Session) -> None:
        """Save a session to disk."""
        path = self._get_session_path(session.key)

        metadata_line = {
            "_type": "metadata",
            "key": session.key,
            "created_at": session.created_at.isoformat(),
            "updated_at": session.updated_at.isoformat(),
            "metadata": session.metadata,
            "last_consolidated": session.last_consolidated
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps(metadata_line, ensure_ascii=False) + "\n")
            for msg in session.messages:
                f.write(json.dumps(msg, ensure_ascii=False) + "\n")

        self._cache[session.key] = session

    def archive_and_reset(self, key: str) -> bool:
        """Archive the current session JSONL and reset the active file."""
        session = self.get_or_create(key)
        primary = self._get_session_path(key)

        had_history = bool(session.messages) or primary.exists()
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        suffix = uuid4().hex[:8]
        safe_key = safe_filename(key.replace(":", "_"))
        archive_name = f"{safe_key}.{ts}.{suffix}.jsonl"

        if primary.exists():
            self.archive_dir.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy2(str(primary), str(self.archive_dir / archive_name))

        session.clear()
        self.save(session)
        self.invalidate(key)
        return had_history

    def invalidate(self, key: str) -> None:
        """Remove a session from the in-memory cache."""
        self._cache.pop(key, None)

    def list_sessions(self) -> list[dict[str, Any]]:
        """
        List all sessions.

        Returns:
            List of session info dicts.
        """
        sessions = []

        for path in self.sessions_dir.glob("*.jsonl"):
            try:
                # Read just the metadata line
                with open(path, encoding="utf-8") as f:
                    first_line = f.readline().strip()
                    if first_line:
                        data = json.loads(first_line)
                        if data.get("_type") == "metadata":
                            key = data.get("key") or path.stem.replace("_", ":", 1)
                            sessions.append({
                                "key": key,
                                "created_at": data.get("created_at"),
                                "updated_at": data.get("updated_at"),
                                "path": str(path)
                            })
            except Exception:
                continue

        return sorted(sessions, key=lambda x: x.get("updated_at", ""), reverse=True)
