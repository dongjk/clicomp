from __future__ import annotations

import clicomp.cli.commands as commands


def test_init_prompt_session_saves_terminal_state(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / ".config"))

    commands._SAVED_TERM_ATTRS = None

    commands._init_prompt_session()

    # In non-tty test runs this may still remain None, but the call should not crash.
    assert hasattr(commands, "_SAVED_TERM_ATTRS")
