"""Streaming renderer for CLI output.

Uses Rich Live with auto_refresh=False for stable, flicker-free
markdown rendering during streaming. Ellipsis mode handles overflow.
"""

from __future__ import annotations

import sys
import time
from contextlib import contextmanager

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.text import Text

from clicomp import __logo__


def _make_console() -> Console:
    return Console(file=sys.stdout)


class ThinkingSpinner:
    """Spinner that shows 'clicomp is thinking...' with pause support."""

    def __init__(self, console: Console | None = None):
        self._console = console or _make_console()
        self._spinner = self._console.status("[dim]clicomp is thinking...[/dim]", spinner="dots")
        self._active = False
        self._visible = False

    def __enter__(self):
        self._spinner.start()
        self._active = True
        self._visible = True
        return self

    def __exit__(self, *exc):
        self.stop(clear=True)
        return False

    def stop(self, *, clear: bool = False) -> None:
        if self._spinner and self._visible:
            self._spinner.stop()
            self._visible = False
            if clear:
                self._console.print(end="\r")
                self._console.file.write("\033[2K")
                self._console.file.flush()
        self._active = False

    def pause(self):
        """Context manager: temporarily hide spinner for clean output."""

        @contextmanager
        def _ctx():
            was_active = self._active and self._visible
            if was_active:
                self.stop(clear=True)
            try:
                yield
            finally:
                if was_active:
                    self._spinner.start()
                    self._visible = True
                    self._active = True

        return _ctx()


class StreamRenderer:
    """Rich Live streaming with markdown. auto_refresh=False avoids render races.

    Deltas arrive pre-filtered (no <think> tags) from the agent loop.

    Flow per round:
      spinner -> first visible delta -> header + Live renders ->
      on_end -> Live stops (content stays on screen)
    """

    def __init__(self, render_markdown: bool = True, show_spinner: bool = True):
        self._md = render_markdown
        self._show_spinner = show_spinner
        self._buf = ""
        self._live: Live | None = None
        self._t = 0.0
        self.streamed = False
        self._spinner: ThinkingSpinner | None = None
        self._start_spinner()

    def _render(self):
        return Markdown(self._buf) if self._md and self._buf else Text(self._buf or "")

    def _start_spinner(self) -> None:
        if self._show_spinner:
            self._spinner = ThinkingSpinner()
            self._spinner.__enter__()

    def _stop_spinner(self) -> None:
        if self._spinner:
            self._spinner.__exit__(None, None, None)
            self._spinner = None

    async def on_delta(self, delta: str) -> None:
        self.streamed = True
        self._buf += delta
        if self._live is None:
            if not self._buf.strip():
                return
            self._stop_spinner()
            c = _make_console()
            c.print()
            c.print(f"[cyan]{__logo__} clicomp[/cyan]")
            self._live = Live(self._render(), console=c, auto_refresh=False)
            self._live.start()
        now = time.monotonic()
        if "\n" in delta or (now - self._t) > 0.05:
            self._live.update(self._render())
            self._live.refresh()
            self._t = now

    async def on_end(self, *, resuming: bool = False) -> None:
        if self._live:
            self._live.update(self._render())
            self._live.refresh()
            self._live.stop()
            self._live = None
        self._stop_spinner()
        if resuming:
            self._buf = ""
            self._start_spinner()
        else:
            _make_console().print()

    async def close(self) -> None:
        """Stop spinner/live without rendering a final streamed round."""
        if self._live:
            self._live.stop()
            self._live = None
        self._stop_spinner()
