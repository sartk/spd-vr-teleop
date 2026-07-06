"""TUI helpers: log tee for the streamer, supervisor app for users.

This module hosts two things:

1. `install_log_tee()` — wraps the streamer's stdout/stderr so each line goes
   to the original terminal stream AND a file inside the session directory.
   The streamer calls this at boot.

2. `StreamerSupervisorApp` (and `main()` / entry point `mujoco-vr-tui`) — a
   Textual supervisor that runs in its own process. It launches the streamer
   subprocess, tails its stdout pipe, polls /api/status, and POSTs to
   /api/command to drive the streamer. Lets the user pick or switch scenes
   without restarting the supervisor.
"""
from __future__ import annotations

import io
import json
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from pathlib import Path
from typing import IO


# --------------------------------------------------------------------------- #
# Log tee (used by the streamer process itself)                               #
# --------------------------------------------------------------------------- #


class _LogTee(io.TextIOBase):
    """Line-buffered stream that fans writes to the original terminal stream
    and (once a path is set) to a log file. Pre-attachment lines are
    flushed when set_log_file() runs.
    """

    def __init__(self, passthrough: IO[str]) -> None:
        self._passthrough = passthrough
        self._partial = ""
        self._lock = threading.Lock()
        self._files: list[IO[str]] = []
        self._file_lock = threading.Lock()
        self._pending_for_file: list[str] = []

    def writable(self) -> bool:
        return True

    def write(self, s: str) -> int:
        if not isinstance(s, str):
            return 0
        try:
            self._passthrough.write(s)
        except Exception:
            pass
        with self._lock:
            self._partial += s
            while "\n" in self._partial:
                line, self._partial = self._partial.split("\n", 1)
                self._dispatch_line(line)
        return len(s)

    def flush(self) -> None:
        try:
            self._passthrough.flush()
        except Exception:
            pass
        with self._file_lock:
            for f in self._files:
                try:
                    f.flush()
                except Exception:
                    pass

    def _dispatch_line(self, line: str) -> None:
        with self._file_lock:
            if self._files:
                for f in self._files:
                    try:
                        f.write(line + "\n")
                        f.flush()
                    except Exception:
                        pass
            else:
                self._pending_for_file.append(line)

    def set_log_file(self, path: Path) -> None:
        """Add another log file the tee will write to (does NOT replace
        previous files). The first call also flushes any pending pre-attach
        lines into the new file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._file_lock:
            f = open(path, "a", buffering=1, encoding="utf-8")
            # Flush pending only on the FIRST attached file so we don't
            # duplicate boot-time prints into every subsequent log.
            if not self._files:
                for line in self._pending_for_file:
                    try:
                        f.write(line + "\n")
                    except Exception:
                        pass
                self._pending_for_file.clear()
            self._files.append(f)
            try:
                f.flush()
            except Exception:
                pass


def install_log_tee() -> tuple[_LogTee, _LogTee]:
    """Replace sys.stdout/sys.stderr with line-buffered tees that also write to
    a file once `set_log_file(path)` is called. Returns the (stdout, stderr)
    tee instances so the caller can attach files later.
    """
    stdout_tee = _LogTee(sys.__stdout__)
    stderr_tee = _LogTee(sys.__stderr__)
    sys.stdout = stdout_tee  # type: ignore[assignment]
    sys.stderr = stderr_tee  # type: ignore[assignment]
    return stdout_tee, stderr_tee


# --------------------------------------------------------------------------- #
# Supervisor TUI (separate process)                                           #
# --------------------------------------------------------------------------- #

def _discover_scene_types() -> tuple[str, ...]:
    """Source of truth: the scene_builder's SceneType literal members."""
    try:
        from mujoco_vr_teleop.scene_builder import _task_scenes
        return tuple(_task_scenes())
    except Exception:
        return ("jenga", "spell_and_stow")


SCENE_TYPES = _discover_scene_types()

TUI_KEY_COMMANDS: dict[str, str] = {
    "s": "save_checkpoint",
    "p": "toggle_pause",
    "d": "discard",
    "r": "reset",
    "k": "set_keyframe",
    "[": "prev_task",
    "]": "next_task",
    "up": "raise_table",
    "down": "lower_table",
}


class StreamerHandle:
    """Owns a subprocess running `mujoco-vr-stream`, plus a thread that
    drains its stdout/stderr into a shared deque.
    """

    def __init__(
        self,
        stream_argv: list[str],
        log_buffer: deque[tuple[str, str]],
    ) -> None:
        self._argv = stream_argv
        self._log_buffer = log_buffer
        self._proc: subprocess.Popen | None = None
        self._stdout_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._exit_code: int | None = None

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    @property
    def exit_code(self) -> int | None:
        if self._proc is None:
            return None
        return self._proc.poll()

    @property
    def argv(self) -> list[str]:
        return list(self._argv)

    def start(self) -> None:
        if self.running:
            return
        self._exit_code = None
        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        self._proc = subprocess.Popen(
            self._argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
        )
        self._stdout_thread = threading.Thread(
            target=self._drain_pipe,
            args=(self._proc.stdout, "stdout"),
            daemon=True,
            name="streamer-stdout",
        )
        self._stderr_thread = threading.Thread(
            target=self._drain_pipe,
            args=(self._proc.stderr, "stderr"),
            daemon=True,
            name="streamer-stderr",
        )
        self._stdout_thread.start()
        self._stderr_thread.start()
        self._log_buffer.append(("stdout", f"[supervisor] launched: {' '.join(shlex.quote(a) for a in self._argv)}"))

    def _drain_pipe(self, pipe: IO[str] | None, kind: str) -> None:
        if pipe is None:
            return
        try:
            for line in iter(pipe.readline, ""):
                if not line:
                    break
                self._log_buffer.append((kind, line.rstrip("\n")))
        except Exception:
            pass
        finally:
            try:
                pipe.close()
            except Exception:
                pass

    def stop(self, timeout: float = 5.0) -> None:
        if self._proc is None:
            return
        if self._proc.poll() is not None:
            return
        try:
            self._proc.send_signal(signal.SIGTERM)
            self._proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self._log_buffer.append(("stderr", "[supervisor] streamer SIGTERM timeout, sending SIGKILL"))
            self._proc.kill()
            try:
                self._proc.wait(timeout=2.0)
            except Exception:
                pass
        except Exception as exc:
            self._log_buffer.append(("stderr", f"[supervisor] stop error: {exc}"))


# --------------------------------------------------------------------------- #
# HTTP helpers                                                                #
# --------------------------------------------------------------------------- #


def _http_get_json(url: str, timeout: float = 1.0) -> dict | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


def _http_post_json(url: str, body: dict, timeout: float = 1.0) -> dict | None:
    try:
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}, method="POST"
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Textual supervisor                                                          #
# --------------------------------------------------------------------------- #


def _import_textual():
    """Lazy-import Textual so the streamer side doesn't need it loaded."""
    from rich.align import Align
    from rich.columns import Columns
    from rich.console import Group
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Vertical
    from textual.screen import ModalScreen
    from textual.widgets import Button, RichLog, Static
    from textual.widgets import OptionList
    from textual.widgets.option_list import Option

    return {
        "Align": Align,
        "Columns": Columns,
        "Group": Group,
        "Panel": Panel,
        "Table": Table,
        "Text": Text,
        "App": App,
        "ComposeResult": ComposeResult,
        "Binding": Binding,
        "Vertical": Vertical,
        "ModalScreen": ModalScreen,
        "Button": Button,
        "RichLog": RichLog,
        "Static": Static,
        "OptionList": OptionList,
        "Option": Option,
    }


def _foot_pedals_renderable(tx):
    PEDAL_COLORS = {"A": "green", "B": "yellow", "C": "red"}

    def pedal(letter: str, label: str):
        color = PEDAL_COLORS[letter]
        body = tx["Text"].assemble(
            ("\n",),
            (f" {letter} ", f"bold black on {color}"),
            "\n\n",
            (label, "bold white"),
            justify="center",
        )
        return tx["Panel"](body, width=20, padding=(0, 1), border_style=color)

    pedals = tx["Columns"](
        [pedal("A", "checkpoint"), pedal("B", "pause/resume"), pedal("C", "reset")],
        padding=(0, 2),
        align="center",
        expand=True,
    )
    return tx["Group"](
        tx["Align"].center(tx["Text"]("◉ Foot Pedals ◉", style="bold bright_white")),
        pedals,
    )


def _status_renderable(tx, state: str, lines: list[tuple[str, str]]):
    mode_color = {
        "PLAYGROUND": "magenta",
        "PAUSED": "yellow",
        "RECORDING": "bold green",
        "ACTIVE": "cyan",
        "IDLE": "white",
        "OFFLINE": "bright_black",
    }.get(state, "white")
    badge = tx["Text"].assemble(("● ", mode_color), (state, f"bold {mode_color}"))
    rows = tx["Table"].grid(padding=(0, 2))
    rows.add_column(style="dim", no_wrap=True)
    rows.add_column(style="white")
    for label, value in lines:
        rows.add_row(label, value)
    return tx["Group"](badge, tx["Text"](""), rows)


def build_supervisor_app(
    tx,
    *,
    stream_argv_factory,
    initial_scene_type: str | None,
    port: int,
):
    """Construct the Textual supervisor app class with closures over the
    process-launch factory and the HTTP base URL.
    """
    log_buffer: deque[tuple[str, str]] = deque(maxlen=10000)
    handle: dict[str, StreamerHandle | None] = {"current": None}
    pending_scene: dict[str, str | None] = {"value": initial_scene_type}

    base_url = f"http://127.0.0.1:{port}"

    BAR_GLYPHS = " ▁▂▃▄▅▆▇█"

    def _render_scene_row(scene_type: str):
        """Build a Rich renderable showing scene name + 7-day total +
        bar chart of daily minutes (with 'Xh Ym' captions over each bar).
        """
        from . import state_store

        metrics = state_store.load_scene_metrics(scene_type, days=7)
        total = sum(m for _, m, _ in metrics)
        max_minutes = max((m for _, m, _ in metrics), default=0.0) or 1.0

        # Top row: 7 cells of "X h Y m" or blank if zero, each cell width 7.
        cell_w = 7
        captions = []
        bars = []
        for _, m, _ in metrics:
            if m > 0:
                txt = state_store.format_minutes(m)
            else:
                txt = ""
            captions.append(txt[:cell_w].center(cell_w))
            # Map m to glyph height (8 levels).
            height = int(round((m / max_minutes) * (len(BAR_GLYPHS) - 1)))
            bar_glyph = BAR_GLYPHS[height]
            bars.append(bar_glyph.center(cell_w))
        caption_line = "".join(captions)
        bars_line = "".join(bars)

        name = scene_type.replace("_", " ").title()
        total_label = state_store.format_minutes(total) if total > 0 else "—"
        # Two-line right pane: captions (small dim text) above bars (bright).
        chart = tx["Text"].assemble(
            (caption_line, "dim"),
            "\n",
            (bars_line, "bold cyan"),
        )

        layout = tx["Table"].grid(padding=(0, 2), expand=True)
        layout.add_column(width=18, no_wrap=True)
        layout.add_column(width=12, no_wrap=True)
        layout.add_column()
        layout.add_row(
            tx["Text"](name, style="bold white"),
            tx["Text"](f"week  {total_label}", style="green" if total > 0 else "dim"),
            chart,
        )
        return layout

    class ScenePicker(tx["ModalScreen"]):
        BINDINGS = [tx["Binding"]("escape", "cancel", "cancel")]

        CSS = """
        ScenePicker { align: center middle; }
        #picker { width: 88; height: auto; max-height: 80%; border: round $accent; padding: 1 2; }
        #picker-title { padding-bottom: 1; text-align: center; }
        #picker-list { height: auto; max-height: 24; }
        """

        def compose(self):
            with tx["Vertical"](id="picker"):
                yield tx["Static"](
                    "Pick a scene  (↑/↓ to navigate, Enter to select, Esc to cancel)",
                    id="picker-title",
                )
                options = [
                    tx["Option"](_render_scene_row(name), id=name)
                    for name in SCENE_TYPES
                ]
                yield tx["OptionList"](*options, id="picker-list")

        def on_mount(self) -> None:
            try:
                opt_list = self.query_one("#picker-list", tx["OptionList"])
                opt_list.focus()
                opt_list.highlighted = 0
            except Exception:
                pass

        def on_option_list_option_selected(self, event) -> None:  # type: ignore[override]
            chosen = event.option.id
            if chosen and chosen in SCENE_TYPES:
                self.dismiss(chosen)
            else:
                self.dismiss(None)

        def action_cancel(self) -> None:
            self.dismiss(None)

    class StreamerSupervisorApp(tx["App"]):
        CSS = """
        Screen { layout: vertical; background: $surface; }
        #status { height: 8; border: round $primary; padding: 0 1; }
        #commands { height: 11; border: round $accent; padding: 0 1; }
        #log { border: round $secondary; min-height: 8; }
        """

        BINDINGS = [
            tx["Binding"]("s", "send('save_checkpoint')", "save"),
            tx["Binding"]("p", "send('toggle_pause')", "pause"),
            tx["Binding"]("d", "send('discard')", "discard"),
            tx["Binding"]("r", "send('reset')", "reset"),
            tx["Binding"]("k", "send('set_keyframe')", "keyframe"),
            tx["Binding"]("left_square_bracket", "send('prev_task')", "prev task"),
            tx["Binding"]("right_square_bracket", "send('next_task')", "next task"),
            tx["Binding"]("up", "send('raise_table')", "raise table"),
            tx["Binding"]("down", "send('lower_table')", "lower table"),
            tx["Binding"]("w", "switch_scene", "switch scene"),
            tx["Binding"]("q", "quit", "quit"),
        ]

        def compose(self):
            with tx["Vertical"]():
                self._status_widget = tx["Static"](
                    _status_renderable(tx, "OFFLINE", [("status", "(starting...)")]),
                    id="status",
                    expand=True,
                )
                self._status_widget.border_title = "Status"
                yield self._status_widget

                with tx["Vertical"](id="commands"):
                    yield tx["Static"](
                        _foot_pedals_renderable(tx),
                        id="commands-pedals",
                        expand=True,
                    )

                self._log_widget = tx["RichLog"](
                    id="log",
                    wrap=True,
                    highlight=False,
                    markup=True,
                    auto_scroll=True,
                )
                self._log_widget.border_title = "Log"
                yield self._log_widget

        def on_mount(self) -> None:
            self.query_one("#commands", tx["Vertical"]).border_title = (
                "Controls — s save · p pause · d discard · r reset · k keyframe · ↑/↓ table · W switch scene · Q quit"
            )
            self._last_log_index = 0
            if pending_scene["value"] is None:
                self._open_picker()
            else:
                self._launch_streamer(pending_scene["value"])
            self.set_interval(0.1, self._drain_log)
            self.set_interval(0.4, self._refresh_status)

        # --- streamer lifecycle --- #

        def _launch_streamer(self, scene_type: str) -> None:
            self._stop_current()
            argv = stream_argv_factory(scene_type)
            handle["current"] = StreamerHandle(argv, log_buffer)
            handle["current"].start()
            pending_scene["value"] = scene_type

        def _stop_current(self) -> None:
            cur = handle["current"]
            if cur is None:
                return
            log_buffer.append(("stdout", "[supervisor] stopping streamer..."))
            cur.stop()
            handle["current"] = None

        def action_switch_scene(self) -> None:
            self._open_picker()

        def _open_picker(self) -> None:
            def _on_pick(choice):
                if choice is None:
                    return
                self._launch_streamer(choice)
            self.push_screen(ScenePicker(), _on_pick)

        # --- HTTP polling --- #

        def _refresh_status(self) -> None:
            cur = handle["current"]
            if cur is None or not cur.running:
                code = cur.exit_code if cur is not None else None
                if cur is not None and code is not None:
                    # Streamer died — surface the last few stderr lines
                    # right in the status pane so the user can see the
                    # traceback without hunting in the log pane.
                    tail = []
                    for kind, line in reversed(log_buffer):
                        if kind == "stderr" and not line.startswith("[supervisor]"):
                            tail.append(line)
                            if len(tail) >= 8:
                                break
                    rows: list[tuple[str, str]] = [
                        ("status", f"streamer exited (code {code})"),
                    ]
                    for line in reversed(tail):
                        rows.append(("stderr", line[:120]))
                    rows.append(("hint", "press W to pick a scene"))
                    self._status_widget.update(
                        _status_renderable(tx, "OFFLINE", rows),
                    )
                else:
                    self._status_widget.update(
                        _status_renderable(
                            tx,
                            "OFFLINE",
                            [("status", "no streamer running"), ("hint", "press W to pick a scene")],
                        )
                    )
                return
            data = _http_get_json(f"{base_url}/api/status", timeout=0.5)
            if not data:
                self._status_widget.update(
                    _status_renderable(
                        tx,
                        "OFFLINE",
                        [("status", "waiting for streamer to come online...")],
                    )
                )
                return
            mode = (data.get("state") or "idle").upper()
            rows: list[tuple[str, str]] = []
            rows.append(
                (
                    "episode",
                    f"{data.get('episode', 0)}   "
                    f"frames {data.get('frame_count', 0)}   "
                    f"checkpoints {data.get('marker_count', 0)}   "
                    f"recorded {(data.get('recorded_minutes') or 0):.1f} min",
                )
            )
            task = data.get("task") or {}
            instr = (task.get("instruction") or "").strip()
            if instr:
                idx = task.get("task_index")
                total = task.get("total_tasks")
                if isinstance(idx, int) and isinstance(total, int):
                    rows.append(("task", f"{idx + 1}/{total}: {instr[:80]}"))
                else:
                    rows.append(("task", instr[:80]))
            scene = pending_scene["value"] or "?"
            rows.append(("scene", scene))
            self._status_widget.update(_status_renderable(tx, mode, rows))

        # --- log drain --- #

        def _drain_log(self) -> None:
            snapshot = list(log_buffer)
            new_lines = snapshot[self._last_log_index:]
            self._last_log_index = len(snapshot)
            for kind, line in new_lines:
                safe = line.replace("[", r"\[")
                if kind == "stderr":
                    self._log_widget.write(f"[red]{safe}[/red]")
                else:
                    self._log_widget.write(safe)

        # --- command dispatch --- #

        def action_send(self, command: str) -> None:
            cur = handle["current"]
            if cur is None or not cur.running:
                log_buffer.append(("stderr", f"[supervisor] cannot send '{command}': streamer not running"))
                return
            result = _http_post_json(
                f"{base_url}/api/command", {"command": command}, timeout=1.0
            )
            if result is None:
                log_buffer.append(("stderr", f"[supervisor] failed to send '{command}'"))

        async def action_quit(self) -> None:
            self._stop_current()
            self.exit()

        def on_unmount(self) -> None:
            self._stop_current()

    return StreamerSupervisorApp


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #


def _streamer_executable() -> list[str]:
    """Return the argv prefix for invoking the streamer.

    Prefer the installed `mujoco-vr-stream` script; fall back to running the
    module via the current Python interpreter so dev installs without entry
    points still work.
    """
    import shutil

    exe = shutil.which("mujoco-vr-stream")
    if exe:
        return [exe]
    return [sys.executable, "-m", "mujoco_vr_teleop.vr_streamer"]


def _split_supervisor_args(
    argv: list[str],
) -> tuple[str | None, int, str | None, list[str]]:
    """Pull --builder.scene-type, --port, and --mode out of argv (if present)
    and return the remaining args verbatim for forwarding to the streamer.
    """
    scene_type: str | None = None
    port = 8012
    mode: str | None = None
    forward: list[str] = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--builder.scene-type":
            scene_type = argv[i + 1] if i + 1 < len(argv) else None
            i += 2
            continue
        if a.startswith("--builder.scene-type="):
            scene_type = a.split("=", 1)[1]
            i += 1
            continue
        if a == "--port":
            try:
                port = int(argv[i + 1])
            except (ValueError, IndexError):
                pass
            forward.extend(argv[i:i + 2])
            i += 2
            continue
        if a.startswith("--port="):
            try:
                port = int(a.split("=", 1)[1])
            except ValueError:
                pass
            forward.append(a)
            i += 1
            continue
        if a == "--mode":
            mode = argv[i + 1] if i + 1 < len(argv) else None
            forward.extend(argv[i:i + 2])
            i += 2
            continue
        if a.startswith("--mode="):
            mode = a.split("=", 1)[1]
            forward.append(a)
            i += 1
            continue
        forward.append(a)
        i += 1
    if scene_type not in (None, *SCENE_TYPES):
        sys.stderr.write(
            f"Unknown scene type {scene_type!r}; valid: {', '.join(SCENE_TYPES)}\n"
        )
        scene_type = None
    return scene_type, port, mode, forward


COLLECTOR_ONLY_SCENES = ("jenga", "spell_and_stow", "dishrack", "hang_mugs", "bottles_in_bin", "cup_stack")


def main() -> None:
    scene_type, port, mode, forward = _split_supervisor_args(sys.argv[1:])
    base = _streamer_executable()

    # In collector mode the operator can only pick scenes that are fully
    # authored / ready for data collection. Narrow the picker accordingly.
    global SCENE_TYPES
    if mode == "collector":
        SCENE_TYPES = tuple(s for s in SCENE_TYPES if s in COLLECTOR_ONLY_SCENES)
        if scene_type is not None and scene_type not in SCENE_TYPES:
            sys.stderr.write(
                f"Scene {scene_type!r} is not enabled in collector mode; "
                f"valid: {', '.join(SCENE_TYPES)}\n"
            )
            scene_type = None

    def stream_argv_factory(chosen_scene: str) -> list[str]:
        return [*base, "--builder.scene-type", chosen_scene, *forward]

    try:
        tx = _import_textual()
    except ImportError as exc:
        sys.stderr.write(
            "Textual is required for the supervisor TUI. "
            "Install with: pip install textual\n"
            f"({exc})\n"
        )
        sys.exit(1)

    AppCls = build_supervisor_app(
        tx,
        stream_argv_factory=stream_argv_factory,
        initial_scene_type=scene_type,
        port=port,
    )
    app = AppCls()
    app.run()


if __name__ == "__main__":
    main()
