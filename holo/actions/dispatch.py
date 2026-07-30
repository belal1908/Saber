from __future__ import annotations

import shlex
import subprocess


def dispatch(zone: str, action: dict) -> None:
    kind = action.get("type")

    if kind == "shell":
        subprocess.Popen(shlex.split(action["command"]))
    elif kind == "applescript":
        subprocess.Popen(["osascript", "-e", action["script"]])
    elif kind == "keystroke":
        script = f'tell application "System Events" to keystroke "{action["key"]}"'
        subprocess.Popen(["osascript", "-e", script])
    else:
        raise ValueError(f"Unknown action type for zone {zone!r}: {kind!r}")
