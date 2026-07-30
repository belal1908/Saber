from __future__ import annotations

import json

from holo.config import ZONE_CONFIG_PATH, ZONES

DEFAULT_ACTIONS = {
    "rear-left": {"type": "shell", "command": "echo 'rear-left tapped'"},
    "rear-right": {"type": "shell", "command": "echo 'rear-right tapped'"},
    "front-left": {"type": "shell", "command": "echo 'front-left tapped'"},
    "front-right": {"type": "shell", "command": "echo 'front-right tapped'"},
}


def load_actions() -> dict[str, dict]:
    if not ZONE_CONFIG_PATH.exists():
        save_actions(DEFAULT_ACTIONS)
        return DEFAULT_ACTIONS
    data = json.loads(ZONE_CONFIG_PATH.read_text())
    return {zone: data.get(zone, DEFAULT_ACTIONS[zone]) for zone in ZONES}


def save_actions(actions: dict[str, dict]) -> None:
    ZONE_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    ZONE_CONFIG_PATH.write_text(json.dumps(actions, indent=2))
