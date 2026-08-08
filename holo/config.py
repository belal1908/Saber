from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODEL_PATH = DATA_DIR / "model.json"
ZONE_CONFIG_PATH = DATA_DIR / "zone_config.json"

SAMPLE_RATE = 48_000
BLOCK_SIZE = 1024

# Onset detection
ONSET_WINDOW_MS = 10
ONSET_ENERGY_MULTIPLIER = 4.0  # spike must exceed noise floor * this

# Feature extraction window taken after an onset is detected
IMPULSE_WINDOW_MS = 80

ZONES = ["rear-left", "rear-right", "front-left", "front-right"]

# How often the runtime app fires a chirp when using a probe-mode model
PROBE_POLL_INTERVAL_S = 0.5
