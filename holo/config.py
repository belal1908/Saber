from pathlib import Path

# Not project-relative (Path(__file__).parent.parent / "data") on purpose: inside
# a py2app bundle, pure-Python modules are zipped into python3XX.zip, so __file__
# resolves to a path *inside* that zip archive and any path arithmetic off it
# treats the zip file itself as if it were a directory, breaking mkdir(). Standard
# macOS per-user app data location works identically in dev and packaged builds.
DATA_DIR = Path.home() / "Library" / "Application Support" / "Saber"
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
