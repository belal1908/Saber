# Saber

A Python/macOS reimplementation of [Holo](https://github.com/JustinGamer191/Holo) — turns the desk around your MacBook into four tap zones (rear-left, rear-right, front-left, front-right) using acoustic classification of tap impulses picked up by the mic.

Same pipeline as the original Swift app, different stack: `sounddevice` for mic capture, `numpy`/`scipy` for FFT, MFCC-style feature extraction, and adaptive noise filtering, `scikit-learn` for the zone classifier, `rumps` for the menu-bar shell.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

macOS will prompt for microphone access the first time the app runs — grant it in System Settings → Privacy & Security → Microphone.

## Calibrate (train the zone classifier)

Passive mode (tap each zone, detected via onset threshold):

```bash
python -m holo.model.train --samples-per-zone 15
```

Active-probe mode (fires a 15.5–21kHz chirp and classifies the desk's response instead of waiting for a tap):

```bash
python -m holo.model.train --samples-per-zone 15 --use-probe
```

Both modes track ambient noise (fan, typing, music) and subtract it from the impulse/response spectrum before extracting features, so training-time and runtime features stay matched. Follow the prompts: tap (or probe) each zone the requested number of times. Saves weights to `~/Library/Application Support/Saber/model.json`.

## Configure zone actions

Edit `~/Library/Application Support/Saber/zone_config.json` (created on first run) to map each zone to a `shell`, `applescript`, or `keystroke` action. See `holo/actions/dispatch.py` for the supported action types.

## Run

```bash
python -m holo.app
```

Starts a menu-bar icon (◎) with Start/Stop Listening controls. The app reads which mode the loaded `~/Library/Application Support/Saber/model.json` was trained in (`passive` or `probe`, recorded automatically by `holo.model.train`) and matches its runtime feature extraction to it — passive mode listens continuously with the same live adaptive noise filtering used during training; probe mode fires a chirp every `PROBE_POLL_INTERVAL_S` (0.5s by default, see `holo/config.py`) and classifies the response.

## Build a standalone .app

```bash
pip install -e ".[build]"
python setup.py py2app
```

Produces `dist/Saber.app` — a double-clickable menu-bar app that needs no terminal or venv. It requests mic access via the `NSMicrophoneUsageDescription` in its bundled `Info.plist`, and runs with no Dock icon (menu-bar only). Because it bundles numpy/scipy/scikit-learn, the build can take a few minutes.

## Tests

```bash
pytest
```

## Architecture

```
holo/
├── audio/capture.py         # sounddevice mic stream + ring buffer
├── audio/probe.py            # active chirp probe (15.5-21kHz) emit/capture
├── dsp/onset.py               # adaptive energy-threshold tap detection
├── dsp/adaptive_filter.py     # spectral subtraction against tracked ambient noise
├── dsp/features.py            # FFT -> mel filterbank -> MFCC + spectral shape
├── model/classifier.py        # regularized logistic regression, JSON-persisted
├── model/train.py             # interactive calibration CLI (passive or probe)
├── actions/dispatch.py        # zone -> shell/AppleScript/keystroke
├── actions/registry.py        # zone -> action JSON config
└── app.py                     # rumps menu-bar app tying it together
```
