# holo-py

A Python/macOS reimplementation of [Holo](https://github.com/JustinGamer191/Holo) — turns the desk around your MacBook into four tap zones (rear-left, rear-right, front-left, front-right) using acoustic classification of tap impulses picked up by the mic.

Same pipeline as the original Swift app, different stack: `sounddevice` for mic capture, `numpy`/`scipy` for FFT and MFCC-style feature extraction, `scikit-learn` for the zone classifier, `rumps` for the menu-bar shell.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

macOS will prompt for microphone access the first time the app runs — grant it in System Settings → Privacy & Security → Microphone.

## Calibrate (train the zone classifier)

```bash
python -m holo.model.train --samples-per-zone 15
```

Follow the prompts: tap each zone the requested number of times. Saves weights to `data/model.json`.

## Configure zone actions

Edit `data/zone_config.json` (created on first run) to map each zone to a `shell`, `applescript`, or `keystroke` action. See `holo/actions/dispatch.py` for the supported action types.

## Run

```bash
python -m holo.app
```

Starts a menu-bar icon (◎) with Start/Stop Listening controls.

## Tests

```bash
pytest
```

## Architecture

```
holo/
├── audio/capture.py     # sounddevice mic stream + ring buffer
├── dsp/onset.py          # adaptive energy-threshold tap detection
├── dsp/features.py       # FFT -> mel filterbank -> MFCC + spectral shape
├── model/classifier.py   # regularized logistic regression, JSON-persisted
├── model/train.py        # interactive calibration CLI
├── actions/dispatch.py   # zone -> shell/AppleScript/keystroke
├── actions/registry.py   # zone -> action JSON config
└── app.py                # rumps menu-bar app tying it together
```
