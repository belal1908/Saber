from unittest.mock import patch

from holo.audio.capture import AudioCapture


def test_list_devices_preserves_true_sounddevice_index():
    """Filtering out non-input devices must not shift the index of the
    ones that remain — --device needs the index from the *unfiltered* list."""
    fake_devices = [
        {"name": "Output-only", "max_input_channels": 0, "max_output_channels": 2},
        {"name": "Built-in Mic", "max_input_channels": 1, "max_output_channels": 0},
        {"name": "USB Mic", "max_input_channels": 2, "max_output_channels": 0},
    ]
    with patch("holo.audio.capture.sd.query_devices", return_value=fake_devices):
        devices = AudioCapture.list_devices()

    assert [d["index"] for d in devices] == [1, 2]
    assert [d["name"] for d in devices] == ["Built-in Mic", "USB Mic"]
