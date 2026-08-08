from holo.model.train import _parse_device


def test_parse_device_numeric_string_becomes_int():
    assert _parse_device("2") == 2
    assert isinstance(_parse_device("0"), int)


def test_parse_device_name_stays_string():
    assert _parse_device("USB Mic") == "USB Mic"
