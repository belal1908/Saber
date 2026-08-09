from holo.dsp.onset import ImpulseCapture


def test_not_pending_before_any_onset():
    capture = ImpulseCapture(window_s=0.08)
    assert not capture.pending
    assert not capture.ready(now=1000.0)


def test_onset_starts_a_deadline_window_s_later():
    capture = ImpulseCapture(window_s=0.08)
    capture.on_block(is_onset=True, now=100.0)
    assert capture.pending
    assert not capture.ready(now=100.05)  # too early — impulse hasn't rung in yet
    assert capture.ready(now=100.08)


def test_repeated_onsets_while_pending_do_not_push_the_deadline_back():
    """A tap's ringing can trigger multiple onset-like blocks; the deadline
    must stay anchored to the *first* detection, matching train.py's
    behavior of sleeping once right after the first onset, not restarting
    the wait on every subsequent loud block."""
    capture = ImpulseCapture(window_s=0.08)
    capture.on_block(is_onset=True, now=100.0)
    capture.on_block(is_onset=True, now=100.05)  # should not move the deadline
    assert capture.ready(now=100.08)


def test_consume_clears_pending_state_for_the_next_tap():
    capture = ImpulseCapture(window_s=0.08)
    capture.on_block(is_onset=True, now=100.0)
    capture.consume()
    assert not capture.pending
    assert not capture.ready(now=100.08)

    capture.on_block(is_onset=True, now=200.0)
    assert capture.pending
    assert capture.ready(now=200.08)
