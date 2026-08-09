from holo.model.diagnose import format_scores


def test_format_scores_ranks_highest_first():
    scores = {"rear-left": 0.1, "front-right": 0.6, "front-left": 0.2, "rear-right": 0.1}
    output = format_scores(scores, predicted="front-right")
    order = [output.index(zone) for zone in ("front-right", "front-left")]
    assert order == sorted(order)  # front-right (60%) appears before front-left (20%)


def test_format_scores_marks_the_predicted_zone():
    scores = {"rear-left": 0.7, "front-right": 0.1, "front-left": 0.1, "rear-right": 0.1}
    output = format_scores(scores, predicted="rear-left")
    assert "*rear-left" in output
    assert "*front-right" not in output
