from otter.test_files import test_case
import pandas as pd

OK_FORMAT = False
name = "q2"

@test_case(points=1, success_message="Your longer visits are selected and sorted correctly.")
def test_long_visits(long_visits):
    assert isinstance(long_visits, pd.DataFrame), "Save your result as the DataFrame long_visits."
    assert list(long_visits.columns) == ["student_id", "event", "hours_spent"], "Keep only the three requested columns, in the requested order."
    assert len(long_visits) == 5 and set(long_visits.student_id) == {101, 104, 107, 108, 111}, "Include every visit of at least 2 hours, including visits of exactly 2 hours."
    assert long_visits.hours_spent.is_monotonic_decreasing, "Sort hours_spent from highest to lowest."
    actual = long_visits.set_index("student_id").sort_index()
    expected = pd.DataFrame({"student_id": [101, 104, 107, 108, 111], "event": ["Game Night", "game night", "GAME NIGHT", "study jam", "Study Jam"], "hours_spent": [2, 2.5, 3, 2, 2.5]}).set_index("student_id")
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False, check_index_type=False, obj="Selected visits (keep the original values)")
