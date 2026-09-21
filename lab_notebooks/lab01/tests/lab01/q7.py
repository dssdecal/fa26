from otter.test_files import test_case
import pandas as pd

OK_FORMAT = False
name = "q7"

@test_case(points=1, success_message="Your Boolean column and combined filter are correct.")
def test_combined_filter(clean, happy_long_visits):
    assert isinstance(clean, pd.DataFrame) and "long_visit" in clean, "Create clean['long_visit'] first."
    assert pd.api.types.is_bool_dtype(clean.long_visit), "long_visit should be a Boolean column."
    assert len(clean) == 11 and clean.student_id.nunique() == 11, "Preserve all respondents in clean; store the filtered result separately."
    assert set(clean.loc[clean.long_visit, "student_id"]) == {101, 104, 107, 108, 111}, "long_visit should mark every visit lasting at least two hours."
    assert isinstance(happy_long_visits, pd.DataFrame), "Save the filtered table as happy_long_visits."
    assert list(happy_long_visits.columns) == ["student_id", "event", "hours_spent", "rating"], "Keep the four requested columns."
    expected = pd.DataFrame({"student_id": [101, 104, 107, 111], "event": ["game night", "game night", "game night", "study jam"], "hours_spent": [2., 2.5, 3., 2.5], "rating": [5., 5., 4., 5.]}).set_index("student_id")
    actual = happy_long_visits.set_index("student_id").sort_index()
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False, check_index_type=False, obj="Responses meeting BOTH conditions")
