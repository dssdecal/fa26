from otter.test_files import test_case
import pandas as pd

OK_FORMAT = False
name = "q4"

@test_case(points=1, success_message="Labels are consistent and each response appears once.")
def test_labels_and_duplicates(clean, responses):
    assert isinstance(clean, pd.DataFrame), "Save the cleaned table as clean."
    assert len(clean) == 11 and clean.student_id.nunique() == 11, "Remove the exact repeated response while retaining all 11 students."
    expected = pd.Series(["game night", "study jam", "career panel", "game night", "study jam", "career panel", "game night", "study jam", "career panel", "game night", "study jam"], index=range(101, 112), name="event")
    actual = clean.set_index("student_id").event.sort_index()
    pd.testing.assert_series_equal(actual, expected, check_dtype=False, check_names=False, check_index_type=False, obj="Event labels (strip surrounding spaces and lowercase)")
    assert len(responses) == 12 and responses.event.nunique() == 8, "Preserve responses as the original data; clean a copy."
