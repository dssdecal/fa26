from otter.test_files import test_case
import pandas as pd

OK_FORMAT = False
name = "q5"

@test_case(points=1, success_message="Invalid ratings are missing and valid ratings are preserved.")
def test_ratings(clean):
    assert isinstance(clean, pd.DataFrame), "Run the earlier cleaning exercises first."
    assert len(clean) == 11 and clean.student_id.nunique() == 11, "Keep the rows; replace invalid rating values rather than deleting responses."
    expected = pd.Series([5., 4., 4., 5., 3., float("nan"), 4., float("nan"), 3., 4., 5.], index=range(101, 112))
    actual = clean.set_index("student_id").rating.sort_index()
    pd.testing.assert_series_equal(actual, expected, check_dtype=False, check_names=False, check_index_type=False, obj="Ratings (replace out-of-range entries with missing values)")
