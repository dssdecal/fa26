from otter.test_files import test_case
import pandas as pd

OK_FORMAT = False
name = "q6"

@test_case(points=1, success_message="Hours are filled using the median, with their missingness recorded.")
def test_missing_hours(clean, median_hours):
    assert isinstance(clean, pd.DataFrame), "Save your cleaned table as clean."
    assert median_hours == 1.75, "Compute the median from the known hours after removing duplicates."
    assert "hours_were_missing" in clean, "Record which hours were missing before filling."
    assert pd.api.types.is_bool_dtype(clean.hours_were_missing), "hours_were_missing should contain True/False values."
    assert set(clean.loc[clean.hours_were_missing, "student_id"]) == {105}, "Flag the originally missing hours. If you reran Q6, rerun from Q4 to restore the missing value first."
    assert len(clean) == 11 and clean.student_id.nunique() == 11, "Retain all 11 unique respondents."
    expected = pd.Series([2., 1.5, 1., 2.5, 1.75, 1.5, 3., 2., 1., 1.5, 2.5], index=range(101, 112))
    actual = clean.set_index("student_id").hours_spent.sort_index()
    pd.testing.assert_series_equal(actual, expected, check_dtype=False, check_names=False, check_index_type=False, obj="Hours (fill only the missing entry)")
    assert clean.rating.isna().sum() == 2, "Leave the two missing ratings missing."
