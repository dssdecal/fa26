from otter.test_files import test_case
import pandas as pd

OK_FORMAT = False
name = "q5"

@test_case(points=1, success_message="Your crowded feature is correct and the original data are unchanged.")
def test_crowded(features, sessions):
    original = pd.read_csv("study_sessions.csv", parse_dates=["date"])
    pd.testing.assert_frame_equal(sessions, original, obj="Original sessions (work on a copy)")
    assert isinstance(features, pd.DataFrame), "Save your working table as features."
    assert list(features.columns) == list(original.columns) + ["crowded"], "Keep the original columns and add crowded."
    pd.testing.assert_frame_equal(features.drop(columns="crowded"), original, obj="Original feature columns")
    assert pd.api.types.is_bool_dtype(features["crowded"]), "crowded should contain Boolean True/False values."
    pd.testing.assert_series_equal(features["crowded"], (original["occupancy_pct"] >= 70).rename("crowded"), obj="crowded (include exactly 70%)")
