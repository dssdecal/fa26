from otter.test_files import test_case
import pandas as pd

OK_FORMAT = False
name = "q6"

@test_case(points=1, success_message="Your input columns and target are correctly separated.")
def test_selected_inputs(X, y):
    original = pd.read_csv("study_sessions.csv", parse_dates=["date"])
    expected = pd.DataFrame({"crowded": original["occupancy_pct"] >= 70, "outlets": original["outlets"]})
    assert isinstance(X, pd.DataFrame), "X should be a DataFrame with the two input columns."
    assert isinstance(y, pd.Series), "y should be the focus_rating Series."
    pd.testing.assert_frame_equal(X, expected, obj="X (crowded and outlets, in that order)")
    pd.testing.assert_series_equal(y, original["focus_rating"], obj="y (the target focus_rating)")
