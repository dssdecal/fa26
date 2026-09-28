from otter.test_files import test_case
import pandas as pd

OK_FORMAT = False
name = "q7"

@test_case(points=1, success_message="Your indicators correctly encode each study space.")
def test_space_indicators(space_indicators):
    original = pd.read_csv("study_sessions.csv", parse_dates=["date"])
    expected = pd.get_dummies(original["space"], prefix="space", dtype=int)
    assert isinstance(space_indicators, pd.DataFrame), "Save the indicator DataFrame as space_indicators."
    assert set(space_indicators.columns) == set(expected.columns), "Use one column per space with prefix='space'."
    assert all(pd.api.types.is_integer_dtype(dtype) for dtype in space_indicators.dtypes), "Use dtype=int for 0/1 indicators."
    pd.testing.assert_frame_equal(space_indicators[expected.columns], expected, check_dtype=False, obj="Space indicators (preserve each row's category)")
