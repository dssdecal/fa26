from otter.test_files import test_case
import pandas as pd

OK_FORMAT = False
name = "q8"

@test_case(points=1, success_message="Your grouped counts and means are correct and sorted.")
def test_grouped_summary(rating_summary):
    assert isinstance(rating_summary, pd.DataFrame), "Save your grouped table as rating_summary."
    expected = pd.DataFrame({"count": [4, 3, 2], "mean": [4.5, 4., 3.5]}, index=pd.Index(["game night", "study jam", "career panel"], name="event"))
    pd.testing.assert_frame_equal(rating_summary, expected, check_dtype=False, check_index_type=False, obj="Rating summary (count non-missing ratings and sort means descending)")
