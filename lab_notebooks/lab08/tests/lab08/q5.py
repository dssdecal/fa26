from otter.test_files import test_case
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

OK_FORMAT = False
name = 'q5'
points = 1

@test_case(points=None, hidden=False)
def test_summary(summary, dining, original_dining):
    expected = original_dining.groupby("dining_hall").agg(mean_rating=("rating", "mean"), valid_responses=("rating", "count"), missing_responses=("rating", lambda s: s.isna().sum()))
    assert set(summary.columns) == set(expected.columns), "Use the three requested column names."
    pd.testing.assert_frame_equal(summary.sort_index()[expected.columns], expected, check_dtype=False)
    pd.testing.assert_frame_equal(dining, original_dining)
