from otter.test_files import test_case
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

OK_FORMAT = False
name = 'q6'
points = 1

@test_case(points=None, hidden=False)
def test_validation(valid_ratings):
    for values, expected in [([1, 5, np.nan], True), ([0, 3], False), ([6, 2], False), ([-1], False), ([np.nan], True), ([2.5, 4], True)]:
        assert bool(valid_ratings(pd.DataFrame({"rating": values}))) == expected, "Check all non-missing values against the inclusive 1–5 range."
