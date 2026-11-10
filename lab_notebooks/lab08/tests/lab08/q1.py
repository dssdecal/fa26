from otter.test_files import test_case
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

OK_FORMAT = False
name = 'q1'
points = 1

@test_case(points=None, hidden=False)
def test_inspection(n_rows, n_missing, hall_names, dining):
    assert n_rows == len(dining), "Count all survey rows."
    assert n_missing == dining.rating.isna().sum(), "Count missing ratings."
    assert hall_names == set(dining.dining_hall), "Use the set of hall names."
