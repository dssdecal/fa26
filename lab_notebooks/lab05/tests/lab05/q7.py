from otter.test_files import test_case
import numpy as np
from sklearn.model_selection import train_test_split

OK_FORMAT = False

name = "q7"
points = 1

@test_case(points=None, hidden=False)
def test_filter(meets_goal, threshold_table):
    expected = threshold_table[threshold_table["recall"] >= 0.70]
    assert list(meets_goal.index) == list(expected.index), "Keep exactly the threshold rows whose recall is at least 0.70."

@test_case(points=None, hidden=False)
def test_threshold(chosen_threshold, meets_goal):
    assert np.isclose(chosen_threshold, meets_goal["threshold"].max()), "Choose the highest threshold that meets the recall goal."
