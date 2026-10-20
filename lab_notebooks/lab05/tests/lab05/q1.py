from otter.test_files import test_case
import numpy as np
from sklearn.model_selection import train_test_split

OK_FORMAT = False

name = "q1"
points = 1

@test_case(points=None, hidden=False)
def test_sizes(train, validation, test):
    assert (len(train), len(validation), len(test)) == (1200, 400, 400), "Use a 60/20/20 split: 1200 training, 400 validation, and 400 test rows."

@test_case(points=None, hidden=False)
def test_rows_and_seed(orders, train, validation, test):
    expected_dev, expected_test = train_test_split(orders, test_size=0.2, random_state=42, stratify=orders["refund_requested"])
    expected_train, expected_validation = train_test_split(expected_dev, test_size=0.25, random_state=42, stratify=expected_dev["refund_requested"])
    assert train.index.equals(expected_train.index), "Use random_state=42 and stratify by refund_requested for the training/validation split."
    assert validation.index.equals(expected_validation.index), "Use 25% of dev as validation, with random_state=42 and stratification."
    assert test.index.equals(expected_test.index), "Use 20% of all rows as test, with random_state=42 and stratification."
