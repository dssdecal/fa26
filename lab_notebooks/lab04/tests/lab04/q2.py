from otter.test_files import test_case
import pandas as pd

OK_FORMAT = False
name = "q2"


@test_case(points=1, success_message="Your feature table is one-hot encoded correctly.")
def test_features(X):
    tips = pd.read_csv("tips.csv")
    category_orders = {
        "sex": ["Male", "Female"],
        "smoker": ["Yes", "No"],
        "day": ["Thur", "Fri", "Sat", "Sun"],
        "time": ["Lunch", "Dinner"],
    }
    for column, categories in category_orders.items():
        tips[column] = pd.Categorical(tips[column], categories=categories)
    columns = ["total_bill", "size", "sex", "smoker", "day", "time"]
    expected = pd.get_dummies(tips[columns], drop_first=True, dtype=int)
    assert isinstance(X, pd.DataFrame), "X should be a pandas DataFrame."
    pd.testing.assert_frame_equal(X, expected, obj="X")
