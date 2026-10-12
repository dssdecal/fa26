from otter.test_files import test_case
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

OK_FORMAT = False
name = "q11"


@test_case(points=1, success_message="Your depth comparison table is correct.")
def test_depth_results(depth_results, X_train, X_test, g_train, g_test):
    rows = []
    for depth in [1, 2, 3, 5, 8, 12]:
        tree = DecisionTreeClassifier(max_depth=depth, random_state=42).fit(X_train, g_train)
        rows.append({
            "max_depth": depth,
            "train_acc": tree.score(X_train, g_train),
            "test_acc": tree.score(X_test, g_test),
        })
    expected = pd.DataFrame(rows)
    pd.testing.assert_frame_equal(
        depth_results[["max_depth", "train_acc", "test_acc"]].reset_index(drop=True),
        expected,
    )
