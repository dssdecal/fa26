from otter.test_files import test_case
import numpy as np
from sklearn.tree import DecisionTreeClassifier

OK_FORMAT = False
name = "q10"


@test_case(points=1, success_message="Your depth-3 decision tree is correct.")
def test_tree(tree_model, tree_acc, X_train, X_test, g_train, g_test):
    assert isinstance(tree_model, DecisionTreeClassifier), "tree_model should be a DecisionTreeClassifier."
    assert tree_model.max_depth == 3, "Set max_depth=3."
    assert tree_model.random_state == 42, "Set random_state=42."
    expected = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X_train, g_train)
    assert np.isclose(tree_acc, expected.score(X_test, g_test)), "Check tree_acc."
