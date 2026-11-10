from otter.test_files import test_case
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

OK_FORMAT = False
name = 'q2'
points = 1

@test_case(points=None, hidden=False)
def test_retrieval(retrieve, documents, vectorizer, doc_matrix):
    for query, k in [("missing blank ratings", 2), ("privacy publish names identifiers", 3), ("meal credits guest passes", 1)]:
        scores = cosine_similarity(vectorizer.transform([query]), doc_matrix).ravel()
        idx = np.argsort(-scores, kind="stable")[:k]
        expected = documents.iloc[idx].loc[scores[idx] > 0, "chunk_id"].tolist()
        actual = retrieve(query, k)
        assert actual.chunk_id.tolist() == expected, "Return the highest scoring positive matches in order."
        assert np.allclose(actual.score, scores[idx][scores[idx] > 0]), "Keep the matching similarity scores."
    assert retrieve("zzzxxyyqq").empty, "Do not return zero-score passages."
