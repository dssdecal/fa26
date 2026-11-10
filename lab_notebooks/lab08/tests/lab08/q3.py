from otter.test_files import test_case
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

OK_FORMAT = False
name = 'q3'
points = 1

@test_case(points=None, hidden=False)
def test_prompt(build_prompt, documents):
    prompt = build_prompt("How should ratings be handled?", documents.iloc[:2])
    assert isinstance(prompt, str), "Return the prompt as a string."
    assert "How should ratings be handled?" in prompt, "Include the question."
    for row in documents.iloc[:2].itertuples():
        assert row.chunk_id in prompt and row.source in prompt and row.text in prompt, "Include the IDs, source names, and full passage text."
    assert "i do not know from these documents" in prompt.lower(), "Specify the requested abstention phrase."
    # Other prompt requirements are reviewed manually; wording need not be identical.
