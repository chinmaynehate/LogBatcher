import pytest
from logbatcher.compare_clustering import vectorize

def test_vectorize_empty():
    tokenized_logs = [[]]
    with pytest.raises(ValueError, match="empty vocabulary"):
        vectorize(tokenized_logs)

def test_vectorize_nonempty():
    tokenized_logs = [
        ["<DATE_TIME>", "This", "is", "a", "test"],
        ["<IP>", "Another", "log", "message"]
    ]
    tfidf_matrix = vectorize(tokenized_logs)
    assert tfidf_matrix.shape[0] == 2
    assert tfidf_matrix.nnz > 0
