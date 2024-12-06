import pytest
from logbatcher.compare_clustering import tokenize

def test_tokenize_basic():
    log_line = "2024-01-01 INFO This is a test log message 10.0.0.1 error=404"
    tokens = tokenize(log_line)
    assert '<DATE_TIME>' in tokens
    assert '<LEVEL>' in tokens
    assert '<IP>' in tokens
    assert isinstance(tokens, list)
    assert len(tokens) > 0
