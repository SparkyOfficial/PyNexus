"""
Tests for the analysis module of PyNexus.
"""

import pytest
import pandas as pd
import pynexus as nx


def test_describe():
    """Test the describe function."""
    df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    desc = nx.describe(df)
    assert isinstance(desc, pd.DataFrame)
    # Check that describe produces results
    assert len(desc) > 0


def test_describe_extended():
    """Test the extended describe function."""
    df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    result = nx.describe(df, extended=True)
    # When extended=True, should return tuple
    assert isinstance(result, tuple)
    assert len(result) == 2
    desc, extended = result
    assert isinstance(desc, pd.DataFrame)
    # extended should be a dict
    assert isinstance(extended, dict)


def test_filter():
    """Test the filter function."""
    df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    filtered = nx.filter(df, 'a > 1')
    assert isinstance(filtered, pd.DataFrame)
    assert len(filtered) == 2
    assert list(filtered['a']) == [2, 3]


def test_groupby_stats():
    """Test the groupby_stats function."""
    df = pd.DataFrame({'category': ['a', 'b', 'a', 'b'], 'value': [1, 2, 3, 4]})
    stats = nx.groupby_stats(df, 'category', 'value', 'mean')
    assert isinstance(stats, pd.Series)
    assert len(stats) == 2


def test_correlation_matrix():
    """Test the correlation_matrix function."""
    df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    corr = nx.correlation_matrix(df)
    assert isinstance(corr, pd.DataFrame)
    assert corr.shape == (2, 2)