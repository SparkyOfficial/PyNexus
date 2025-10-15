"""
Tests for the core module of PyNexus.
"""

import pytest
import numpy as np
import pandas as pd
import pynexus as nx


def test_array():
    """Test the array function."""
    # Test with list input
    arr = nx.array([1, 2, 3])
    assert isinstance(arr, np.ndarray)
    assert len(arr) == 3
    
    # Test with nested list
    arr2d = nx.array([[1, 2], [3, 4]])
    assert isinstance(arr2d, np.ndarray)
    assert arr2d.shape == (2, 2)


def test_table():
    """Test the table function."""
    # Test with dict input
    df = nx.table({'a': [1, 2], 'b': [3, 4]})
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert list(df.columns) == ['a', 'b']
    
    # Test with no input
    df_empty = nx.table()
    assert isinstance(df_empty, pd.DataFrame)
    assert len(df_empty) == 0


def test_matrix():
    """Test the matrix function."""
    # Test with nested list
    mat = nx.matrix([[1, 2], [3, 4]])
    assert isinstance(mat, np.matrix)
    assert mat.shape == (2, 2)


def test_series():
    """Test the series function."""
    # Test with list input
    s = nx.series([1, 2, 3, 4])
    assert isinstance(s, pd.Series)
    assert len(s) == 4