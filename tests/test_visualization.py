"""
Tests for the visualization module of PyNexus.
"""

import pytest
import pandas as pd
import pynexus as nx


def test_plot():
    """Test the plot function."""
    # Create test data
    df = pd.DataFrame({'x': [1, 2, 3], 'y': [1, 4, 2]})
    
    # Test basic plotting (this won't actually display a plot in tests)
    # We're mainly checking that the function runs without error
    ax = nx.plot(df, 'x', 'y')
    assert ax is not None
    
    # Test different plot kinds
    ax_bar = nx.plot(df, 'x', 'y', kind='bar')
    assert ax_bar is not None
    
    ax_line = nx.plot(df, 'x', 'y', kind='line')
    assert ax_line is not None


def test_plot_auto():
    """Test the plot_auto function."""
    # Create test data
    df = pd.DataFrame({'x': [1, 2, 3], 'y': [1, 4, 2]})
    
    # Test auto plotting
    ax = nx.plot_auto(df)
    assert ax is not None


def test_subplot_grid():
    """Test the subplot_grid function."""
    # Create test data
    df = pd.DataFrame({'x': [1, 2, 3], 'y': [1, 4, 2]})
    data_list = [df['x'], df['y']]
    
    # Test subplot grid creation
    fig, axes = nx.subplot_grid(data_list)
    assert fig is not None
    assert axes is not None