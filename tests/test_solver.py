"""
Tests for the solver module of PyNexus.
"""

import pytest
import sympy as sp
import pynexus as nx


def test_solve():
    """Test the solve function."""
    # Test quadratic equation
    x = sp.Symbol('x')
    solutions = nx.solve(x**2 - 4, x)
    assert isinstance(solutions, list)
    assert len(solutions) == 2
    # Solutions should be -2 and 2
    assert -2 in solutions
    assert 2 in solutions
    
    # Test linear equation
    solutions_linear = nx.solve(2*x - 4, x)
    assert isinstance(solutions_linear, list)
    assert len(solutions_linear) == 1
    assert solutions_linear[0] == 2


def test_differentiate():
    """Test the differentiate function."""
    x = sp.Symbol('x')
    expr = x**3 + 2*x**2 + x
    derivative = nx.differentiate(expr, x)
    assert derivative is not None


def test_integrate():
    """Test the integrate function."""
    x = sp.Symbol('x')
    expr = x**2
    integral = nx.integrate(expr, x)
    assert integral is not None
    
    # Test definite integral
    definite_integral = nx.integrate(expr, x, (0, 1))
    assert definite_integral is not None


def test_limit():
    """Test the limit function."""
    x = sp.Symbol('x')
    expr = sp.sin(x) / x
    lim = nx.limit(expr, x, 0)
    assert lim is not None


def test_series():
    """Test the series function."""
    x = sp.Symbol('x')
    expr = sp.exp(x)
    expansion = nx.series(expr, x, 0, 5)
    assert expansion is not None