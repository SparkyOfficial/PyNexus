"""
solver module for PyNexus.

цей модуль надає можливості символьних обчислень.
автор: Андрій Будильников
"""

import sympy as sp


def solve(expr, *args, **kwargs):
    """
    solve symbolic equations.
    
    args:
        expr: symbolic expression to solve
        *args: additional arguments to pass to sp.solve
        **kwargs: additional keyword arguments to pass to sp.solve
        
    returns:
        list: solutions to the equation
        
    example:
        >>> from sympy import symbols
        >>> x = symbols('x')
        >>> solutions = solve(x**2 - 4, x)
        >>> print(solutions)
        [-2, 2]
    """
    return sp.solve(expr, *args, **kwargs)


def differentiate(expr, var=None):
    """
    differentiate a symbolic expression.
    
    args:
        expr: symbolic expression to differentiate
        var: variable to differentiate with respect to (optional)
        
    returns:
        sympy expression: derivative of the expression
        
    example:
        >>> from sympy import symbols
        >>> x = symbols('x')
        >>> expr = x**3 + 2*x**2 + x
        >>> derivative = differentiate(expr, x)
        >>> print(derivative)
        3*x**2 + 4*x + 1
    """
    if var is None:
        return sp.diff(expr)
    else:
        return sp.diff(expr, var)


def integrate(expr, var=None, limits=None):
    """
    integrate a symbolic expression.
    
    args:
        expr: symbolic expression to integrate
        var: variable to integrate with respect to (optional)
        limits: tuple of (lower, upper) limits for definite integral (optional)
        
    returns:
        sympy expression: integral of the expression
        
    example:
        >>> from sympy import symbols
        >>> x = symbols('x')
        >>> expr = x**2
        >>> integral = integrate(expr, x)
        >>> print(integral)
        x**3/3
    """
    if limits is not None:
        # definite integral
        if var is None:
            return sp.integrate(expr, limits)
        else:
            return sp.integrate(expr, (var, limits[0], limits[1]))
    else:
        # indefinite integral
        if var is None:
            return sp.integrate(expr)
        else:
            return sp.integrate(expr, var)


def limit(expr, var, point):
    """
    calculate the limit of a symbolic expression.
    
    args:
        expr: symbolic expression
        var: variable to take the limit of
        point: point to approach
        
    returns:
        sympy expression: limit of the expression
        
    example:
        >>> from sympy import symbols
        >>> x = symbols('x')
        >>> expr = sp.sin(x) / x
        >>> lim = limit(expr, x, 0)
        >>> print(lim)
        1
    """
    return sp.limit(expr, var, point)


def series(expr, var, point=0, n=6):
    """
    compute taylor series expansion of a symbolic expression.
    
    args:
        expr: symbolic expression
        var: variable to expand around
        point: point to expand around (default: 0)
        n: order of expansion (default: 6)
        
    returns:
        sympy expression: taylor series expansion
        
    example:
        >>> from sympy import symbols, exp
        >>> x = symbols('x')
        >>> expr = exp(x)
        >>> expansion = series(expr, x, 0, 5)
        >>> print(expansion)
        1 + x + x**2/2 + x**3/6 + x**4/24 + O(x**5)
    """
    return sp.series(expr, var, point, n)