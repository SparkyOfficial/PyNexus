"""
CLI module for PyNexus.

цей модуль надає функціональність командного рядка.
автор: Андрій Будильников
"""

import argparse
import sys


def main():
    """
    Main entry point for the PyNexus CLI.
    """
    parser = argparse.ArgumentParser(
        description="PyNexus - A universal scientific analytics library"
    )
    parser.add_argument(
        "--version", 
        action="version", 
        version="PyNexus 0.2.0"
    )
    
    # Add more CLI arguments here as needed
    parser.add_argument(
        "--example", 
        help="Run an example script",
        action="store_true"
    )
    
    parser.add_argument(
        "--demo", 
        help="Run a comprehensive demo",
        action="store_true"
    )
    
    args = parser.parse_args()
    
    if args.example:
        print("Running PyNexus example...")
        run_example()
    elif args.demo:
        print("Running PyNexus comprehensive demo...")
        run_demo()
    else:
        parser.print_help()


def run_example():
    """
    Run a simple example to demonstrate PyNexus functionality.
    """
    print("PyNexus Example:")
    print(">>> import pynexus as nx")
    
    # This would normally actually run the example, but we'll just show
    # what it would look like
    print(">>> arr = nx.array([1, 2, 3, 4, 5])")
    print(">>> df = nx.table({'name': ['Alice', 'Bob'], 'age': [25, 30]})")
    print(">>> print(nx.describe(df))")
    print("# This would show descriptive statistics")


def run_demo():
    """
    Run a comprehensive demo showing various PyNexus features.
    """
    print("PyNexus Comprehensive Demo")
    print("=" * 30)
    
    print("\n1. Creating arrays and tables:")
    print("   >>> import pynexus as nx")
    print("   >>> arr = nx.array([1, 2, 3, 4, 5])")
    print("   >>> df = nx.table({'name': ['Alice', 'Bob', 'Charlie'], 'age': [25, 30, 35]})")
    
    print("\n2. Data analysis:")
    print("   >>> stats = nx.describe(df)")
    print("   >>> filtered = nx.filter(df, 'age > 25')")
    print("   >>> grouped = nx.groupby_stats(df, 'age', 'name')")
    
    print("\n3. Visualization:")
    print("   >>> nx.plot_auto(df)")
    print("   >>> nx.plot(df, 'name', 'age', kind='bar')")
    
    print("\n4. Symbolic mathematics:")
    print("   >>> from sympy import symbols")
    print("   >>> x = symbols('x')")
    print("   >>> solutions = nx.solve(x**2 - 4, x)")
    print("   >>> derivative = nx.differentiate(x**3 + 2*x**2 + x, x)")
    print("   >>> integral = nx.integrate(x**2, x)")
    
    print("\nFor more information, please refer to the documentation.")


if __name__ == "__main__":
    main()