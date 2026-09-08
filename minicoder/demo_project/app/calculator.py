"""
Demo project for MiniCoder.

This project contains intentional defects that the coding agent
must discover and fix through inspection, searching, and iterative
patch application.
"""

from calculator import add, subtract


def test_calculate():
    """Test the calculator functions."""
    # These should work correctly
    assert add(2, 3) == 5, f"add(2, 3) should be 5, got {add(2, 3)}"
    assert subtract(5, 3) == 2, f"subtract(5, 3) should be 2, got {subtract(5, 3)}"
    
    # These have intentional bugs
    assert add(-1, -1) == -2, f"add(-1, -1) should be -2, got {add(-1, -1)}"
    assert subtract(-1, -1) == 0, f"subtract(-1, -1) should be 0, got {subtract(-1, -1)}"
    assert add(0, 0) == 0, f"add(0, 0) should be 0, got {add(0, 0)}"
    assert subtract(0, 0) == 0, f"subtract(0, 0) should be 0, got {subtract(0, 0)}"


def test_edge_cases():
    """Test edge cases."""
    # These have intentional bugs
    assert add(10, -5) == 5, f"add(10, -5) should be 5, got {add(10, -5)}"
    assert subtract(10, -5) == 15, f"subtract(10, -5) should be 15, got {subtract(10, -5)}"
    assert add(-5, 10) == 5, f"add(-5, 10) should be 5, got {add(-5, 10)}"
    assert subtract(-5, 10) == -15, f"subtract(-5, 10) should be -15, got {subtract(-5, 10)}"