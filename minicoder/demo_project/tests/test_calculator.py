"""
Tests for the calculator module.

This test file contains tests that expose bugs in calculator.py.
The agent must find and fix the bugs in calculator.py.
"""

from calc_test import add, subtract


def test_basic_operations():
    """Test basic calculator operations."""
    assert add(2, 3) == 5
    assert subtract(5, 3) == 2


def test_negative_numbers():
    """Test negative number operations."""
    assert add(-1, -1) == -2
    assert subtract(-1, -1) == 0


def test_zero_cases():
    """Test zero cases."""
    assert add(0, 0) == 0
    assert subtract(0, 0) == 0


def test_mixed_signs():
    """Test mixed sign operations."""
    assert add(10, -5) == 5
    assert subtract(10, -5) == 15
    assert add(-5, 10) == 5
    assert subtract(-5, 10) == -15