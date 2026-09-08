"""
Tests for the validators module.

Contains tests that expose bugs in validators.py.
The agent must find and fix the bugs.
"""

from val_test import validate_email, validate_password, validate_username


def test_email_validation():
    """Test email validation."""
    assert validate_email("user@example.com") is True
    assert validate_email("invalid") is False
    assert validate_email("@example.com") is False
    assert validate_email("user@") is False


def test_password_validation():
    """Test password validation."""
    assert validate_password("password123") is True
    assert validate_password("short") is False
    assert validate_password("NOLOWER") is False
    assert validate_password("noupper") is False


def test_username_validation():
    """Test username validation."""
    assert validate_username("user123") is True
    assert validate_username("ab") is False
    assert validate_username("very_long_username_that_exceeds") is False
    assert validate_username("user@name") is False