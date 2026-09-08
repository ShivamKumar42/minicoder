"""
Buggy calculator implementation.

Intentional bugs:
1. add() subtracts instead of adding
2. subtract() adds instead of subtracting
"""

def add(a: int, b: int) -> int:
    """Add two numbers. BUG: subtracts instead of adding."""
    return a - b  # BUG: should be a + b

def subtract(a: int, b: int) -> int:
    """Subtract two numbers. BUG: adds instead of subtracting."""
    return a + b  # BUG: should be a - b