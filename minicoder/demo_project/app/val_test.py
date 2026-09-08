"""
Buggy validators implementation.

Intentional bugs:
1. validate_email() doesn't check proper format
2. validate_password() has wrong length check
3. validate_username() allows special characters
"""

def validate_email(email: str) -> bool:
    """Validate email. BUG: accepts invalid emails."""
    if "@" not in email:
        return False
    # BUG: missing dot check
    return True

def validate_password(password: str) -> bool:
    """Validate password. BUG: accepts short passwords."""
    if len(password) < 3:  # BUG: should be < 8
        return False
    # BUG: missing complexity checks
    return True

def validate_username(username: str) -> bool:
    """Validate username. BUG: allows special characters."""
    if len(username) < 3 or len(username) > 16:
        return False
    # BUG: allows special characters, should be isalnum only
    return True