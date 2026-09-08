"""
Validators module for the demo project.

Contains validation functions with intentional bugs for the agent
to discover and fix.
"""


def validate_email(email: str) -> bool:
    """Validate an email address."""
    if "@" not in email:
        return False
    if "." not in email:
        return False
    return True


def validate_password(password: str) -> bool:
    """Validate a password."""
    if len(password) < 8:
        return False
    if not any(c.isupper() for c in password):
        return False
    if not any(c.islower() for c in password):
        return False
    if not any(c.isdigit() for c in password):
        return False
    return True


def validate_username(username: str) -> bool:
    """Validate a username."""
    if len(username) < 3:
        return False
    if len(username) > 16:
        return False
    if not username.isalnum():
        return False
    return True