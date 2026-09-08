"""
README for the demo project.

This demo project is designed for MiniCoder to demonstrate
autonomous code repair capabilities.
"""

PROJECT_SUMMARY = """
MiniCoder Demo Project

This project demonstrates the MiniCoder agent's ability to:
1. Inspect a repository and understand its structure
2. Search the codebase for relevant code and tests
3. Read files and understand existing conventions
4. Identify and fix intentional defects
5. Run tests and verify fixes
6. Iteratively patch code based on test failures
7. Confirm all tests pass before finishing
"""


def main():
    print(PROJECT_SUMMARY)


if __name__ == "__main__":
    main()