"""
Generate values for the admin dashboard secrets.

Usage (from backend/):
    python -m app.scripts.hash_admin_password            # prompts twice
    python -m app.scripts.hash_admin_password --stdin    # reads one line from stdin
    python -m app.scripts.hash_admin_password --token-secret   # also print a random ADMIN_TOKEN_SECRET

Paste the printed ADMIN_PASSWORD_HASH (and ADMIN_TOKEN_SECRET) into the
Space secrets. The plain password is never printed or stored.
"""

import argparse
import getpass
import secrets
import sys

from app.services.admin_auth import hash_password, verify_password

MIN_LENGTH = 12


def main() -> int:
    parser = argparse.ArgumentParser(description="Hash an admin password with scrypt.")
    parser.add_argument("--stdin", action="store_true", help="read the password from stdin")
    parser.add_argument("--token-secret", action="store_true", help="also print a random ADMIN_TOKEN_SECRET")
    args = parser.parse_args()

    if args.stdin:
        password = sys.stdin.readline().rstrip("\r\n")
    else:
        password = getpass.getpass("Admin password: ")
        if getpass.getpass("Repeat password: ") != password:
            print("Passwords do not match.", file=sys.stderr)
            return 1

    if len(password) < MIN_LENGTH:
        print(f"Password must be at least {MIN_LENGTH} characters.", file=sys.stderr)
        return 1

    encoded = hash_password(password)
    assert verify_password(password, encoded)
    print(f"ADMIN_PASSWORD_HASH={encoded}")
    if args.token_secret:
        print(f"ADMIN_TOKEN_SECRET={secrets.token_urlsafe(48)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
