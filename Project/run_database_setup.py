"""Entry point 1: creates the database, its six tables and the fourteen sign-in accounts.
Safe to run again: nothing is duplicated.

    python run_database_setup.py
"""

import os
import secrets

from sqlalchemy import inspect

from access.password_security import hash_password
from config import config
from persistence.database import database_engine, session_factory
from persistence.models import Base, UserAccount
from persistence.repository import fetch_user_account, insert_user_account


def _refuse_earlier_versions():
    inspector = inspect(database_engine)
    for table in Base.metadata.sorted_tables:
        if not inspector.has_table(table.name):
            continue
        existing = {column["name"]: bool(column["nullable"]) for column in inspector.get_columns(table.name)}
        expected = {column.name: bool(column.nullable) for column in table.columns}
        if existing != expected:
            raise SystemExit(f"The table {table.name} in {config.APPLICATION_DATABASE_FILE} is from an "
                             f"earlier version: delete {config.APPLICATION_DATABASE_FILE} and "
                             f"{config.CHECKPOINT_DATABASE_FILE}, then run this setup again")


def main():
    """Give the system a database to write to, and the staff their accounts."""
    os.makedirs(config.DATABASE_DIRECTORY, exist_ok=True)
    _refuse_earlier_versions()
    Base.metadata.create_all(database_engine)
    print(f"database: {config.APPLICATION_DATABASE_FILE}")
    print(f"tables: {', '.join(sorted(table.name for table in Base.metadata.sorted_tables))}")
    created = present = 0
    with session_factory() as session:
        for account in config.SEED_USER_ACCOUNTS:
            if fetch_user_account(session, UserAccount.username == account["username"]):
                present += 1
                continue
            salt = secrets.token_hex(config.PASSWORD_SALT_BYTES)
            insert_user_account(session, {
                "username": account["username"],
                "display_name": account["display_name"],
                "role": account["role"],
                "password_hash": hash_password(account["password"], salt),
                "password_salt": salt,
                "session_token": None,
            })
            created += 1
            print(f"  account created  {account['username']:<{config.USERNAME_WIDTH}} "
                  f"{account['display_name']:<{config.DISPLAY_NAME_WIDTH}} {account['role']}")
        session.commit()
    print(f"accounts created: {created}, already present: {present}")


if __name__ == "__main__":
    main()
