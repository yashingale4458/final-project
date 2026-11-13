"""The connection and the session factory, built once when the module is imported.

Importing this module neither creates tables nor writes anything; creating the
tables is the work of run_database_setup.py.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from config import config

database_engine = create_engine(config.APPLICATION_DATABASE_URL,
                                connect_args={"check_same_thread": False})
session_factory = sessionmaker(bind=database_engine, expire_on_commit=False)
