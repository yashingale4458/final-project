"""Reading and writing the CSV files of the data preparation stage.

This is the only data preparation module that touches a CSV file on disk.
Raw files are read as text, guessing nothing; cleaned files are read with
each column's type taken from its contents.
"""

import os

import pandas as pd

from config import config

_NULL_TOKENS = {token.lower() for token in config.NULL_TOKENS}


def load_raw_dataset(file_path):
    """Read one raw extract as text, every null token as a missing value."""
    frame = pd.read_csv(file_path, dtype=str, keep_default_na=False, na_filter=False,
                        skipinitialspace=True, encoding="utf-8")
    frame.columns = frame.columns.str.strip()
    frame = frame.apply(lambda column: column.str.strip())
    is_missing = frame.apply(lambda column: column.str.lower().isin(_NULL_TOKENS))
    return frame.mask(is_missing)


def save_processed_dataset(dataframe, file_path):
    """Save a cleaned table, creating its folder, with no index column."""
    folder = os.path.dirname(file_path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    dataframe.to_csv(file_path, index=False)


def load_processed_dataset(file_path):
    """Read a cleaned table; only an empty cell is missing."""
    return pd.read_csv(file_path, keep_default_na=False, na_values=[""], encoding="utf-8")
