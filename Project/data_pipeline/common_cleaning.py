"""Repairs shared by every cleaning run, the live files, the tools and the profiles.

Every function returns a repaired copy and leaves what it was given untouched.
Nothing here reads or writes a file, and no column is ever removed. A value in
a notation a function does not recognise is refused with a ValueError naming
the column and the values.
"""

import html
import re
from decimal import ROUND_HALF_UP, Decimal

import numpy as np
import pandas as pd

from config import config


# --------------------------------------------------------------------------- helpers
def _refuse(column, values):
    shown = ", ".join(repr(value) for value in sorted({str(value) for value in values}))
    raise ValueError(f"Column {column} holds values that cannot be read: {shown}")


def _round_half_up(number, step):
    return number.quantize(step, rounding=ROUND_HALF_UP)


# --------------------------------------------------------------------------- rows
def remove_duplicate_records(dataframe):
    """Drop rows the extract wrote twice, keeping the first copy."""
    return dataframe.drop_duplicates(keep="first").reset_index(drop=True)


def refuse_missing_values(dataframe, required_columns):
    """Refuse a table in which a required key is blank."""
    for column in required_columns:
        blank_rows = int(dataframe[column].isna().sum())
        if blank_rows:
            raise ValueError(f"Column {column} is blank on {blank_rows} rows")
    return dataframe


def refuse_repeated_references(dataframe, reference_column):
    """Refuse a table in which two rows claim the same identity."""
    values = dataframe[reference_column]
    repeated = values[values.notna() & values.duplicated(keep=False)]
    if not repeated.empty:
        shown = ", ".join(sorted({str(value) for value in repeated}))
        raise ValueError(f"Column {reference_column} repeats the values: {shown}")
    return dataframe


def remove_rows_missing_target(dataframe, target_column):
    """Drop the rows a model cannot learn from."""
    return dataframe[dataframe[target_column].notna()].reset_index(drop=True)


# --------------------------------------------------------------------------- categories
def normalize_category_values(dataframe, value_mappings):
    """Fold every spelling of a value onto its canonical value."""
    frame = dataframe.copy()
    for column, mapping in value_mappings.items():
        if column not in frame.columns:
            continue
        written = frame[column]
        given = written.notna()
        mapped = written.where(~given, written.astype(str).str.lower().map(mapping))
        unknown = given & mapped.isna()
        if unknown.any():
            _refuse(column, written[unknown])
        frame[column] = mapped.astype(object).where(given, np.nan)
    return frame


def convert_application_references(references):
    """Write every application reference as LCT and seven digits."""
    def convert(value):
        if pd.isna(value):
            return np.nan
        match = re.match(config.APPLICATION_REFERENCE_PATTERN, str(value))
        if not match:
            return None
        number = int(match.group(1))
        return f"{config.APPLICATION_REFERENCE_PREFIX}{number:0{config.APPLICATION_REFERENCE_DIGITS}d}"

    converted = references.map(convert)
    unreadable = references.notna() & converted.isna()
    if unreadable.any():
        _refuse(references.name, references[unreadable])
    return converted.astype(object).where(references.notna(), np.nan).rename(references.name)


def refuse_unreadable_identifiers(dataframe, identifier_patterns):
    """Refuse an installer, premises or substation reference in a form no portal uses."""
    for column, pattern in identifier_patterns.items():
        if column not in dataframe.columns:
            continue
        values = dataframe[column]
        given = values.notna()
        matches = values[given].astype(str).map(lambda value: bool(re.fullmatch(pattern, value)))
        if not matches.all():
            _refuse(column, values[given][~matches])
    return dataframe


# --------------------------------------------------------------------------- whole numbers
def _read_whole_numbers(values, pattern, column):
    def read(value):
        if pd.isna(value):
            return pd.NA
        match = re.match(pattern, str(value))
        if not match:
            return None
        return int(match.group(1))

    read_values = values.map(read)
    unreadable = values.notna() & read_values.isna()
    if unreadable.any():
        _refuse(column, values[unreadable])
    return pd.array(read_values.where(values.notna(), pd.NA).tolist(),
                    dtype=config.WHOLE_NUMBER_DTYPE)


def convert_whole_numbers(dataframe, number_patterns):
    """Read counts, cut-out ratings and working days as whole numbers."""
    frame = dataframe.copy()
    for column, pattern in number_patterns.items():
        if column in frame.columns:
            frame[column] = _read_whole_numbers(frame[column], pattern, column)
    return frame


def convert_phases(dataframe):
    """Read the supply's and the equipment's phases as the numbers 1 or 3."""
    frame = dataframe.copy()
    for column in config.PHASE_COLUMNS:
        if column not in frame.columns:
            continue
        written = frame[column]
        given = written.notna()
        mapped = written[given].astype(str).str.lower().map(config.PHASE_MAPPING)
        if mapped.isna().any():
            _refuse(column, written[given][mapped.isna()])
        values = [pd.NA] * len(frame)
        for position, value in zip(np.flatnonzero(given.to_numpy()), mapped.tolist()):
            values[position] = int(value)
        frame[column] = pd.array(values, dtype=config.WHOLE_NUMBER_DTYPE)
    return frame


def treat_impossible_phase_pairs_as_missing(dataframe):
    """Blank the equipment's phases wherever they exceed the supply's."""
    frame = dataframe.copy()
    supply = pd.to_numeric(frame[config.SUPPLY_PHASES_COLUMN], errors="coerce")
    equipment = pd.to_numeric(frame[config.EQUIPMENT_PHASES_COLUMN], errors="coerce")
    impossible = supply.notna() & equipment.notna() & (equipment > supply)
    column = frame[config.EQUIPMENT_PHASES_COLUMN].copy()
    column[impossible.to_numpy()] = pd.NA
    frame[config.EQUIPMENT_PHASES_COLUMN] = column
    return frame


# --------------------------------------------------------------------------- currents and amounts
def convert_power_to_amps_per_phase(kilowatts, phases):
    """Turn a power in kilowatts into the amps it draws on each phase, exactly."""
    watts = Decimal(kilowatts) * config.WATTS_PER_KILOWATT
    amps = watts / config.DECLARED_VOLTS / Decimal(int(phases))
    return _round_half_up(amps, config.AMPS_STEP)


def _strip_separator(text):
    return text.replace(config.THOUSANDS_SEPARATOR, "")


def read_current_value(value, phases):
    """Read one written current as amps per phase; None when it cannot be converted.

    Raises LookupError when the value is written in no known notation.
    """
    text = str(value)
    for notation, pattern in config.CURRENT_NOTATIONS.items():
        match = re.match(pattern, text)
        if not match:
            continue
        number = Decimal(_strip_separator(match.group(1)))
        if notation == config.AMPS_UNIT:
            return _round_half_up(number, config.AMPS_STEP)
        if phases is None or pd.isna(phases):
            return None
        kilowatts = number * config.POWER_UNITS[notation]
        return convert_power_to_amps_per_phase(kilowatts, int(phases))
    raise LookupError(text)


def current_notation(value):
    """Name the notation a current is written in, or None."""
    for notation, pattern in config.CURRENT_NOTATIONS.items():
        if re.match(pattern, str(value)):
            return notation
    return None


def convert_currents(dataframe, current_phases):
    """Read every current, in amps or written as a power, as amps per phase."""
    frame = dataframe.copy()
    for column, phases_column in current_phases.items():
        if column not in frame.columns:
            continue
        written = frame[column]
        phases = frame[phases_column]
        converted, unreadable = [], []
        for value, row_phases in zip(written.tolist(), phases.tolist()):
            if pd.isna(value):
                converted.append(np.nan)
                continue
            try:
                amps = read_current_value(value, row_phases)
            except LookupError:
                unreadable.append(value)
                converted.append(np.nan)
                continue
            converted.append(np.nan if amps is None else float(amps))
        if unreadable:
            _refuse(column, unreadable)
        frame[column] = pd.Series(converted, index=frame.index, dtype=float)
    return frame


def read_amount_value(value, reading):
    """Read one written amount with its reading; raises LookupError when unreadable."""
    text = str(value)
    for notation, pattern in reading["notations"].items():
        match = re.match(pattern, text)
        if match:
            number = Decimal(_strip_separator(match.group(1))) * reading["units"][notation]
            return _round_half_up(number, reading["step"])
    raise LookupError(text)


def amount_notation(value, reading):
    """Name the notation an amount is written in, or None."""
    for notation, pattern in reading["notations"].items():
        if re.match(pattern, str(value)):
            return notation
    return None


def convert_amounts(dataframe, amount_readings):
    """Read powers, power factors and resistances in every notation a portal writes."""
    frame = dataframe.copy()
    for column, reading in amount_readings.items():
        if column not in frame.columns:
            continue
        converted, unreadable = [], []
        for value in frame[column].tolist():
            if pd.isna(value):
                converted.append(np.nan)
                continue
            try:
                converted.append(float(read_amount_value(value, reading)))
            except LookupError:
                unreadable.append(value)
                converted.append(np.nan)
        if unreadable:
            _refuse(column, unreadable)
        frame[column] = pd.Series(converted, index=frame.index, dtype=float)
    return frame


# --------------------------------------------------------------------------- impossible readings
def _blank_where(frame, column, condition):
    values = frame[column].copy()
    values[condition.to_numpy()] = pd.NA if str(values.dtype) == config.WHOLE_NUMBER_DTYPE else np.nan
    frame[column] = values


def treat_values_below_minimum_as_missing(dataframe, minimum_values):
    """Blank the readings too low to be true."""
    frame = dataframe.copy()
    for column, minimum in minimum_values.items():
        if column in frame.columns:
            numbers = pd.to_numeric(frame[column], errors="coerce")
            _blank_where(frame, column, numbers.notna() & (numbers < minimum))
    return frame


def treat_values_above_maximum_as_missing(dataframe, maximum_values):
    """Blank the readings too high to be true."""
    frame = dataframe.copy()
    for column, maximum in maximum_values.items():
        if column in frame.columns:
            numbers = pd.to_numeric(frame[column], errors="coerce")
            _blank_where(frame, column, numbers.notna() & (numbers > maximum))
    return frame


def treat_values_outside_sizes_as_missing(dataframe, allowed_sizes):
    """Blank the readings that are none of the sizes that exist."""
    frame = dataframe.copy()
    for column, sizes in allowed_sizes.items():
        if column in frame.columns:
            numbers = pd.to_numeric(frame[column], errors="coerce")
            _blank_where(frame, column, numbers.notna() & ~numbers.isin(sizes))
    return frame


# --------------------------------------------------------------------------- revisions
def keep_latest_revision_of_each_application(dataframe):
    """Keep each application once, as its highest revision."""
    reference = config.APPLICATION_REFERENCE_COLUMN
    revision = config.REVISION_NUMBER_COLUMN
    blank = dataframe[reference].isna() | dataframe[revision].isna()
    if blank.any():
        named = sorted({str(value) for value in dataframe.loc[blank, reference]})
        raise ValueError(f"Applications {', '.join(named)} leave {reference} or {revision} blank")
    pairs = dataframe[[reference, revision]].astype({revision: "int64"})
    doubled = pairs.duplicated(keep=False)
    if doubled.any():
        named = sorted(set(pairs.loc[doubled, reference]))
        raise ValueError(f"Applications {', '.join(named)} give the same {revision} twice")
    highest = pairs.groupby(reference)[revision].transform("max")
    keep = (pairs[revision] == highest).to_numpy()
    return dataframe[keep].reset_index(drop=True)


# --------------------------------------------------------------------------- free text
def convert_markup_to_plain_text(texts):
    """Recover the words a writer meant from text carrying markup and escapes."""
    def convert(text):
        if pd.isna(text):
            return np.nan
        plain = re.sub(config.MARKUP_PATTERN, " ", str(text))
        plain = html.unescape(plain)
        plain = re.sub(config.WHITESPACE_RUN_PATTERN, " ", plain)
        return plain.strip()

    return texts.map(convert).astype(object).rename(texts.name)


def clean_statement_text(texts):
    """Plain words, and text written wholly in capitals put in small letters."""
    plain = convert_markup_to_plain_text(texts)

    def soften(text):
        if pd.isna(text):
            return np.nan
        return text.lower() if text.isupper() else text

    return plain.map(soften).astype(object).rename(texts.name)
