"""The statement sentences: the table the sentence model learns from."""

import pandas as pd

from config import config
from data_pipeline.document_cleaning import clean_document_records


def clean_sentence_dataset(frame):
    """Labelled sentences of at least the minimum number of words, ready to learn from."""
    sentences = clean_document_records(frame, config.SENTENCE_VALUE_MAPPINGS,
                                       config.SENTENCE_WHOLE_NUMBER_PATTERNS,
                                       config.SENTENCE_ID_COLUMN, config.SENTENCE_TEXT_COLUMN)
    text = sentences[config.SENTENCE_TEXT_COLUMN]
    lengths = [pd.NA if pd.isna(value) else len(value) for value in text]
    sentences[config.SENTENCE_LENGTH_COLUMN] = pd.array(lengths, dtype=config.WHOLE_NUMBER_DTYPE)
    word_counts = text.map(lambda value: 0 if pd.isna(value) else len(str(value).split()))
    keep = sentences[config.SENTENCE_TARGET_COLUMN].notna() & (
        word_counts >= config.MINIMUM_SENTENCE_WORDS)
    return sentences[keep.to_numpy()].reset_index(drop=True)
