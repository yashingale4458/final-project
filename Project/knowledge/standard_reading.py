"""Reads a technology's requirements out of that technology's own connection standard.

This module keeps the requirements out of the code: every figure comes from
the standard's own PDF, read at run time.
"""

import re
from decimal import Decimal

from config import config
from knowledge.chunk_splitting import split_documents_into_sections

_TEXT_PARTS = {config.REQUIREMENT_NAME_PART, config.REQUIREMENT_SENTENCE_PART}


def read_connection_standard(technology, document_entries):
    """Establish every requirement of one technology from its own standard."""
    if technology not in config.CONNECTION_STANDARD_DOCUMENTS:
        raise ValueError(config.NO_STANDARD_CONFIGURED_MESSAGE.format(technology=technology))
    document_name = config.CONNECTION_STANDARD_DOCUMENTS[technology]
    held = [entry for entry in document_entries if entry["document_name"] == document_name]
    if len(held) != 1:
        raise ValueError(f"{document_name} must be handed over exactly once, not {len(held)} times")

    sections = {section["section_number"]: section["section_text"]
                for section in split_documents_into_sections(held)}
    technology_section = sections.get(config.TECHNOLOGY_SECTION_NUMBER)
    requirements_section = sections.get(config.REQUIREMENTS_SECTION_NUMBER)
    if technology_section is None or requirements_section is None:
        raise ValueError(f"{document_name} lacks its technology or requirements section")

    codes = re.findall(config.TECHNOLOGY_CODE_PATTERN, technology_section)
    if codes != [technology]:
        raise ValueError(f"{document_name} does not state the code {technology} exactly once "
                         f"and no other code")

    listing = re.fullmatch(config.REQUIREMENTS_SECTION_PATTERN, requirements_section)
    if listing is None:
        raise ValueError(f"{document_name} does not list its requirements in the fixed form")
    parts = re.split(config.REQUIREMENT_NUMBER_PATTERN, listing.group(1))
    if parts[0].strip():
        raise ValueError(f"{document_name} has text before its first requirement")
    numbers = [int(number) for number in parts[1::2]]
    texts = [text.strip() for text in parts[2::2]]
    if not numbers or numbers != list(range(1, len(numbers) + 1)):
        raise ValueError(f"{document_name} does not number its requirements from 1 in order")

    requirements, kinds_seen = [], set()
    for text in texts:
        fitting = [(kind, re.fullmatch(pattern, text))
                   for kind, pattern in config.REQUIREMENT_SENTENCE_PATTERNS.items()]
        fitting = [(kind, match) for kind, match in fitting if match is not None]
        if len(fitting) != 1:
            raise ValueError(f"{document_name} states a requirement no single form fits: {text}")
        kind, match = fitting[0]
        if kind in kinds_seen:
            raise ValueError(f"{document_name} states the requirement kind {kind} twice")
        kinds_seen.add(kind)
        rules = {part: Decimal(value) for part, value in match.groupdict().items()
                 if part not in _TEXT_PARTS and value is not None}
        requirements.append({
            "requirement_kind": kind,
            "requirement_name": match.group(config.REQUIREMENT_NAME_PART),
            "requirement_sentence": match.group(config.REQUIREMENT_SENTENCE_PART),
            "rules": rules,
        })
    return {"technology": technology, "document_name": document_name,
            "requirements": requirements}
