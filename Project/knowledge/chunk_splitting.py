"""Cuts the documents into sections and passages for the knowledge base.

A passage never spans two sections, and a sentence is never cut in half.
"""

import re

from config import config


def _single_spaced(text):
    return re.sub(config.WHITESPACE_RUN_PATTERN, " ", text).strip()


def split_documents_into_sections(document_entries):
    """Divide each document at its numbered headings."""
    sections = []
    for entry in document_entries:
        name = entry["document_name"]
        text = entry["document_text"]
        headers = re.findall(config.DOCUMENT_HEADER_PATTERN, text)
        if len(headers) != 1:
            raise ValueError(f"{name} does not carry exactly one document header line")
        headings = list(re.finditer(config.SECTION_HEADING_PATTERN, text))
        if not headings:
            raise ValueError(f"{name} has no numbered sections")
        numbers = [int(match.group(1)) for match in headings]
        if numbers != list(range(1, len(numbers) + 1)):
            raise ValueError(f"{name} does not number its sections from 1 in order")
        header_end = re.search(config.DOCUMENT_HEADER_PATTERN, text).end()
        if headings[0].start() < header_end:
            raise ValueError(f"{name} opens a section before its header line")
        for position, heading in enumerate(headings):
            end = headings[position + 1].start() if position + 1 < len(headings) else len(text)
            sections.append({
                "document_name": name,
                "section_number": numbers[position],
                "section_text": _single_spaced(text[heading.start():end]),
            })
    return sections


def _split_sentences(section_text):
    return [piece for piece in re.split(config.SENTENCE_BOUNDARY_PATTERN, section_text) if piece]


def _joined_length(sentences):
    return len(" ".join(sentences)) if sentences else 0


def build_chunk_records(section_entries):
    """Cut each section into overlapping passages of whole sentences."""
    records = []
    passage_numbers = {}
    for section in section_entries:
        name = section["document_name"]
        sentences = _split_sentences(section["section_text"])
        previous = []
        index = 0
        while index < len(sentences):
            carried = []
            if previous:
                for sentence in reversed(previous):
                    if _joined_length([sentence] + carried) <= config.CHUNK_OVERLAP_CHARACTERS:
                        carried.insert(0, sentence)
                    else:
                        break
                while carried and _joined_length(carried + [sentences[index]]) > config.CHUNK_SIZE_CHARACTERS:
                    carried.pop(0)
            fresh = [sentences[index]]
            index += 1
            while index < len(sentences) and \
                    _joined_length(carried + fresh + [sentences[index]]) <= config.CHUNK_SIZE_CHARACTERS:
                fresh.append(sentences[index])
                index += 1
            passage = carried + fresh
            number = passage_numbers.get(name, 0)
            passage_numbers[name] = number + 1
            records.append({
                "chunk_id": f"{name}::{number:04d}",
                "chunk_text": " ".join(passage),
                "document_name": name,
                "section_number": section["section_number"],
                "passage_number": number,
                "carried_characters": len(" ".join(carried)) + 1 if carried else 0,
            })
            previous = passage
    return records
