"""Day 1 - Karthik - knowledge/pdf_extraction.py, chunk_splitting.py, standard_reading.py"""
import os
from decimal import Decimal

import pytest

canvas = pytest.importorskip("reportlab.pdfgen.canvas", reason="pip install reportlab to run these tests")

from config import config
from knowledge.chunk_splitting import build_chunk_records, split_documents_into_sections
from knowledge.pdf_extraction import extract_text_from_directory
from knowledge.standard_reading import read_connection_standard

HEADER = "Document CS 01 | Issue 3 | Owner: Connections | Review cycle: every twelve months"


def standard_lines(code, amps):
    return ["Connection Standard", HEADER,
            "1. Purpose and scope", "This standard covers one technology.",
            "2. The technology", f"The technology's code is {code}.",
            "3. Who applies", "An installer applies.",
            "4. Requirements An application meets these requirements in this order: "
            "1 Registered equipment. The equipment is listed in the approved register. "
            f"2 Rated current. The rated current is at most {amps} amps per phase. "
            "A requirement is never met by a sentence in a statement, and is never waived."]


def write_pdf(path, lines):
    pdf = canvas.Canvas(str(path))
    y = 800
    for line in lines:
        pdf.drawString(30, y, line)
        y -= 20
    pdf.save()


@pytest.fixture
def docs(tmp_path):
    write_pdf(tmp_path / config.CONNECTION_STANDARD_DOCUMENTS["EVCP"], standard_lines("EVCP", "32"))
    write_pdf(tmp_path / config.CONNECTION_STANDARD_DOCUMENTS["HP"], standard_lines("HP", "20"))
    (tmp_path / "notes.txt").write_text("not a pdf")
    return extract_text_from_directory(str(tmp_path))


def test_only_pdfs_in_name_order(docs):
    assert [d["document_name"] for d in docs] == sorted(config.CONNECTION_STANDARD_DOCUMENTS[t] for t in ("EVCP", "HP"))


def test_empty_folder_is_refused(tmp_path):
    with pytest.raises(ValueError, match=str(tmp_path)):
        extract_text_from_directory(str(tmp_path))


def test_sections_numbered_and_without_header(docs):
    sections = split_documents_into_sections(docs)
    assert [s["section_number"] for s in sections] == [1, 2, 3, 4] * 2
    assert sections[0]["section_text"].startswith("1. Purpose")
    assert all("Review cycle" not in s["section_text"] for s in sections)


def test_chunks_are_whole_sentences_with_overlap():
    long_text = " ".join(f"Sentence number {i} is written here to fill space." for i in range(40))
    chunks = build_chunk_records([{"document_name": "A.pdf", "section_number": 1, "section_text": "1. Title " + long_text}])
    assert len(chunks) > 1
    assert all(len(c["chunk_text"]) <= config.CHUNK_SIZE_CHARACTERS for c in chunks)
    assert chunks[0]["carried_characters"] == 0 and chunks[1]["carried_characters"] > 0
    assert [c["passage_number"] for c in chunks] == list(range(len(chunks)))
    assert len({c["chunk_id"] for c in chunks}) == len(chunks)


def test_each_technology_reads_its_own_figures(docs):
    evcp, hp = read_connection_standard("EVCP", docs), read_connection_standard("HP", docs)
    assert evcp["requirements"][1]["rules"] == {"maximum_amps": Decimal("32")}
    assert hp["requirements"][1]["rules"] == {"maximum_amps": Decimal("20")}
    assert evcp["requirements"][0]["requirement_kind"] == "registered_equipment"
    assert evcp["requirements"][0]["rules"] == {}


def test_unknown_technology_refused(docs):
    with pytest.raises(ValueError, match="No connection standard is configured for XYZ"):
        read_connection_standard("XYZ", docs)


@pytest.mark.skipif(not os.path.isdir(config.DOCS_DIRECTORY), reason="real documents not in this folder")
def test_real_documents():
    documents = extract_text_from_directory(config.DOCS_DIRECTORY)
    assert len(documents) == 16
    assert len(split_documents_into_sections(documents)) == 136
    for technology in config.CONNECTION_STANDARD_DOCUMENTS:
        assert read_connection_standard(technology, documents)["requirements"]
