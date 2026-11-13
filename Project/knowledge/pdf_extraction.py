"""Gets the text out of the PDFs and does nothing else with it."""

import os

from pypdf import PdfReader

from config import config


def extract_text_from_directory(docs_directory):
    """Read every PDF in a folder once, in name order."""
    if not os.path.isdir(docs_directory):
        raise ValueError(f"The folder {docs_directory} holds no PDF document")
    names = sorted(name for name in os.listdir(docs_directory)
                   if name.lower().endswith(config.PDF_FILE_SUFFIX.lower())
                   and os.path.isfile(os.path.join(docs_directory, name)))
    if not names:
        raise ValueError(f"The folder {docs_directory} holds no PDF document")
    entries = []
    for name in names:
        path = os.path.join(docs_directory, name)
        reader = PdfReader(path)
        if not reader.pages:
            raise ValueError(f"The PDF {path} has no pages")
        page_texts = []
        for page in reader.pages:
            text = page.extract_text() or ""
            if not text.strip():
                raise ValueError(f"A page of the PDF {path} gives up no text")
            page_texts.append(text)
        entries.append({"document_name": name, "document_text": "\n".join(page_texts)})
    return entries
