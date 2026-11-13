"""Entry point 4: builds the knowledge base from the sixteen documents and reads it back.

    python run_knowledge_indexing.py
"""

import textwrap

from config import config
from knowledge.chunk_splitting import build_chunk_records, split_documents_into_sections
from knowledge.pdf_extraction import extract_text_from_directory
from knowledge.vector_index import build_vector_index, open_vector_index
from tools.retrieval_tools import read_documentation_to_provide


def main():
    """Rebuild the store, then read each standard's documentation to provide back from it."""
    documents = extract_text_from_directory(config.DOCS_DIRECTORY)
    sections = split_documents_into_sections(documents)
    passages = build_chunk_records(sections)
    stored = build_vector_index(passages, config.VECTOR_STORE_DIRECTORY)
    print(f"{len(documents)} documents, {len(sections)} sections, {stored} passages indexed in "
          f"{config.VECTOR_STORE_DIRECTORY}")
    print("what each connection standard asks the installer to provide, read back from the index")
    collection = open_vector_index(config.VECTOR_STORE_DIRECTORY)
    for technology, document_name in config.CONNECTION_STANDARD_DOCUMENTS.items():
        text = read_documentation_to_provide(collection, technology)
        print(f"  {technology}  {document_name}")
        print(textwrap.fill(text, width=100, initial_indent="      ", subsequent_indent="      "))


if __name__ == "__main__":
    main()
