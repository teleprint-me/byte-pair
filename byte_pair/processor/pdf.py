"""
byte_pair/processor/pdf.py

Copyright (C) 2024 Austin Berrio
"""

from pathlib import Path
from typing import List, Union

from poppler import load_from_file


def convert_pdf_to_text(file_name: Union[str, Path]) -> List[str]:
    """
    Convert a PDF document into a list of strings, where each string represents the text of a page.

    Args:
        file_name (Union[str, Path]): The path to the PDF file to be converted.

    Returns:
        List[str]: A list of strings representing the text of each page in the PDF.
    """
    pages: List[str] = []
    pdf_document = load_from_file(file_name=file_name)
    for index in range(pdf_document.pages):
        page = pdf_document.create_page(index)
        pages.append(page.text())
    return pages
