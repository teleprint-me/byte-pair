"""
byte_pair/markdown.py - A Python script for converting HTML files to Markdown format.

This script recursively processes files in the input directory, converting HTML documents to Markdown format. It employs BeautifulSoup and html2text libraries to clean HTML content and perform the conversion. The script also provides detailed logging for error handling during file operations.

Byte Pair Encoding (BPE) Tokenization for Natural Language Processing
Copyright (C) 2024 Austin Berrio
"""

import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Union

import click
import html2text
import tqdm
from bs4 import BeautifulSoup
from markdown_it import MarkdownIt

# Initialize logging
logging.basicConfig(level=logging.INFO)


def read(source_file: str) -> Optional[str]:
    """Read content from a source file."""
    try:
        with open(source_file, "r") as f:
            return f.read()
    except Exception as e:
        logging.error(f"An error occurred while reading from the source file: {e}")
    return None


def write(file_path: str, content: str) -> None:
    """Write content to a destination file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            f.write(content)
    except Exception as e:
        logging.error(f"An error occurred while writing to the destination file: {e}")


def clean_code_blocks(html: str) -> str:
    # Initialize BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")

    # Find all 'pre' tags
    for pre_tag in soup.find_all("pre"):
        # Find all 'a' tags within each 'pre' tag
        for a_tag in pre_tag.find_all("a"):
            # Replace 'a' tag with its contents
            a_tag.replace_with(a_tag.text)

    # Convert back to string
    cleaned_html = str(soup)

    return cleaned_html


def convert_html_to_markdown(
    html: str,
) -> str:
    h = html2text.HTML2Text()
    h.body_width = 0  # No line wrapping
    h.single_line_break = True  # Single newlines turn into <br>
    h.ignore_links = True
    h.ignore_images = True
    h.ignore_tables = True
    h.escape_all = True  # Don't escape special characters

    return h.handle(html).strip()


def replace_code_tags_with_backticks(markdown_text: str) -> str:
    # Replace `[code]` with triple backticks
    markdown_text = re.sub(r"\[code\]", "```", markdown_text)
    # Replace `[/code]` with triple backticks
    markdown_text = re.sub(r"\[/code\]", "```", markdown_text)
    return markdown_text


def collect_files(dir_entry: Union[str, os.DirEntry]) -> List[os.DirEntry]:
    """Collect all file entries in a directory recursively."""
    file_entry_list = []
    for entry in os.scandir(
        dir_entry.path if isinstance(dir_entry, os.DirEntry) else dir_entry
    ):
        if entry.is_file():
            file_entry_list.append(entry)
        elif entry.is_dir():
            file_entry_list.extend(collect_files(entry))
    return file_entry_list


def process_entry(file_entry: os.DirEntry, pbar: tqdm.tqdm, dry_run: bool) -> None:
    """Process a single directory entry (file) and convert it to Markdown if needed."""
    md = MarkdownIt()
    output_path = os.path.join("markdown_dataset", file_entry.path)

    html_content = read(file_entry.path)
    if html_content is None:
        logging.error(
            f"An error occurred while reading the file {file_entry.name}. Skipping."
        )
        return

    html_content = clean_code_blocks(html_content)
    markdown_content = convert_html_to_markdown(html_content)
    markdown_content = markdown_content.replace(".html", ".md")
    output_path = output_path.replace(".html", ".md")

    if not dry_run:
        write(output_path, markdown_content)
        pbar.update(1)
    else:
        logging.info(f"Would write {len(markdown_content)} bytes to {output_path}")


def traverse_directory(
    file_entry_list: List[os.DirEntry],
    n_threads: int,
    dry_run: bool,
) -> None:
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        with tqdm.tqdm(total=len(file_entry_list)) as pbar:
            for _ in executor.map(
                lambda file_entry: process_entry(file_entry, pbar, dry_run),
                file_entry_list,
            ):
                pass


@click.command()
@click.option(
    "-d",
    "--dry-run",
    is_flag=True,
    help="Perform a dry run and fake generating the C and C++ raw dataset.",
)
@click.option(
    "-p",
    "--dir-path",
    default="cppreference-html-20190607/reference/en",
    help="Path to the directory to process.",
)
@click.option(
    "-t",
    "--n_threads",
    default=os.cpu_count() or 4,
    help="Number of threads to use for processing.",
)
def main(dry_run: bool, dir_path: str, n_threads: int):
    logging.info("Starting main function.")
    start_time = time.time()
    file_entry_list = collect_files(dir_path)
    traverse_directory(file_entry_list, n_threads, dry_run)
    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info(f"Elapsed {elapsed_time:.2f} seconds using {n_threads} threads.")


if __name__ == "__main__":
    main()
