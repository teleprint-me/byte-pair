"""
byte_pair/cli/segment.py - Text segmentation for Byte-Pair Encoding

Byte Pair Encoding (BPE) Tokenization for Natural Language Processing
Copyright (C) 2024 Austin Berrio
"""

import argparse
import json
import re
from typing import Generator, List

TOKEN_REGEX = re.compile(
    r"""
    # Match email addresses
    [a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,4}|
    # Match URLs
    http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+|
    # Match words with apostrophes
    \w+'\w+|
    # Match other words
    \w+|
    # Match any non-whitespace character
    [^\w\s]""",
    re.VERBOSE | re.UNICODE,
)


def read_preprocessed_text(input_file: str) -> Generator[str, None, None]:
    """
    Read pre-processed text from a plain text file line by line.

    Args:
        input_file (str): Path to the file containing pre-processed text.

    Yields:
        Generator[str, None, None]: Yields one line of pre-processed text at a time.
    """
    try:
        with open(input_file, "r", encoding="utf-8") as file:
            for line in file:
                yield line.strip()
    except IOError as e:
        raise IOError(f"Error reading file {input_file}: {e}")


def save_segmented_text(segments: List[str], output_file: str) -> None:
    """
    Write segmented text to a JSON file.

    Args:
        segments (List[str]): Segmented text.
        output_file (str): Path to the output JSON file.

    Raises:
        IOError: If there's an error writing to the file.
    """
    try:
        with open(output_file, "w", encoding="utf-8") as file:
            json.dump(segments, file, ensure_ascii=False, indent=4)
    except IOError as e:
        raise IOError(f"Error writing to file {output_file}: {e}")


def clean_text(corpus: str) -> str:
    """
    Clean a given corpus by replacing multiple spaces with a single space and applying other cleaning rules as necessary.

    Args:
        corpus (str): The input text corpus to be cleaned.

    Returns:
        str: The cleaned text corpus.
    """
    corpus = re.sub(r"\s+", " ", corpus)
    # Add other cleaning rules as necessary
    return corpus


def segment_text(corpus: str, lower: bool = False) -> List[str]:
    """
    Segment a text corpus into a list of tokens using the TOKEN_REGEX pattern.

    Args:
        corpus (str): The input text corpus to be segmented.
        lower (bool, optional): If True, convert the text to lowercase before segmentation.

    Returns:
        List[str]: A list of segmented tokens.
    """
    if lower:
        corpus = corpus.lower()
    corpus = clean_text(corpus)
    return TOKEN_REGEX.findall(corpus)


def main(args: argparse.Namespace) -> None:
    """
    Main function for segmenting text corpus using Byte-Pair Encoding (BPE).

    Args:
        args (argparse.Namespace): Command-line arguments including input_file, output_file, and lowercase.
    """
    # Using generator to read preprocessed text line by line
    preprocessed_text = [line for line in read_preprocessed_text(args.input_file)]

    # Segmenting the text
    segmented_text = segment_text(" ".join(preprocessed_text), args.lowercase)

    # Option to output to a file or stdout
    if args.output_file:
        save_segmented_text(segmented_text, args.output_file)
    else:
        for seg in segmented_text:
            print(seg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Segment text into meaningful sub-units."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to the input file for vocabulary segmentation.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        help="Path to the output file for the segmented vocabulary.",
    )
    parser.add_argument(
        "--lowercase",
        action="store_true",
        help="Convert text to lowercase before segmentation.",
    )
    args = parser.parse_args()
    main(args)
