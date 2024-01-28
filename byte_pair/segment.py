"""
byte_pair/segment.py - Text segmentation for Byte-Pair Encoding

Byte Pair Encoding (BPE) Tokenization for Natural Language Processing
Copyright (C) 2024 Austin Berrio

Original Paper: https://arxiv.org/abs/1209.1045
Referenced Paper: https://arxiv.org/abs/1508.07909v5
Referenced Code: https://github.com/rsennrich/subword-nmt
"""
import argparse
import json
import re
from typing import List

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


def read_preprocessed_text(input_file: str) -> List[str]:
    """
    Read pre-processed text from a plain text file.

    Args:
        input_file (str): Path to the file containing pre-processed text.

    Returns:
        List[str]: A list of raw lines of pre-processed text from the input_file.
    """
    with open(input_file, "r", encoding="utf-8") as file:
        lines = file.readlines()
    return lines


def save_segmented_text(segments: List[str]) -> None:
    """
    Write segmented text to a JSON file.
    """
    ...


def clean_text(corpus: str) -> str:
    # Example: Replace multiple spaces with a single space
    corpus = re.sub(r"\s+", " ", corpus)
    # Add other cleaning rules as necessary
    return corpus


def segment_text(corpus: str, lower: bool = False) -> List[str]:
    if lower:
        corpus = corpus.lower()
    corpus = clean_text(corpus)
    return TOKEN_REGEX.findall(corpus)


def main(args: argparse.Namespace) -> None:
    preprocessed_text = read_preprocessed_text(args.input_file)
    segmented_text = segment_text(preprocessed_text, args.lower)

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
        "--input_file",
        type=str,
        required=True,
        help="Path to the output file for the segmented vocabulary.",
    )
    args = parser.parse_args()
    main(args)
