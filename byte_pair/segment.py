"""
byte_pair/segment.py - Text segmentation for Byte-Pair Encoding

Byte Pair Encoding (BPE) Tokenization for Natural Language Processing
Copyright (C) 2024 Austin Berrio

Original Paper: https://arxiv.org/abs/1209.1045
Referenced Paper: https://arxiv.org/abs/1508.07909v5
Referenced Code: https://github.com/rsennrich/subword-nmt
"""
import argparse
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


def clean_text(text: str) -> str:
    # Example: Replace multiple spaces with a single space
    text = re.sub(r"\s+", " ", text)
    # Add other cleaning rules as necessary
    return text


def segment(text: str, lower: bool = False) -> List[str]:
    if lower:
        text = text.lower()
    text = clean_text(text)
    return TOKEN_REGEX.findall(text)


def main(args: argparse.Namespace) -> None:
    segmented_text: List[str] = list()

    with open(args.corpus_file, "r") as file:
        for line in file.readlines():
            segmented_line: List[str] = segment(line)
            segmented_text.extend(segmented_line)

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
    args = parser.parse_args()
    main(args)
