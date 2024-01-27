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

TOKEN_REGEX = re.compile(r"\w+|[^\w\s]|'\w+", re.UNICODE)


def segment(text: str) -> List[str]:
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
        description="Tokenize text using Byte-Pair Encoding."
    )
    parser.add_argument(
        "--corpus_file",
        type=str,
        required=True,
        help="Path to the corpus file for vocabulary creation.",
    )
    args = parser.parse_args()
    main(args)
