"""
encoder.py - Byte Pair Encoding for building tokenization models

Byte Pair Encoding (BPE) Tokenization for Natural Language Processing
Copyright (C) 2024 Austin Berrio

Original Paper: https://arxiv.org/abs/1209.1045
Referenced Paper: https://paperswithcode.com/method/bpe
Blog Tutorial: https://leimao.github.io/blog/Byte-Pair-Encoding/
"""

import collections
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Token:
    start: str = "<w>"
    stop: str = "</w>"
    unknown: str = "</u>"


class Vocabulary:
    def __init__(
        self,
        token: Optional[Token] = None,
        n_merges: Optional[int] = 10000,
    ):
        self.n_merges = n_merges
        self.token = token if token else Token()
        self.table = collections.defaultdict(int)


def tokenize(text: str) -> List[str]:
    # Implement tokenization logic here
    pass


def train(corpus: List[str]):
    # Implement training logic here
    pass


def segment(word: str) -> List[str]:
    # Implement word segmentation logic here
    pass


def main():
    Vocabulary(n_merges=10000)  # adjustable parameter


if __name__ == "__main__":
    main()
