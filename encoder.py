"""
encoder.py - Byte Pair Encoding for building tokenization models

Byte Pair Encoding (BPE) Tokenization for Natural Language Processing
Copyright (C) 2024 Austin Berrio

Original Paper: https://arxiv.org/abs/1209.1045
Referenced Paper: https://paperswithcode.com/method/bpe
Blog Tutorial: https://leimao.github.io/blog/Byte-Pair-Encoding/
"""

import collections
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class TokenConstants:
    start: str = "<w>"
    stop: str = "</w>"
    unknown: str = "</u>"


@dataclass
class Vocabulary:
    n_merges: Optional[int] = 10000
    token_constants: Optional[TokenConstants] = field(default_factory=TokenConstants)
    collection: Dict[str, int] = field(
        default_factory=lambda: collections.defaultdict(int)
    )
    frequency: Dict[str, int] = field(
        default_factory=lambda: collections.defaultdict(int)
    )
    mapping: Dict[str, List[str]] = field(default_factory=dict)


def get_token_length(token: str, token_constants: TokenConstants) -> int:
    """
    Calculate the length of a token, accounting for the stop token if present.

    Args:
        token (str): The token whose length needs to be calculated.
        token_constants (TokenConstants): The token constants instance object.

    Returns:
        int: The length of the token.
    """
    if token.endswith(token_constants.stop):
        # Adjust length to account for stop token
        return len(token) - len(token_constants.stop) + 1
    return len(token)


def sort_key(item: Tuple[str, int], token_constants: TokenConstants) -> Tuple[int, int]:
    """
    Define the sorting key for tokens based on their lengths and frequencies.

    Args:
        item (Tuple[str, int]): A tuple containing token and its frequency.
        token_constants (TokenConstants): The token constants instance object.

    Returns:
        Tuple[int, int]: A tuple containing token length and frequency.
    """
    return (get_token_length(item[0], token_constants.stop), item[1])


def sort_tokens(
    token_frequencies: Dict[str, int], token_constants: TokenConstants
) -> List[str]:
    """
    Sort tokens based on their frequencies and token lengths.

    Args:
        token_frequencies (Dict[str, int]): A dictionary mapping tokens to their frequencies.
        token_constants (TokenConstants): The token constants instance object.

    Returns:
        List[str]: A list of sorted tokens.
    """
    sorted_tokens_tuple = sorted(
        token_frequencies.items(),
        key=lambda item: sort_key(item, token_constants),
        reverse=True,
    )
    return [token for (token, freq) in sorted_tokens_tuple]


def map_corpus(vocab: Vocabulary) -> Vocabulary:
    """
    Map tokens in the vocabulary to their original words and calculate token frequencies.

    Args:
        vocab (Vocabulary): The vocabulary object containing token frequency information.

    Returns:
        Vocabulary: The updated vocabulary object with tokenization mapping and frequencies.
    """
    vocab.frequency = collections.defaultdict(int)
    vocab.mapping = dict()

    for word, frequency in vocab.collection.items():
        split_word = word.split()  # Use space delimiter
        original_word = "".join(split_word)  # Omit space delimiter
        for token in split_word:
            vocab.frequency[token] += frequency
        vocab.mapping[original_word] = split_word

    return vocab


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
