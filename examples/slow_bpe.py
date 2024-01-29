"""
examples/slow_bpe.py - Neural Machine Translation of Rare Words with Subword Units

Byte Pair Encoding (BPE) Tokenization for Natural Language Processing
Copyright (C) 2024 Austin Berrio

Paper: https://arxiv.org/abs/1508.07909v5
"""
import argparse
import collections
import json
import re
from typing import Dict, Tuple


def get_stats(vocab: Dict[str, int]) -> Dict[Tuple[str, str], int]:
    """
    Calculate frequencies of pairs of adjacent symbols in the vocabulary.

    Args:
        vocab (dict): Dictionary with space-separated symbols as keys and frequencies as values

    Returns:
        dict: Dictionary of symbol pairs (tuple) and their combined frequency
    """
    symbol_pairs_frequency = collections.defaultdict(int)

    for word, frequency in vocab.items():
        symbols = word.split()
        for index in range(len(symbols) - 1):
            symbol_pair = (symbols[index], symbols[index + 1])
            symbol_pairs_frequency[symbol_pair] += frequency

    return symbol_pairs_frequency


def merge_vocab(
    symbol_pair: Tuple[str, str], input_vocab: Dict[str, int]
) -> Dict[str, int]:
    """
    Merge a given pair of symbols in the vocabulary.

    Args:
        symbol_pair (tuple): Tuple of two strings, the pair of symbols to merge
        input_vocab (dict): Input vocabulary

    Returns:
        dict: New vocabulary with the specified pair merged
    """
    output_vocab = {}
    bigram = re.escape(" ".join(symbol_pair))
    pattern = re.compile(r"(?<!\S)" + bigram + r"(?!\S)")

    for word in input_vocab:
        merged_word = pattern.sub("".join(symbol_pair), word)
        output_vocab[merged_word] = input_vocab[word]

    return output_vocab


def load_vocab_from_json(json_file: str) -> Dict[str, int]:
    """
    Load a vocabulary from a JSON file.

    Args:
        json_file (str): Path to the JSON file containing the vocabulary.

    Returns:
        dict: Vocabulary loaded from the JSON file.
    """
    try:
        with open(json_file, "r") as file:
            vocab = json.load(file)
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Error loading JSON vocabulary: {e}")

    return vocab


def main(args: argparse.Namespace) -> None:
    if args.vocab_json:
        vocab = load_vocab_from_json(args.vocab_json)
    else:
        # Default vocabulary
        vocab = {
            "l o w </w>": 5,
            "l o w e r </w>": 2,
            "n e w e s t </w>": 6,
            "w i d e s t </w>": 3,
        }

    for i in range(args.num_merges):
        pairs = get_stats(vocab)
        best = max(pairs, key=pairs.get)
        vocab = merge_vocab(best, vocab)
        print(best)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--num_merges",
        type=int,
        default=10,
        help="Number of BPE merges (default is 10)",
    )

    parser.add_argument(
        "--vocab_json",
        type=str,
        help="Path to a JSON file containing the initial vocabulary (optional)",
    )

    args = parser.parse_args()

    main(args)
