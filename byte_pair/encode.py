"""
encode.py - Byte Pair Encoding for building tokenization models

Byte Pair Encoding (BPE) Tokenization for Natural Language Processing
Copyright (C) 2024 Austin Berrio

Paper: https://arxiv.org/abs/1508.07909v5
Blog: https://leimao.github.io/blog/Byte-Pair-Encoding/
"""

import argparse
import json
import re
from collections import defaultdict
from typing import Dict, Tuple


def get_stats(vocab: Dict[str, int]) -> Dict[Tuple[str, str], int]:
    """
    Calculate frequencies of pairs of adjacent symbols in the vocabulary.

    Args:
        vocab (dict): Dictionary with space-separated symbols as keys and frequencies as values

    Returns:
        dict: Dictionary of symbol pairs (tuple) and their combined frequency
    """
    symbol_pairs_frequency = defaultdict(int)

    for word, frequency in vocab.items():
        symbols = word.split()
        for index in range(len(symbols) - 1):
            symbol_pair = (symbols[index], symbols[index + 1])
            symbol_pairs_frequency[symbol_pair] += frequency

    return symbol_pairs_frequency


def merge_vocab(
    symbol_pair: Tuple[str, str],
    input_vocab: Dict[str, int],
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


def read_vocab(input_file: str) -> Dict[str, int]:
    """
    Read vocabulary from a text file.

    Args:
        input_file (str): Path to the file containing vocabulary

    Returns:
        dict: Dictionary with space-separated symbols as keys and frequencies as values
    """
    vocab = {}
    with open(input_file, "r", encoding="utf-8") as file:
        for line in file:
            word, freq = line.split()
            vocab[word] = int(freq)
    return vocab


def save_vocab(output_file: str, vocab: Dict[str, int]) -> None:
    """
    Write vocabulary to a JSON file.

    Args:
        output_file (str): Path to the file to write the vocabulary to
        vocab (dict): A dictionary containing the merged results
    """
    with open(output_file, "w") as file:
        json.dump(vocab, file, indent=4)


def main(args):
    """
    Main function to run the BPE encoding.

    Args:
        args (argparse.Namespace): Command-line arguments including input_file, output_file, and n_merges.
    """
    # Read vocabulary from input_file or use default vocabulary
    if args.input_file:
        vocab = read_vocab(args.input_file)
    else:
        vocab = {
            "l o w </w>": 5,
            "l o w e r </w>": 2,
            "n e w e s t </w>": 6,
            "w i d e s t </w>": 3,
        }

    # Set the number of merges or use the default value
    n_merges = args.n_merges if args.n_merges else 10

    # Perform BPE merges
    for i in range(n_merges):
        pairs = get_stats(vocab)
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        vocab = merge_vocab(best, vocab)
        # Uncomment the following line to print merge details
        # print(f"Merge #{i + 1}: {best}")

    # Save the resulting vocabulary to the output file if specified
    if args.output_file:
        save_vocab(args.output_file, vocab)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Byte-Pair Encoding (BPE) for text compression."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        help="Path to the input file containing the initial vocabulary (optional).",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        help="Path to the output file to save the final merged vocabulary (optional).",
    )
    parser.add_argument(
        "--n_merges",
        type=int,
        help="Number of BPE merges to perform (optional, default is 10).",
    )
    args = parser.parse_args()
    main(args)
