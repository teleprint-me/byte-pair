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
from typing import Dict, List, Tuple

from byte_pair.segment import read_preprocessed_text, segment_text


def map_tokens_to_words(vocab: Dict[str, int]) -> Dict[str, str]:
    """
    Map each token in the vocabulary to its original word.

    Args:
        vocab (Dict[str, int]): The vocabulary object.

    Returns:
        Dict[str, str]: A dictionary mapping each token to its original word.
    """
    token_to_word_map = {}
    for word in vocab:
        split_word = word.split()
        original_word = "".join(split_word)
        token_to_word_map[original_word] = split_word
    return token_to_word_map


def calculate_token_frequencies(vocab: Dict[str, int]) -> Dict[str, int]:
    """
    Calculate token frequencies based on the vocabulary.

    Args:
        vocab (Dict[str, int]): The vocabulary object containing token frequency information.

    Returns:
        Dict[str, int]: A dictionary mapping each token to its frequency.
    """
    token_frequencies = defaultdict(int)
    for word, frequency in vocab.items():
        for token in word.split():
            token_frequencies[token] += frequency
    return token_frequencies


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


def prepare_vocab(segmented_text: List[str]) -> Dict[str, int]:
    """
    Prepare the initial vocabulary from segmented text with accurate frequencies.

    Args:
        segmented_text (List[str]): List of words from the segmented text.

    Returns:
        Dict[str, int]: Dictionary with tokens as keys and frequencies as values.
    """
    vocab = defaultdict(int)
    for word in segmented_text:
        # Forming the token with spaces between characters and an end-of-word symbol
        token = " ".join(word) + " </w>"
        # Increment the frequency count for each occurrence
        vocab[token] += 1
    return vocab


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
        preprocessed_text = [line for line in read_preprocessed_text(args.input_file)]
        segmented_text = segment_text(" ".join(preprocessed_text))
    else:
        segmented_text = segment_text(
            "This is just an *example*.\n"
            "**Natural language processing** is fun!\n"
            "There is a _cool_ breeze today.\n"
            "There is a _warm_ breeze today.\n"
        )

    vocab = prepare_vocab(segmented_text)
    print(segmented_text)
    print(vocab)

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
        print(f"Merge #{i + 1}: {best}")

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
