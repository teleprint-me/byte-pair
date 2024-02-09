"""
examples/fast_bpe.py - A Formal Perspective on Byte-Pair Encoding

Byte Pair Encoding (BPE) Tokenization for Natural Language Processing
Copyright (C) 2024 Austin Berrio

Paper: https://aclanthology.org/2023.findings-acl.38.pdf
"""
import argparse
from collections import Counter
from typing import List, Tuple, Union


def byte_pair_encoding(
    input_sequence: Union[str, List[str]], vocabulary_size: int
) -> List[str]:
    """
    Perform Byte-Pair Encoding on a given sequence.

    Args:
        input_sequence (Union[str, List]): The input string or list of tokens.
        vocabulary_size (int): The desired size of the vocabulary.

    Returns:
        List: The encoded sequence.
    """

    for _ in range(vocabulary_size):
        # Count occurrences of adjacent pairs
        pairs = Counter(zip(input_sequence, input_sequence[1:]))
        # Find the most common pair
        most_common_pair = pairs.most_common(1)[0][0]
        # Merge the most common pair in the sequence
        input_sequence = merge_pairs(input_sequence, most_common_pair)

    return input_sequence


def merge_pairs(sequence: List[str], pair: Tuple[str, str]) -> List[str]:
    """
    Merge instances of a specific pair in the sequence.

    Args:
        sequence (List): The input sequence.
        pair (Tuple): The pair to merge in the sequence.

    Returns:
        List: The sequence with the pairs merged.
    """
    merged_sequence = []

    while sequence:
        # Check if the pair is at the start of the sequence
        if tuple(sequence[:2]) == pair:
            merged_sequence.append(pair)
            sequence = sequence[2:]
        else:
            merged_sequence.append(sequence.pop(0))

    return merged_sequence


def join_pairs(sequence: List, stop: str = "</w>") -> str:
    """
    Recursively join the tokens and pairs into a string, respecting the structure created by merges.

    Args:
    sequence (List): The sequence of tokens and pairs after BPE merges.
    stop (str): The end-of-word token.

    Returns:
    str: The joined sequence.
    """

    def join_item(item):
        # Base case: if the item is a string, return it
        if isinstance(item, str):
            return item if item != stop else ""

        # Recursively join items in the tuple
        joined = "".join(join_item(sub_item) for sub_item in item)

        # Check for end-of-word in the tuple and add stop token if present
        if stop in item:
            joined += " " + stop

        return joined

    # Process each item in the sequence
    joined_sequence = [join_item(item) for item in sequence]

    # Filter out empty strings and join the sequence with a separator
    return ", ".join(filter(None, joined_sequence))


def main(args: argparse.Namespace) -> None:
    # Split the input sequence into tokens
    tokens = args.input_sequence.split()

    # Perform initial byte pair encoding
    pairs = byte_pair_encoding(tokens, args.num_merges)

    # Iterate through the merges
    for _ in range(args.num_merges):
        # Count occurrences of adjacent pairs
        pair_counts = Counter(zip(pairs, pairs[1:]))
        if not pair_counts:
            break
        # Find the most common pair
        most_common_pair = pair_counts.most_common(1)[0][0]
        # Merge the most common pair in the sequence
        pairs = merge_pairs(pairs, most_common_pair)
        print(f"Merged pair: {most_common_pair}, Resulting tokens: {pairs}")

    # Join the final pairs into a string
    joined_pairs = join_pairs(pairs, args.stop_token)

    print("Final Joined Pairs:", joined_pairs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_sequence",
        type=str,
        default="l o w </w> l o w e r </w> l o w e s t </w>",
        help="Sequence of characters with an end-of-word token",
    )

    parser.add_argument(
        "--stop_token",
        type=str,
        default="</w>",
        help="The end-of-word token (default is </w>)",
    )

    parser.add_argument(
        "--num_merges", type=int, default=3, help="Number of BPE merges (default is 3)"
    )

    args = parser.parse_args()

    main(args)
