"""
encode.py - Byte Pair Encoding for building tokenization models

Byte Pair Encoding (BPE) Tokenization for Natural Language Processing
Copyright (C) 2024 Austin Berrio

Paper: https://arxiv.org/abs/1508.07909v5
Blog: https://leimao.github.io/blog/Byte-Pair-Encoding/
"""

import argparse
import collections
import re


def get_stats(vocab):
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i + 1]] += freq
    return pairs


def merge_vocab(pair, v_in):
    v_out = {}
    bigram = re.escape(" ".join(pair))
    p = re.compile(r"(?<!\S)" + bigram + r"(?!\S)")
    for word in v_in:
        w_out = p.sub("".join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out


def main(args: argparse.Namespace) -> None:
    vocab = {
        "l o w </w>": 5,
        "l o w e r </w>": 2,
        "n e w e s t </w>": 6,
        "w i d e s t </w>": 3,
    }
    num_merges = 10

    for i in range(num_merges):
        pairs = get_stats(vocab)
        best = max(pairs, key=pairs.get)
        vocab = merge_vocab(best, vocab)
        print(best)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compress text corpus using Byte-Pair Encoding."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=False,
        help="Path to the input file for text based compression.",
    )
    args = parser.parse_args()
    main(args)
