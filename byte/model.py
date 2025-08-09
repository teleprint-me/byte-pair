"""
@file byte.model
"""

import collections
import re


def get_pairs(vocab: dict[str, int]) -> dict[tuple[str, str], int]:
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()  # this wipes out whitespace
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i + 1]] += freq
    return pairs


def merge_vocab(vocab: dict[str, int], pair: tuple[str, str]) -> dict[str, int]:
    new_vocab = {}
    bigram = re.escape(" ".join(pair))  # whitespace get's merged here
    p = re.compile(r"(?<!\S)" + bigram + r"(?!\S)")
    for word in vocab:
        pair = p.sub("".join(pair), word)  # empty string if whitespace made it this far
        new_vocab[pair] = vocab[word]
    return new_vocab


vocab = {
    "l o w </w>": 5,
    "l o w e r </w>": 2,
    "n e w e s t </w>": 6,
    "w i d e s t </w>": 3,
}

num_merges = 10
for i in range(num_merges):
    pairs = get_pairs(vocab)
    best = max(pairs, key=pairs.get)
    vocab = merge_vocab(vocab, best)
    print(best)
