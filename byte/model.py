"""
@file byte.model
"""

import argparse
import collections
import json
import re


def get_freqs(words: list[str]):
    # pair frequencies
    freqs = {}
    for word in words:
        if word in freqs:
            freqs[word] += 1
        else:
            freqs[word] = 1

    # pretty-print without losing structure
    print(json.dumps(freqs, indent=2, ensure_ascii=False))

    return freqs


def get_vocab(path: str | None = None, stop: str = "</w>") -> dict[str, int]:
    # Load raw words
    if path:
        with open(path, "r", encoding="utf-8") as f:
            words = re.findall(r"\S+", f.read())
    else:
        words = "lo low lower newest wide wider widest".split()

    # Count frequencies
    freqs = get_freqs(words)

    # Build BPE-initial vocab: "c h a r s </w>" per word
    vocab: dict[str, int] = {}
    for w, c in freqs.items():
        symbols = " ".join(list(w)) + f" {stop}"
        vocab[symbols] = c
    return vocab


def get_pairs(vocab: dict[str, int]) -> dict[tuple[str, str], int]:
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()  # relies on space-separated symbols
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i + 1])] += freq
    return pairs


def get_merges(vocab: dict[str, int], pair: tuple[str, str]) -> dict[str, int]:
    bigram = re.escape(" ".join(pair))
    p = re.compile(r"(?<!\S)" + bigram + r"(?!\S)")
    new_vocab: dict[str, int] = {}
    for word, freq in vocab.items():
        new_word = p.sub("".join(pair), word)
        new_vocab[new_word] = freq
    return new_vocab


parser = argparse.ArgumentParser()
parser.add_argument("-m", "--num-merges", default=10, type=int)
parser.add_argument("-f", "--file-path", default=None, type=str)
parser.add_argument("-e", "--eos", default="</w>", type=str)
args = parser.parse_args()

vocab = get_vocab(args.file_path, args.eos)
print("Initial Vocab:")
print(vocab)

for i in range(args.num_merges):
    pairs = get_pairs(vocab)
    if not pairs:
        print(f"Exhausted all pairs at step {i}.")
        break
    best = max(pairs, key=pairs.get)
    vocab = get_merges(vocab, best)
    print(best)

print("Final Vocab")
print(json.dumps(vocab, indent=2))
