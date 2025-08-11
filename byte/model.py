"""
@file byte.model
"""

import argparse
import json


def get_words(path: str = None) -> list[str]:
    if path:
        with open(path, "r", encoding="utf-8") as file:
            return file.read().split()
    return "lo low lower newest wide wider widest".split()


def get_freqs(words: list[str]) -> dict[str, int]:
    freqs = {}
    for word in words:
        if word in freqs:
            freqs[word] += 1
        else:
            freqs[word] = 1
    return freqs


def get_vocab(freqs: str = None, stop: str = None) -> dict[str, int]:
    vocab: dict[str, int] = {}
    for w, c in freqs.items():
        if stop:
            symbols = " ".join(list(w)) + f" {stop}"
        else:
            symbols = " ".join(list(w))
        vocab[symbols] = c
    return vocab


def get_pairs(vocab: dict[str, int]) -> dict[tuple[str, str], int]:
    new_pairs = {}
    for word, freq in vocab.items():
        syms = word.split()  # relies on space-separated symbols
        for i in range(len(syms) - 1):
            new_pair = (syms[i], syms[i + 1])
            new_pairs[new_pair] = new_pairs.get(new_pair, 0) + freq
    return new_pairs


def get_best(pairs: dict[tuple[str, str], int]) -> tuple[tuple[str, str], int]:
    best_pair = ()
    best_freq = -1
    for pair, freq in pairs.items():
        if freq > best_freq:
            best_pair, best_freq = pair, freq
        elif freq == best_freq and best_pair and pair < best_pair:
            best_pair = pair  # lexicographic tie-break
    return best_pair, best_freq


def get_merges(vocab: dict[str, int], pair: tuple[str, str]) -> dict[str, int]:
    a, b = pair
    new_vocab: dict[str, int] = {}

    for word, freq in vocab.items():
        syms = word.split()  # spaces need to be escaped
        out = []
        i = 0
        while i < len(syms):
            if i + 1 < len(syms) and syms[i] == a and syms[i + 1] == b:
                out.append(a + b)  # merge the pair
                i += 2  # skip the next symbol (non-overlapping)
            else:
                out.append(syms[i])
                i += 1
        new_word = " ".join(out)
        new_vocab[new_word] = new_vocab.get(new_word, 0) + freq  # sum collisions

    return new_vocab


def train(vocab: dict[str, int], num_merges: int) -> dict[str, int]:
    for i in range(num_merges):
        pairs = get_pairs(vocab)
        if not pairs:
            print(f"Exhausted all pairs at step {i}.")
            break
        best, freq = get_best(pairs)
        vocab = get_merges(vocab, best)
        print(f"best[{i}]: ({best}, {freq})")
    return vocab


parser = argparse.ArgumentParser()
parser.add_argument("-m", "--merges", default=10, type=int)
parser.add_argument("-c", "--corpus", default=None, type=str)
parser.add_argument("-e", "--eos", default=None, type=str)
args = parser.parse_args()

words = get_words(args.corpus)
freqs = get_freqs(words)
vocab = get_vocab(freqs, args.eos)
print("Initial Vocab:")
print(json.dumps(vocab, indent=2))

vocab = train(vocab, args.merges)

print("Final Vocab")
print(json.dumps(vocab, indent=2))
