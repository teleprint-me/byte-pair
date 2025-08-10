"""
@file byte.model
"""

import argparse
import collections
import json
import regex as re


def get_freqs(words: list[str]):
    freqs = {}
    for word in words:
        if word in freqs:
            freqs[word] += 1
        else:
            freqs[word] = 1
    return freqs


def get_words(path: str = None) -> list[str]:
    # It's very annoying that there's no escaping this.
    # The primary issue with regex is that delims are omitted from the final results.
    # I've tried using other tools, but the inverse is true as well. e.g. re.split()
    # A key realization I had was that for most langs,
    # we only need to split at spaces and punctuation.
    # The caveat is that we must capture spaces and punctuation as well.
    # The best way to probably achieve this is to do it manually.
    # https://github.com/openai/gpt-2/blob/master/src/encoder.py#L53C35-L53C109
    PRE = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
    if path:
        with open(path, "r", encoding="utf-8") as file:
            return re.findall(PRE, file.read())
    return re.findall(PRE, "lo low lower newest wide wider widest")


def get_vocab(path: str = None, stop: str = None) -> dict[str, int]:
    words = get_words(path)
    freqs = get_freqs(words)
    vocab: dict[str, int] = {}
    for w, c in freqs.items():
        if stop:
            symbols = " ".join(list(w)) + f" {stop}"
        else:
            symbols = " ".join(list(w))
        vocab[symbols] = c
    return vocab


def get_pairs(vocab: dict[str, int]) -> dict[tuple[str, str], int]:
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()  # relies on space-separated symbols
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i + 1])] += freq
    return pairs


def get_best_pair(
    pairs: dict[tuple[str, str], int]
) -> tuple[tuple[str, str], int] | tuple[None, int]:
    best_pair = None
    best_freq = -1
    for pair, freq in pairs.items():
        if freq > best_freq:
            best_pair, best_freq = pair, freq
        elif freq == best_freq and best_pair is not None and pair < best_pair:
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
        best = get_best_pair(pairs)[0]
        vocab = get_merges(vocab, best)
        print(f"best[{i}]: {best}")
    return vocab


parser = argparse.ArgumentParser()
parser.add_argument("-m", "--merges", default=10, type=int)
parser.add_argument("-c", "--corpus", default=None, type=str)
parser.add_argument("-e", "--eos", default=None, type=str)
args = parser.parse_args()

vocab = get_vocab(args.corpus, args.eos)
print("Initial Vocab:")
print(json.dumps(vocab, indent=2))

vocab = train(vocab, args.merges)

print("Final Vocab")
print(json.dumps(vocab, indent=2))
