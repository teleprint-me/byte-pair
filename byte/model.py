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


def get_vocab(path: str | None = None, stop: str = None) -> dict[str, int]:
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


def get_best_pair(pairs: dict[tuple[str, str], int]) -> tuple[tuple[str, str], int]:
    # get the best pair
    best_pair = None  # ("l", "o")
    best_freq = -1  # frequency
    for pair, freq in pairs.items():
        # break ties (mitigates collisions and overlapping boundaries)
        if (freq > best_freq) or (
            freq == best_freq and (best_pair is None or pair < best_pair)
        ):
            best_pair = pair
            best_freq = freq
    return best_pair, best_freq


def get_merges(vocab: dict[str, int], pair: tuple[str, str]) -> dict[str, int]:
    a, b = pair
    new_vocab: dict[str, int] = {}

    for word, freq in vocab.items():
        syms = word.split()
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
        print(best)
    return vocab


def trace(words: list[str], merges: list[tuple[str, str]], eos: str | None):
    # Precompile all merge regexes once, in order
    pats = [
        re.compile(rf"(?<!\S){re.escape(a)} {re.escape(b)}(?!\S)") for (a, b) in merges
    ]

    def apply_word(w: str) -> str:
        s = " ".join(list(w)) + (f" {eos}" if eos else "")
        for p, (a, b) in zip(pats, merges):
            s = p.sub(a + b, s)
        return s

    for w in words:
        print(f"{w:>12} -> {apply_word(w)}")


parser = argparse.ArgumentParser()
parser.add_argument("-m", "--num-merges", default=10, type=int)
parser.add_argument("-f", "--file-path", default=None, type=str)
parser.add_argument("-e", "--eos", default=None, type=str)
args = parser.parse_args()

vocab = get_vocab(args.file_path, args.eos)
print("Initial Vocab:")
print(json.dumps(vocab, indent=2))

vocab = train(vocab, args.num_merges)

print("Final Vocab")
print(json.dumps(vocab, indent=2))
