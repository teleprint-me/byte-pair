"""
@file scanner.py
"""

import unicodedata as ud
import argparse
import json


# we only need to split at spaces and punctuation while preserving the delims.
def scan(text: str):
    buf = []
    cat_prev = None  # previous category

    def flush():
        if buf:
            yield "".join(buf)
            buf.clear()

    for ch in text:
        cat = ud.category(ch)
        if ch.isspace():
            yield from flush()
            yield ch
            cat_prev = None
            continue
        is_letter = cat.startswith("L")
        is_digit = cat.startswith("N")
        same_bucket = (
            (is_letter and cat_prev == "L")
            or (is_digit and cat_prev == "N")
            or (not is_letter and not is_digit and cat_prev == "P")
        )
        if not buf or same_bucket:
            buf.append(ch)
        else:
            yield from flush()
            buf.append(ch)
        cat_prev = "L" if is_letter else ("N" if is_digit else "P")
    yield from flush()


def get_words(path: str = None) -> list[str]:
    if path:
        with open(path, "r", encoding="utf-8") as file:
            return list(scan(file.read()))
    return list(scan("lo low lower newest wide wider widest"))


def get_freqs(words: list[str]) -> dict[str, int]:
    freqs = {}
    for word in words:
        if word in freqs:
            freqs[word] += 1
        else:
            freqs[word] = 1
    return freqs


def get_vocab(freqs: dict[str, int]) -> dict[str, int]:
    vocab = {}
    for word, freq in freqs.items():
        vocab[word] = vocab.get(word, 0) + freq
    return vocab


def get_pairs(vocab: dict[str, int]) -> dict[tuple[str, str], int]:
    pairs = {}
    for word, freq in vocab.items():
        symbols = list(word)
        for i in range(len(symbols) - 1):
            a, b = symbols[i], symbols[i + 1]
            pairs[(a, b)] = pairs.get((a, b), 0) + freq
    return pairs


def get_best(pairs: dict[tuple[str, str], int]) -> tuple[tuple[str, str], int]:
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
        syms = list(word)  # no need for spacing
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


parser = argparse.ArgumentParser()
parser.add_argument("-t", "--text", default=None, type=str)
args = parser.parse_args()

words = get_words(args.text)
print("Words:")
print(json.dumps(words, indent=2, ensure_ascii=False))

freqs = get_freqs(words)
print("Frequencies:")
print(json.dumps(freqs, indent=2, ensure_ascii=False))

vocab = get_vocab(freqs)
print("Vocab:")
print(json.dumps(vocab, indent=2, ensure_ascii=False))

pairs = get_pairs(vocab)
print("Pairs:")
for pair, freq in pairs.items():
    print(pair, freq)

best, freq = get_best(pairs)
print(f"best: ({best}, {freq})")

vocab = get_merges(vocab, best)
print("Merges:")
print(json.dumps(vocab, indent=2, ensure_ascii=False))
