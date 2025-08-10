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
