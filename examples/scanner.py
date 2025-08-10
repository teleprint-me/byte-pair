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


parser = argparse.ArgumentParser()
parser.add_argument("-t", "--text", default=None, type=str)
args = parser.parse_args()

words = list(set(get_words(args.text)))
print(json.dumps(words, indent=2, ensure_ascii=False))
