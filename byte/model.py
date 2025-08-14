"""
Copyright © 2025 Austin Berrio

@file byte.model
@brief A compact, deterministic, stopless BPE trainer (character-level prototype).

- Deterministic tie-breaks (freq desc, then lexicographic)
- Correct pair counting and collision summing on merges
- Stable token id layout: base chars first, then merges in training order
- Rank & score tables (rank 0 handled correctly)
- JSON save/load of the model

Usage:
  python -m byte.model -c samples/simple.md -m 15 -v
  python -m byte.model --save my_bpe.json -c samples/simple.md -m 200
  python -m byte.model --load my_bpe.json -v
"""

import argparse
import functools
import json
import math
import os
from pathlib import Path
from typing import Optional


class Vocab:
    @staticmethod
    def file(path: Optional[str] = None) -> str:
        """Read text from plain text file."""

        if path and Path(path).is_file():
            with open(path, "r", encoding="utf-8") as f:
                return f.read()

        if path and Path(path).is_dir():
            content = ""
            for obj in os.listdir(path):
                with open(Path(path) / obj, "r", encoding="utf-8") as f:
                    content += f.read() + "\n"  # add newline after each file
            return content

        # default text if no path provided
        return "lo low lower newest wide wider widest"

    @staticmethod
    def pre_tokenize(text: str) -> list[str]:
        """Pre-tokenize the text input."""
        # break at all spaces
        return text.split()  # e.g. " ", "\t", "\r", "\n", etc.

    @staticmethod
    def frequencies(pre: list[str]) -> dict[str, int]:
        """Accumulate word frequencies."""
        freqs = {}
        for word in pre:
            freqs[word] = freqs.get(word, 0) + 1
        return freqs

    @staticmethod
    def symbols(freqs: dict[str, int]) -> dict[str, int]:
        """Initialize word tokens."""
        vocab = {}
        for word, freq in freqs.items():
            symbols = " ".join(list(word))
            vocab[symbols] = freq
        return vocab

    @staticmethod
    def tokenize(text: str) -> dict[str, int]:
        pre = Vocab.pre_tokenize(text)
        freqs = Vocab.frequencies(pre)
        return Vocab.symbols(freqs)

    @staticmethod
    def build(path: Optional[str] = None) -> dict[str, int]:
        """Build the initial vocabulary."""
        text = Vocab.file(path)
        return Vocab.tokenize(text)


class Model:
    @staticmethod
    def pairs(vocab: dict[str, int]) -> dict[tuple[str, str], int]:
        new_pairs = {}
        for word, freq in vocab.items():
            syms = word.split()  # relies on space-separated symbols
            for i in range(len(syms) - 1):
                new_pair = (syms[i], syms[i + 1])
                new_pairs[new_pair] = new_pairs.get(new_pair, 0) + freq
        return new_pairs

    @staticmethod
    def best(pairs: dict[tuple[str, str], int]) -> tuple[tuple[str, str], int]:
        best_pair = ()
        best_freq = -1
        for pair, freq in pairs.items():
            if freq > best_freq:
                best_pair, best_freq = pair, freq
            elif freq == best_freq and best_pair and pair < best_pair:
                best_pair = pair  # lexicographic tie-break
        return best_pair, best_freq

    @staticmethod
    def merges(vocab: dict[str, int], pair: tuple[str, str]) -> dict[str, int]:
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


class Tokenizer:
    def __init__(self, vocab: dict[str, int], special: Optional[dict[str, str]] = None):
        self.model = {
            "type": "BPE",
            "version": "0.1.6",
            "vocab": vocab,
            "merges": [],
            "special": special
            or {
                "bos": "<|bos|>",  # beginning of sentence
                "eos": "<|eos|>",  # end of sentence
                "pad": "<|pad|>",  # padding token
                "unk": "<|unk|>",  # unknown token
            },
        }

    def __len__(self) -> int:
        return len(self.tokens)

    @property
    def type(self) -> str:
        return self.model["type"]

    @property
    def version(self) -> str:
        return self.model["version"]

    @property
    def vocab(self) -> dict[str, int]:
        return self.model["vocab"]

    @vocab.setter
    def vocab(self, value: dict[str, int]) -> None:
        self.model["vocab"] = value

    @property
    def merges(self) -> list[tuple[str, str]]:
        return self.model["merges"]

    @merges.setter
    def merges(self, value: list[tuple[str, str]]):
        self.model["merges"] = value

    @property
    def special(self) -> dict[str, str]:
        return self.model["special"]

    @special.setter
    def special(self, value: dict[str, str]):
        self.model["special"] = value

    def invalidate_cache(self) -> None:
        # property objects keep the cached function on .fget
        for prop in (
            "unicode",
            "tokens",
            "token_to_id",
            "id_to_token",
            "ranks",
            "scores",
        ):
            getattr(self.__class__, prop).fget.cache_clear()

    def train(self, num_merges: int) -> None:
        print("[training] Initialized.")
        self.invalidate_cache()
        self.merges = []
        for i in range(num_merges):
            pairs = Model.pairs(self.vocab)
            if not pairs:
                print(f"Exhausted all pairs at step {i}.")
                break
            best, freq = Model.best(pairs)
            self.merges.append(best)
            self.vocab = Model.merges(self.vocab, best)
            print(f"[training] merge[{i}] ({best}, {freq})")
        print("[training] Completed.")

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as file:
            json.dump(self.model, file, ensure_ascii=False, indent=2)

    def load(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as file:
            self.model = json.load(file)
        self.invalidate_cache()  # clear stale caches

    def pretty(self, s: str) -> str:
        out = []
        for ch in s:
            if ch.isprintable():
                out.append(ch)
            else:
                out.append(f"\\x{ord(ch):02x}")
        return "".join(out)

    def dump(self) -> None:
        # Dump the model parts
        print("Model:")
        model = self.model.copy()  # don't modify the original model!
        model["vocab_size"] = self.vocab_size
        model["num_merges"] = self.num_merges
        print(json.dumps(model, indent=2, ensure_ascii=False))
        # Dump the tokenizer
        print(f"Tokenizer (size={len(self)}):")
        view = {self.pretty(k): v for k, v in self.token_to_id.items()}
        for k, v in sorted(view.items(), key=lambda kv: kv[1]):  # sort by id, not token
            print(json.dumps(k), ":", v)

    @property
    @functools.lru_cache
    def unicode(self) -> dict[int, str]:
        # exact bijection: 0..255 -> single Unicode char (Latin-1 is perfect)
        return {b: chr(b) for b in range(256)}

    @property
    @functools.lru_cache
    def tokens(self) -> list[str]:
        # Initialize a set to store tokens
        tokens = set()
        # Add all base characters to the tokens set
        tokens.update(self.unicode.values())  # Must include alphabet!
        # Iterate over each pair in the learned merges
        tokens.update(a + b for a, b in self.merges)  # Must include merges!
        # Add all special tokens to the tokens set
        tokens.update(self.special.values())  # Must include special tokens!
        # Assign ids in lexical order
        return sorted(list(tokens))  # Must be sorted!

    @property
    def num_merges(self) -> int:
        return len(self.merges)

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    @property
    @functools.lru_cache
    def token_to_id(self) -> dict[str, int]:
        return {token: idx for idx, token in enumerate(self.tokens)}

    @property
    @functools.lru_cache
    def id_to_token(self) -> dict[int, str]:
        return {idx: token for idx, token in enumerate(self.tokens)}

    @property
    def bos_id(self) -> int:
        return self.token_to_id.get(self.special.get("bos"), -1)

    @property
    def eos_id(self) -> int:
        return self.token_to_id.get(self.special.get("eos"), -1)

    @property
    def pad_id(self) -> int:
        return self.token_to_id.get(self.special.get("pad"), -1)

    @property
    def unk_id(self) -> int:
        return self.token_to_id.get(self.special.get("unk"), -1)

    @property
    @functools.lru_cache
    def ranks(self) -> dict[str, int]:
        # required to calculate scores
        ranks = {}
        for i, (a, b) in enumerate(self.merges):  # must be merges!
            token = a + b
            ranks[token] = i
        return ranks

    @property
    @functools.lru_cache
    def scores(self) -> dict[str, float]:
        # used in prompt-processing
        scores = {}
        for t in self.tokens:  # must be tokens!
            r = self.ranks.get(t)
            scores[t] = -math.log(r + 1) if r is not None else float("-inf")
        return scores

    def encode(
        self, text: str, add_bos: bool = False, add_eos: bool = False
    ) -> list[int]:
        # Map text -> byte-to-unicode base tokens
        text = "".join(self.unicode[b] for b in text.encode("utf-8"))
        ids = [self.token_to_id[ch] for ch in text]

        # Greedy merges using scores
        while self.scores:  # skip if no merges were learned
            best_score = float("-inf")
            best_idx = None

            # scan for best pair
            for i in range(len(ids) - 1):
                tok_a = self.id_to_token.get(ids[i], self.special["unk"])
                tok_b = self.id_to_token.get(ids[i + 1], self.special["unk"])
                merged = tok_a + tok_b
                score = self.scores.get(merged, float("-inf"))
                if score > best_score:
                    best_score = score
                    best_idx = i

            if best_idx is None:
                break  # no more merges

            # merge (@todo should handle missing tokens)
            tok_a = self.id_to_token[ids[best_idx]]
            tok_b = self.id_to_token[ids[best_idx + 1]]
            merged = tok_a + tok_b
            ids[best_idx] = self.token_to_id[merged]
            del ids[best_idx + 1]

        if add_bos:
            ids.insert(0, self.bos_id)
        if add_eos:
            ids.append(self.eos_id)

        return ids

    def decode(self, ids: list[int]) -> str:
        stream = "".join(self.id_to_token.get(i, self.special["unk"]) for i in ids)
        return bytes(ord(ch) for ch in stream).decode("utf-8", errors="replace")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--merges", default=10, type=int)
    parser.add_argument("-c", "--corpus", default=None, type=str)
    parser.add_argument("-s", "--save", default=None, type=str)
    parser.add_argument("-l", "--load", default=None, type=str)
    parser.add_argument("-p", "--prompt", default="Hello, world!", type=str)
    parser.add_argument("-b", "--bos", action="store_true")
    parser.add_argument("-e", "--eos", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    tokenizer = None
    if args.load:
        # load model (vocab + merges) from disk
        tokenizer = Tokenizer(vocab={})
        tokenizer.load(args.load)
    else:
        vocab = Vocab.build(args.corpus)
        tokenizer = Tokenizer(vocab)
        tokenizer.train(args.merges)

    if args.save:
        tokenizer.save(args.save)

    if args.verbose:
        tokenizer.dump()

    print(f"Tokenizer (size={len(tokenizer)})")
    print(f"Prompt: {args.prompt}")
    ids = tokenizer.encode(args.prompt, args.bos, args.eos)
    print(f"encoded: {ids}")
    print(f"decoded: {tokenizer.decode(ids)}")


if __name__ == "__main__":
    main()
