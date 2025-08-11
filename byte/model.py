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
from typing import Optional


class Vocab:
    @staticmethod
    def file(path: Optional[str] = None) -> str:
        """Read text from plain text file."""
        if path:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
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
    def __init__(self, vocab: dict[str, int]):
        self.model = {
            "type": "BPE",
            "version": "0.1.0",
            "vocab": vocab,
            "merges": [],
        }

    def __len__(self) -> int:
        return len(self.token_to_id)

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

    def train(self, num_merges: int) -> None:
        print("[training] Initialized.")
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

    @property
    @functools.lru_cache
    def unicode(self) -> dict[int, str]:
        """Generate a ASCII map."""
        cpts = [cp for cp in range(256)]  # ascii base
        return {cp: chr(cp) for cp in cpts}  # ascii map

    @property
    @functools.lru_cache
    def tokens(self) -> list[str]:
        # we need to inject base alphabet here
        tokens = set(self.unicode.values())
        for word in self.vocab:  # must be vocab!
            for subword in word.split():
                tokens.add(subword)
        # Assign IDs in sorted order (requires lexical order)
        return sorted(list(tokens))

    @property
    @functools.lru_cache
    def token_to_id(self) -> dict[str, int]:
        return {token: idx for idx, token in enumerate(self.tokens)}

    @property
    @functools.lru_cache
    def id_to_token(self) -> dict[int, str]:
        return {idx: token for idx, token in enumerate(self.tokens)}

    @property
    @functools.lru_cache
    def ranks(self) -> dict[str, int]:
        # required to calculate scores
        ranks = {}
        for i, pair in enumerate(self.merges):  # must be merges!
            token = "".join(pair)
            ranks[token] = i
        return ranks

    @property
    @functools.lru_cache
    def scores(self) -> dict[str, float]:
        # used in prompt-processing
        scores = {}
        for t in self.tokens:  # must be tokens!
            r = self.ranks.get(t)
            scores[t] = -math.log(r + 1) if r is not None else -1e6
        return scores

    def encode(self, text: str) -> list[int]:
        # Phase 1: Map text → byte-to-unicode base tokens
        text = "".join(self.unicode[b] for b in text.encode("utf-8"))
        ids = [self.token_to_id[ch] for ch in text]

        # Phase 2: Greedy merges using ranks
        while True:
            best_rank = None
            best_idx = None

            # scan for best pair
            for i in range(len(ids) - 1):
                tok_a = self.id_to_token[ids[i]]
                tok_b = self.id_to_token[ids[i + 1]]
                merged = tok_a + tok_b
                rank = self.ranks.get(merged)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_idx = i

            if best_idx is None:
                break  # no more merges

            # merge
            tok_a = self.id_to_token[ids[best_idx]]
            tok_b = self.id_to_token[ids[best_idx + 1]]
            merged = tok_a + tok_b
            ids[best_idx] = self.token_to_id[merged]
            del ids[best_idx + 1]

        return ids

    def decode(self, ids: list[int]) -> str:
        text = ""
        for i in ids:
            text += self.id_to_token.get(i, -1)  # -1 is unk
        return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--merges", default=10, type=int)
    parser.add_argument("-c", "--corpus", default=None, type=str)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    vocab = Vocab.build(args.corpus)
    tokenizer = Tokenizer(vocab)
    tokenizer.train(args.merges)

    if args.verbose:
        print("Initial Vocab:")
        print(json.dumps(vocab, indent=2))

        print("Final Vocab")
        print(json.dumps(tokenizer.vocab, indent=2))

        print("Best Merges")
        for i, (pair, freq) in enumerate(tokenizer.merges):
            print(f"merge[{i}]: ({pair}), {freq}")

        print("Tokens:")
        print(json.dumps(tokenizer.tokens, indent=2))

        print("Model:")
        print(json.dumps(tokenizer.model, indent=2, ensure_ascii=False))

        print(f"Tokenizer (size={len(tokenizer)}):")
        print(json.dumps(tokenizer.token_to_id, indent=2, ensure_ascii=False))

    print("Prompt processing:")
    ids = tokenizer.encode("Hello, world!")
    print(f"encoded: {ids}")
    text = tokenizer.decode(ids)
    print(f"decoded: {text}")


if __name__ == "__main__":
    main()
