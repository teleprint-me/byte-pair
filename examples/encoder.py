"""
encoder.py - Byte Pair Encoding for building tokenization models

Byte Pair Encoding (BPE) Tokenization for Natural Language Processing
Copyright (C) 2024 Austin Berrio

Original Paper: https://arxiv.org/abs/1209.1045
Referenced Paper: https://paperswithcode.com/method/bpe
Blog Tutorial: https://leimao.github.io/blog/Byte-Pair-Encoding/
"""

import argparse
import collections
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import tqdm


@dataclass(frozen=True)
class TokenConstants:
    start: str = "<w>"
    stop: str = "</w>"
    unknown: str = "</u>"


@dataclass
class Vocabulary:
    collection: Dict[str, int] = field(
        default_factory=lambda: collections.defaultdict(int)
    )
    n_merges: Optional[int] = 10000
    token_constants: Optional[TokenConstants] = field(default_factory=TokenConstants)


def get_token_length(token: str, token_constants: TokenConstants) -> int:
    """
    Calculate the length of a token, accounting for the stop token if present.

    Args:
        token (str): The token whose length needs to be calculated.
        token_constants (TokenConstants): The token constants instance object.

    Returns:
        int: The length of the token.
    """
    if token.endswith(token_constants.stop):
        # Adjust length to account for stop token
        return len(token) - len(token_constants.stop) + 1
    return len(token)


def sort_key(item: Tuple[str, int], token_constants: TokenConstants) -> Tuple[int, int]:
    """
    Define the sorting key for tokens based on their lengths and frequencies.

    Args:
        item (Tuple[str, int]): A tuple containing token and its frequency.
        token_constants (TokenConstants): The token constants instance object.

    Returns:
        Tuple[int, int]: A tuple containing token length and frequency.
    """
    return (get_token_length(item[0], token_constants), item[1])


def sort_tokens(
    token_frequencies: Dict[str, int], token_constants: TokenConstants
) -> List[str]:
    """
    Sort tokens based on their frequencies and token lengths.

    Args:
        token_frequencies (Dict[str, int]): A dictionary mapping tokens to their frequencies.
        token_constants (TokenConstants): The token constants instance object.

    Returns:
        List[str]: A list of sorted tokens.
    """
    sorted_tokens_tuple = sorted(
        token_frequencies.items(),
        key=lambda item: sort_key(item, token_constants),
        reverse=True,
    )
    return [token for (token, freq) in sorted_tokens_tuple]


def map_corpus(vocab: Vocabulary) -> Tuple[Dict[str, int], Dict[str, str]]:
    """
    Map tokens in the vocabulary to their original words and calculate token frequencies.

    Args:
        vocab (Vocabulary): The vocabulary object containing token frequency information.

    Returns:
        Vocabulary: The updated vocabulary object with tokenization mapping and frequencies.
    """
    vocab_frequency = collections.defaultdict(int)
    vocab_mapping = dict()

    for word, frequency in vocab.collection.items():
        split_word = word.split()  # Use space delimiter
        original_word = "".join(split_word)  # Omit space delimiter
        for token in split_word:
            vocab_frequency[token] += frequency
        vocab_mapping[original_word] = split_word

    return vocab_frequency, vocab_mapping


def get_vocabulary(corpus: List[str], **kwargs) -> Vocabulary:
    """
    Get the vocabulary based on the given corpus.

    Args:
        corpus (list): List of strings representing the text corpus.
        kwargs (dict): Arguments passed to Vocabulary constructor.

    Returns:
        Vocabulary: The updated Vocabulary instance containing the computed vocabulary.

    Raises:
        ValueError: If corpus is empty or contains non-string elements.
    """
    if not corpus or not all(isinstance(text, str) for text in corpus):
        raise ValueError("Corpus must be a list of strings.")

    vocab = Vocabulary(**kwargs)

    # Break down corpus into lines.
    for line in corpus:
        # Break down the line into words.
        for word in line.split():
            # Group space-separated characters by bounding them with a stop token.
            token = " ".join(list(word)) + vocab.token_constants.stop
            # Add token to vocab using a unique integer.
            vocab.collection[token] += 1

    return vocab


def calculate_token_pair_frequencies(vocab: Vocabulary) -> Dict[Tuple[str, str], int]:
    """
    Calculate token pair frequencies based on the provided vocabulary.

    Args:
        vocab (Vocabulary): The Vocabulary instance containing the token frequency information.

    Returns:
        dict: A dictionary mapping token pairs to their frequencies.
    """
    token_pair_frequencies = collections.defaultdict(int)

    for token, frequency in vocab.collection.items():
        symbols = token.split()

        for step in range(len(symbols) - 1):
            current_symbol = symbols[step]  # get current step
            next_symbol = symbols[step + 1]  # then get next step
            pair = (current_symbol, next_symbol)
            token_pair_frequencies[pair] += frequency

    return token_pair_frequencies


def merge_token_pair(vocab: Vocabulary, token_pair: Tuple[str, str]) -> Vocabulary:
    """
    Merge tokens in the vocabulary based on a given pair.

    Args:
        vocab (Vocabulary): The Vocabulary instance containing the token frequency information.
        token_pair (tuple): A tuple containing two strings representing the pair of tokens to merge.

    Returns:
        Vocabulary: The updated Vocabulary instance with merged tokens.
    """
    output_collection = collections.defaultdict(int)
    lookbehind = r"(?<!\S)"
    spaced_token_pair = re.escape(" ".join(token_pair))
    lookahead = r"(?!\S)"
    bigram_pattern = re.compile(lookbehind + spaced_token_pair + lookahead)

    for token, frequency in vocab.collection.items():
        merged_token_pair = bigram_pattern.sub("".join(token_pair), token)
        output_collection[merged_token_pair] += frequency

    # Consider using
    #   vocab.collection = output_collection
    # if memory becomes an issue upon scaling.
    return Vocabulary(
        collection=output_collection,
        n_merges=vocab.n_merges,
        token_constants=vocab.token_constants,
    )


def find_token_matches(token: str, string: str) -> List[Tuple[int, int]]:
    """
    Find all matches of a given token in the string.

    Args:
        token (str): The token to find in the string.
        string (str): The string to search within.

    Returns:
        List[Tuple[int, int]]: A list of tuples representing the start and end positions of each match.
    """
    token_reg = re.escape(token.replace(".", "[.]"))
    return [(m.start(0), m.end(0)) for m in re.finditer(token_reg, string)]


def tokenize_substring(
    substring: str,
    sorted_tokens: List[str],
    unknown_token: Optional[str] = "</u>",
) -> List[str]:
    """
    Tokenize a substring using sorted tokens.

    Args:
        substring (str): The substring to tokenize.
        sorted_tokens (List[str]): The list of sorted tokens for tokenization.
        unknown_token (str): The token to use for unknown sequences.

    Returns:
        List[str]: A list of tokens representing the tokenized substring.
    """
    if substring == "":
        return []

    tokens = []
    i = 0
    while i < len(substring):
        # Find the longest token that matches the start of the substring
        for token in sorted_tokens:
            if substring[i:].startswith(token):
                tokens.append(token)
                i += len(token)
                break
        else:
            # No matching token found; use unknown token and advance by one character
            tokens.append(unknown_token)
            i += 1

    return tokens


def tokenize_string(
    string: str,
    sorted_tokens: List[str],
    unknown_token: Optional[str] = "</u>",
) -> List[str]:
    """
    Tokenize a string based on a list of sorted tokens.

    Args:
        string (str): The string to tokenize.
        sorted_tokens (List[str]): The list of sorted tokens for tokenization.
        unknown_token (str): The token to use for unknown sequences.

    Returns:
        List[str]: A list of tokens representing the tokenized string.
    """
    string_tokens = []
    while string:
        matched = False
        for token in sorted_tokens:
            if string.startswith(token):
                string_tokens.append(token)
                string = string[len(token) :]
                matched = True
                break

        if not matched:
            string_tokens.append(unknown_token)
            string = string[1:]  # Move forward by one character

    return string_tokens


def tokenize(text: str) -> List[str]:
    # Implement tokenization logic here
    pass


def train(corpus: List[str]):
    # Implement training logic here
    pass


def segment(word: str) -> List[str]:
    # Implement word segmentation logic here
    pass


def main(args):
    vocab = None

    with open(args.corpus_file, "r") as file:
        vocab = get_vocabulary(
            corpus=file.readlines(),
            n_merges=args.n_merges,
            token_constants=TokenConstants(),
        )

    for _ in tqdm.tqdm(range(vocab.n_merges), desc="Merging vocabulary"):
        vocab_frequency, vocab_mapping = map_corpus(vocab=vocab)
        token_pair_frequencies = calculate_token_pair_frequencies(vocab=vocab)

        if not token_pair_frequencies:
            break

        top_token_pair = max(token_pair_frequencies, key=token_pair_frequencies.get)
        vocab = merge_token_pair(vocab=vocab, token_pair=top_token_pair)

    sorted_tokens = sort_tokens(
        token_frequencies=vocab_frequency,
        token_constants=vocab.token_constants,
    )
    given_token = args.given_token

    if given_token in vocab_mapping:
        # Tokenizer knows the given word
        results: List[str] = tokenize_string(
            string=given_token,
            sorted_tokens=sorted_tokens,
            unknown_token=vocab.token_constants.unknown,
        )
    else:
        # Tokenizer failed to recognize the given word
        results: List[str] = tokenize_string(
            string=given_token,
            sorted_tokens=sorted_tokens,
            unknown_token=vocab.token_constants.unknown,
        )

    for result in results:
        print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tokenize text using Byte-Pair Encoding."
    )
    parser.add_argument(
        "--corpus_file",
        type=str,
        required=True,
        help="Path to the corpus file for vocabulary creation.",
    )
    parser.add_argument(
        "--n_merges",
        type=int,
        default=10000,
        help="Number of merges to perform during vocabulary creation.",
    )
    parser.add_argument(
        "--given_token",
        type=str,
        required=True,
        help="The token to tokenize using the generated vocabulary.",
    )

    args = parser.parse_args()
    main(args)
