"""
@file byte.wiki
@brief A simple example of using the Byte Pair Model.
"""

import argparse
import os
import re
from pathlib import Path

import pandas as pd
import requests

from byte.model import Tokenizer, Vocab


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and tokenize Wikipedia data."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Path to save the parquet files to.",
    )
    parser.add_argument(
        "--num-parts",
        type=int,
        required=False,
        default=1,
        help="Number of parquet file parts to download.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        required=False,
        default=10,
        help="Number of samples to write.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=False,
        default=42,
        help="Random seed for sampling.",
    )
    parser.add_argument(
        "--num-merges",
        type=int,
        required=False,
        default=10,
        help="Number of merges to perform.",
    )
    parser.add_argument(
        "--save",
        type=str,
        required=True,
        help="Path to save the model.",
    )
    return parser.parse_args()


def download_file(url: str, path: Path):
    # download a single file from a URL to a local path
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get("Content-Length", 0))
    downloaded_size = 0
    chunk_size = 8192
    last_print = -1  # So 0% will print

    print(f"Downloading {total_size} bytes from {url} to {path}")
    with open(path, "wb") as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                downloaded_size += len(chunk)
                percent = int((downloaded_size / total_size) * 100) if total_size else 0
                if percent // 10 > last_print // 10:
                    print(f"[{percent}%] Downloaded {downloaded_size} bytes")
                    last_print = percent
    print(f"Download complete! {path}")


def get_parts(num_parts: int) -> list[str]:
    """
    Construct list of part filenames to download and process.
    """
    # Always start from part 0 as the "seed"
    base = "train-00000-of-00041.parquet"
    if num_parts <= 1:
        return [base]
    # Use regex to extract max_part value ("00041")
    m = re.match(r".*-of-(\d+)\.parquet", base)
    max_part = int(m.group(1)) if m else 41  # fallback if no match
    return [f"train-{i:05d}-of-{max_part:05d}.parquet" for i in range(num_parts)]


def download_parts(url: str, output_dir: Path, parts: list[str]) -> None:
    # Download all necessary parquet parts (skip if present)
    for part in parts:
        print(f"Preparing {part}...")
        path = output_dir / part
        if not path.is_file():
            download_file(f"{url}/{part}", path)
        else:
            print(f"Part already exists: {path}")


def build_corpus(
    output_dir: Path, parts: list[str], num_samples: int, seed: int = 42
) -> list[str]:
    # Build corpus from all sampled text
    corpus = []
    print("Building corpus for training...")
    for part in parts:
        path = output_dir / part
        print(f"Processing {path}...")
        df = pd.read_parquet(path)

        if "text" not in df.columns:
            raise ValueError(f"Column 'text' not found in DataFrame {path}")

        n = min(num_samples, len(df))
        samples = df["text"].sample(n=n, random_state=seed)
        corpus.extend(samples)
        print(f"Added {len(samples)} samples from {path} to corpus.")
    print(f"Corpus built: {len(corpus)} total samples.")
    return corpus


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    url = "https://huggingface.co/datasets/wikimedia/wikipedia/resolve/main/20231101.en"
    parts = get_parts(args.num_parts)
    download_parts(url, output_dir, parts)
    corpus = build_corpus(output_dir, parts, args.num_samples, args.seed)
    text = "\n".join(corpus)  # join all samples into a single string
    print(f"Corpus text length: {len(text)} characters.")

    # Tokenizer workflow
    vocab = Vocab.tokenize(text)
    tokenizer = Tokenizer(vocab)
    print("Tokenizer initialized.")

    tokenizer.train(args.num_merges)
    print("Model trained successfully.")

    tokenizer.save(args.save)
    print(f"Tokenizer with {len(tokenizer)} tokens saved to {args.save}")

    # Smoke test: encode and decode an example string
    example = "Hello, world"
    print(f"\nEncoding sample text: {example!r}")
    ids = tokenizer.encode(example)
    print("Encoded:", ids)
    print("Decoded:", tokenizer.decode(ids))


if __name__ == "__main__":
    main()
