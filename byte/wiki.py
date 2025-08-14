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


def download(url: str, path: Path):
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


def main():
    args = parse_args()

    url = "https://huggingface.co/datasets/wikimedia/wikipedia/resolve/main/20231101.en"
    parts = ["train-00000-of-00041.parquet"]  # default part

    # split the base string using re.findall()
    if args.num_parts > 1:
        # split at each word, hyphen, or period
        # e.g. ['train', '-', '00000', '-', 'of', '-', '00041', '.', 'parquet']
        split = re.findall(r"\w+|-|.", args.part)
        # Get the final part as an integer
        last = int(split[6])  # max of 41 parts
        # Build the new parts list
        for i in range(1, args.num_parts + 1):  # inclusive?
            parts.append(f"train-{i:05d}-of-{last:05d}.parquet")

    os.makedirs(args.output_dir, exist_ok=True)
    for part in parts:
        print(f"Downloading {part}...")
        path = Path(args.output_dir) / part
        if not path.is_file():
            download(f"{url}/{part}", path)

    # need to compile corpus for training
    corpus = ""  # corpus is just a blob of text
    print("Building corpus for training...")
    for part in parts:
        path = Path(args.output_dir) / part
        print(f"Processing {path}...")
        with open(path, "rb") as f:
            df = pd.read_parquet(f)

        if "text" not in df.columns:
            raise ValueError(f"Column 'text' not found in DataFrame {path}")

        # Not sure if I should sample all samples or just a subset
        samples = df["text"].sample(n=args.num_samples)
        corpus += "\n".join(samples)
        print(f"Processed {len(samples)} samples")
    print("Corpus built successfully!")

    # tokenize the corpus
    vocab = Vocab.tokenize(corpus)
    tokenizer = Tokenizer(vocab)
    print("Tokenizer initialized!")

    tokenizer.train(args.num_merges)
    print("Model trained successfully!")

    tokenizer.save(args.save)
    print(f"Tokenizer with {len(tokenizer)} tokens saved to {args.save}")

    encode = "Hello, world"  # example text to encode
    print(f"Encoding text: {encode}")

    ids = tokenizer.encode(encode)
    print("Encoded text:", ids)
    print("Decoded text:", tokenizer.decode(ids))
    print("Encoding/decoding roundtrip successful.")


if __name__ == "__main__":
    main()
