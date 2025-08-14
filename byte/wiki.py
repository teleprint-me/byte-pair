"""
@file byte.wiki
@brief A simple example of using the Byte Pair Model.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import requests

from byte.model import Tokenizer, Vocab


def download(url: str, path: Path):
    # download a parquet file from huggingface and save it locally
    response = requests.get(url, stream=True)  # ?download=true
    response.raise_for_status()
    total_size = int(response.headers.get("Content-Length", 0))
    downloaded_size = 0
    chunk_size = 8192  # 8KB chunks
    print(f"Downloading {total_size} bytes from {url} to {path}")
    with open(path, "wb") as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                downloaded_size += len(chunk)
                f.write(chunk)
                interval = downloaded_size // total_size * 100
                if interval % 10:  # interval between 0 and 10
                    print(f"[{interval}] Downloaded {downloaded_size} bytes")
    print(f"Download complete! {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and tokenize Wikipedia data."
    )
    parser.add_argument(
        "--part",
        type=str,
        required=False,
        help="Name of the part to download.",
    )
    parser.add_argument(
        "--parquet",
        type=str,
        required=True,
        help="Path to save the parquet file.",
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
    part = args.part or "train-00000-of-00041.parquet"
    path = Path(args.parquet)
    if not path.is_file():
        download(f"{url}/{part}", path)

    print(f"Loading {path}")
    with open(path, "rb") as f:
        df = pd.read_parquet(f)

    print(df.head())
    print(df.info())
    print(df.describe())
    print(df.columns)

    print("Sampling text...")
    text = ""
    for i, sample in enumerate(df["text"].sample(n=args.num_samples)):
        text += sample + "\n"
        print(f"Sample {i + 1}: {sample}")
    print("Done!")

    vocab = Vocab.tokenize(text)
    tokenizer = Tokenizer(vocab)
    print("Tokenizer loaded successfully!")

    tokenizer.train(args.num_merges)
    print("Model trained successfully!")

    tokenizer.save(args.save)
    print(f"Tokenizer with {len(tokenizer)} tokens saved to {args.save}")

    encode = "Hello, world"  # example text to encode
    print(f"Encoding text: {encode}")

    ids = tokenizer.encode(encode)
    print("Encoded text:", ids)
    print("Decoded text:", tokenizer.decode(ids))
    print("Tokenizer loaded successfully!")


if __name__ == "__main__":
    main()
