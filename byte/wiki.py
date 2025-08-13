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


def download(url: str, path: str) -> Path:
    # download a parquet file from huggingface and save it locally
    response = requests.get(url, stream=True)  # ?download=true
    response.raise_for_status()
    total_size = 0
    print(f"Downloading {url} to {path}")
    with open(path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                total_size += len(chunk)
                f.write(chunk)
                print(f"Downloaded {total_size} bytes")
    print(f"Download complete! {path}")
    return Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and tokenize Wikipedia data."
    )
    parser.add_argument(
        "--part", type=str, required=False, help="Name of the part to download."
    )
    parser.add_argument(
        "--path", type=str, required=True, help="Path to save the parquet file."
    )
    return parser.parse_args()


# RangeIndex: 156289 entries, 0 to 156288
# Data columns (total 4 columns):
#  #   Column  Non-Null Count   Dtype
# ---  ------  --------------   -----
#  0   id      156289 non-null  object
#  1   url     156289 non-null  object
#  2   title   156289 non-null  object
#  3   text    156289 non-null  object
# dtypes: object(4)
# memory usage: 4.8+ MB
def main():
    args = parse_args()

    url = "https://huggingface.co/datasets/wikimedia/wikipedia/resolve/main/20231101.en"
    part = args.part or "train-00000-of-00041.parquet"
    path = Path(args.path)
    if not path.is_file():
        path = download(f"{url}/{part}", args.path)
    print(f"Loaded {path}")

    with open(path, "rb") as f:
        df = pd.read_parquet(f)

    print(df.head())
    print(df.info())
    print(df.describe())
    print(df.columns)

    for text in df["text"].sample(n=100):
        print(text)


if __name__ == "__main__":
    main()
