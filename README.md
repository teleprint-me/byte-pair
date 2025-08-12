# Byte-Pair

## Overview

The Byte-Pair Encoder (BPE) is a tokenization method widely used in natural
language processing. This Python implementation is based on the paper
[**Neural Machine Translation of Rare Words with Subword Units**](https://arxiv.org/abs/1508.07909v5)
and influenced by
[Lei Mao's tutorial](https://leimao.github.io/blog/Byte-Pair-Encoding/).

## Features

- **Zero Dependencies** — The implementation is self-contained and does not
  require any external libraries.
- **Tokenization** — Perform tokenization using the Byte-Pair Encoding
  algorithm.
- **Vocabulary Management** — Build, analyze, and persist vocabularies.
- **Pair Frequency Analysis** — Calculate frequencies of token pairs for
  subword learning.

## Installation

1. **Clone the Repository**

   ```sh
   git clone https://github.com/teleprint-me/byte-pair.git
   cd byte-pair
   ```

2. **Set Up a Virtual Environment**

   ```sh
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt # optional dev dependencies
   ```

## Quick Start

### Dry Run with Verbose Output

```sh
python -m byte.model -c samples/simple.md -m 15 -v
```

### Save a Trained Tokenizer

```sh
python -m byte.model --save tokenizer.json -c samples/simple.md -m 20
```

### Process an Entire Directory

```sh
python -m byte.model --save tokenizer.json -c samples -m 2500
```

### Load and Inspect a Tokenizer

```sh
python -m byte.model --load tokenizer.json -v
```

### Predict Token Pairs for Text

```sh
python -m byte.model --load tokenizer.json -p "Hello, world!" -v
```

### Show Full CLI Options

```sh
python -m byte.model -h
```

## Contributing

Issues, feature suggestions, and pull requests are welcome. See the
[LICENSE](LICENSE) file for full licensing terms.

## License

This project is licensed under the GNU Affero General Public License (AGPL).

## Acknowledgments

Thanks to Lei Mao for the tutorial that helped shape this implementation.

## References

- [On Computable Numbers, with an Application to the Entscheidungsproblem](https://archive.org/details/Turing1936OnCumputableNumbers)
- [Prediction and Entropy of Printed English](https://archive.org/details/bstj30-1-50)
- [A New Algorithm for Data Compression Optimization](https://arxiv.org/abs/1209.1045)
- [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909)
- [Byte Pair Encoding](https://leimao.github.io/blog/Byte-Pair-Encoding/)
- [Better language models and their implications](https://openai.com/index/better-language-models/)
- [A Formal Perspective on Byte-Pair Encoding](https://arxiv.org/abs/2306.16837)
- [A Statistical Extension of Byte-Pair Encoding](https://paperswithcode.com/paper/a-statistical-extension-of-byte-pair-encoding)
- [Byte Pair Encoding is Suboptimal for Language Model Pretraining](https://arxiv.org/abs/2004.03720v2)
- [Language Model Tokenizers Introduce Unfairness Between Languages](https://arxiv.org/abs/2305.15425)
- [Rethinking Tokenization](https://arxiv.org/abs/2403.00417)
- [wHy DoNt YoU jUsT uSe ThE lLaMa ToKeNiZeR??](https://huggingface.co/blog/catherinearnett/dangers-of-tokenizer-recycling)
- [Egalitarian Language Representation in Language Models](https://arxiv.org/abs/2409.11501)
- [Dynamic Chunking for End-to-End Hierarchical Sequence Modeling](https://arxiv.org/abs/2507.07955v2)
