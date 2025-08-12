# Byte Pair Encoding

## Overview

The Byte-Pair Encoder (BPE) is a powerful tokenization method widely used in
natural language processing. This Python implementation of BPE is inspired by
the paper
[Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909v5)
and guided by
[Lei Mao's educational tutorial](https://leimao.github.io/blog/Byte-Pair-Encoding/).

## Features

- **Tokenization**: Efficient tokenization using Byte-Pair Encoding.
- **Vocabulary Management**: Tools for managing and analyzing vocabulary.
- **Token Pair Frequency**: Calculate token pair frequencies for subword units.

## Getting Started

To get started with Byte-Pair Encoder, follow these simple steps:

1. **Clone the Repository**

```sh
git clone https://github.com/teleprint-me/byte-pair.git
cd byte-pair
```

2. **Install Dependencies**

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3. **Run the Code**

### Dry Run

- **Verbose Output**: Run the model with verbose output to see detailed
  information during execution.

```sh
python -m byte.model -c samples/simple.md -m 15 -v
```

### Persist the Model

- **Save Tokenizer**: Persist the model to a JSON file for later use.

```sh
python -m byte.model --save tokenizer.json -c samples/simple.md -m 200
```

- **Process Directory**: Process a directory of plain text files.

```sh
python -m byte.model --save tokenizer.json -c samples -m 200
```

## Load the Model

- **Load Tokenizer**: Load the model from a JSON file and verify its
  functionality.

```sh
python -m byte.model --load tokenizer.json -v
```

## Predict Token Pairs

- **Predict Tokens**: Predict token pairs for a given text.

```sh
python -m byte.model --load tokenizer.json -p "Hello, world!" -v
```

## Usage

For comprehensive usage instructions and options, consult the documentation:

```sh
python -m byte.model -h
```

## Contributing

Contributions are welcome! If you have suggestions, bug reports, or
improvements, please don't hesitate to submit issues or pull requests.

## License

This project is licensed under the AGPL (GNU Affero General Public License).
For detailed information, see the [LICENSE](LICENSE) file.

## Acknowledgments

Special thanks to Lei Mao for the blog tutorial that inspired this
implementation.

## References

- [On Computable Numbers, with an application to the Entscheidungsproblem](https://archive.org/details/Turing1936OnCumputableNumbers)
- [Prediction and Entropy of Printed English](https://archive.org/details/bstj30-1-50)
- [A New Algorithm for Data Compression Optimization](https://arxiv.org/abs/1209.1045)
- [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909)
- [Byte Pair Encoding](https://leimao.github.io/blog/Byte-Pair-Encoding/)
- [A Formal Perspective on Byte-Pair Encoding](https://arxiv.org/abs/2306.16837)
- [A Statistical Extension of Byte-Pair Encoding](https://paperswithcode.com/paper/a-statistical-extension-of-byte-pair-encoding)
- [Byte Pair Encoding is Suboptimal for Language Model Pretraining](https://arxiv.org/abs/2004.03720v2)
- [Language Model Tokenizers Introduce Unfairness Between Languages](https://arxiv.org/abs/2305.15425)
- [Rethinking Tokenization](https://arxiv.org/abs/2403.00417)
- [wHy DoNt YoU jUsT uSe ThE lLaMa ToKeNiZeR??](https://huggingface.co/blog/catherinearnett/dangers-of-tokenizer-recycling)
- [Egalitarian Language Representation in Language Models](https://arxiv.org/abs/2409.11501)
- [Dynamic Chunking for End-to-End Hierarchical Sequence Modeling](https://arxiv.org/abs/2507.07955v2)
