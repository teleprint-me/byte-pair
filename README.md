# Byte-Pair Encoder

## Overview

The Byte-Pair Encoder (BPE) is a powerful tokenization method widely used in natural language processing. This Python implementation of BPE is inspired by the paper [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909v5) and guided by [Lei Mao's educational tutorial](https://leimao.github.io/blog/Byte-Pair-Encoding/).

## Features

- **Tokenization**: Efficient tokenization using Byte-Pair Encoding.
- **Vocabulary Management**: Tools for managing and analyzing vocabulary.
- **Token Pair Frequency**: Calculate token pair frequencies for subword units.

## Getting Started

To get started with Byte-Pair Encoder, follow these simple steps:

1. **Clone the Repository**
   ```sh
   git clone https://github.com/teleprint-me/byte-pair.git
   ```

2. **Install Dependencies**
   ```sh
   virtualenv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Run the Code**
   ```sh
   python -m byte_pair.encode --input_file samples/taming_shrew.md --output_file local/vocab.json --n_merges 5000
   ```

## Usage

For comprehensive usage instructions and options, consult the documentation:

```sh
python -m byte_pair.encode --help
```

## Documentation

Detailed information on how to use and contribute to the project is available in the [documentation](docs).

## Contributing

Contributions are welcome! If you have suggestions, bug reports, or improvements, please don't hesitate to submit issues or pull requests.

## License

This project is licensed under the AGPL (GNU Affero General Public License). For detailed information, see the [LICENSE](LICENSE) file.

## Acknowledgments

Special thanks to Lei Mao for the blog tutorial that inspired this implementation.

## Additional Resources

- Original Paper: [A New Algorithm for Data Compression Optimization](https://arxiv.org/abs/1209.1045)
- Johns Hopkins Paper: [A Formal Perspective on Byte-Pair Encoding](https://aclanthology.org/2023.findings-acl.38.pdf)
- Amazon Research: [A Statistical Extension of Byte-Pair Encoding](https://paperswithcode.com/paper/a-statistical-extension-of-byte-pair-encoding)
