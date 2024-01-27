# Byte-Pair Encoder

## Overview
The Byte-Pair Encoder (BPE) is a powerful tokenization method used in natural language processing. This repository contains a Python implementation of BPE, inspired by the original paper [A New Algorithm for Data Compression Optimization](https://arxiv.org/abs/1209.1045), the referenced paper [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909v5), and [Lei Mao's blog tutorial](https://leimao.github.io/blog/Byte-Pair-Encoding/).

## Features
- Tokenization using Byte-Pair Encoding.
- Vocabulary management and statistics.
- Token pair frequency calculation.

## Getting Started
Follow these steps to get started with Byte-Pair Encoder:

1. Clone this repository.

   ```sh
   git clone https://github.com/teleprint-me/byte-pair.git
   ```

2. Install the required dependencies (if any).

   ```sh
   virtualenv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. Run the code to tokenize text and analyze token statistics using the included text.

   ```sh
   python encoder.py --corpus_file taming_shrew.md --given_token "red</w>" --n_merges 10000
   ```

## Usage
For detailed usage instructions, please refer to the codebase by running:

```sh
python encoder.py --help
```

## Contributing
Contributions are welcome! If you have suggestions, issues, or improvements, please don't hesitate to submit them through issues or pull requests.

## License
This project is licensed under the AGPL (GNU Affero General Public License). For more details, see the [LICENSE](LICENSE) file.

## Acknowledgments
Special thanks to Lei Mao for the blog tutorial that inspired this implementation.