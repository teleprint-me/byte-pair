"""
byte_pair/cli/ocr.py

A command-line OCR tool that performs various image processing operations and extracts text from images.

Byte Pair Encoding (BPE) Tokenization for Natural Language Processing
Copyright (C) 2024 Austin Berrio
"""

import argparse

from byte_pair.processor.ocr import ImageProcessor


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Perform image processing operations and extract text from images."
    )
    parser.add_argument(
        "-i",
        "--image_path",
        required=True,
        type=str,
        help="The file path to the input image.",
    )
    parser.add_argument(
        "-o",
        "--text_path",
        type=str,
        default="",
        help="The file path to save the extracted text. Default is an empty string (print to console).",
    )
    parser.add_argument(
        "-r",
        "--rotate",
        type=int,
        default=0,
        help="Rotate the image by the specified angle in degrees. Default is 0.",
    )
    parser.add_argument(
        "-s",
        "--scale",
        type=int,
        default=0,
        help="Scale the image by the specified factor. Default is 0.",
    )
    parser.add_argument(
        "-g",
        "--grayscale",
        action="store_true",
        help="Convert the image to grayscale.",
    )
    parser.add_argument(
        "-a",
        "--contrast",
        action="store_true",
        help="Enhance the image contrast.",
    )
    parser.add_argument(
        "-b",
        "--burn",
        action="store_true",
        help="Burn the image by adjusting brightness and contrast.",
    )
    parser.add_argument(
        "-p",
        "--preprocess",
        action="store_true",
        help="Preprocess the image for text extraction using adaptive thresholding and erosion-dilation.",
    )
    parser.add_argument(
        "-u",
        "--contours",
        action="store_true",
        help="Extract text from image using contours and bounding rectangular areas.",
    )
    return parser.parse_args()


def main(args: argparse.Namespace):
    """
    Perform image processing operations and extract text from images.
    """
    processor = ImageProcessor(args.image_path)

    if args.rotate:
        processor.rotate_image(args.rotate)

    if args.scale:
        processor.scale_image(args.scale)

    if args.grayscale:
        processor.grayscale_image()

    if args.contrast:
        processor.contrast_image()

    if args.burn:
        processor.burn_image()

    if args.preprocess:
        processor.preprocess_image()

    if args.contours:
        text = processor.extract_text_from_image_contours()
    else:
        text = processor.extract_text_from_image()

    if args.text_path:
        with open(args.text_path, "w") as plaintext:
            plaintext.writelines(text.splitlines())
    else:
        print(text)


if __name__ == "__main__":
    args = get_arguments()
    main(args)
