"""
@file unicode.py
@brief GPT-2 Unicode Mapping
"""

import json


# https://github.com/openai/gpt-2/blob/master/src/encoder.py#L9
def unicode() -> dict[int, str]:
    """Generate a GPT-2 Byte to Unicode map."""
    offset = 0  # Map non-printable to printable codepoints
    codepoints = []  # Build unicode extensions
    for i in range(256):  # ASCII + Unicode extension range
        if chr(i).isprintable():
            codepoints.append(i)  # add a codepoint
        if not chr(i).isprintable():
            codepoints.append(i + offset)  # get a unicode byte
        offset += 1  # increment by 1
    return {cpt: chr(cpt) for cpt in codepoints}


mapping = sorted((k, v) for k, v in unicode().items())
print(json.dumps(mapping, indent=2, ensure_ascii=False))
