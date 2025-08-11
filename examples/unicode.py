"""
@file unicode.py
@brief GPT-2 Unicode Mapping
"""

import json


# https://github.com/openai/gpt-2/blob/master/src/encoder.py#L9
def unicode() -> dict[int, str]:
    """Generate a GPT-2 Byte to Unicode map."""
    codepoints = []  # Build unicode extensions
    for char in range(512):  # ASCII + Unicode extension range
        codepoints.append(char)  # add a codepoint
    return {cpt: chr(cpt) for cpt in codepoints}


print(json.dumps(unicode(), indent=2, ensure_ascii=False))
