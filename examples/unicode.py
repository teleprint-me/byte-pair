"""
@file unicode.py
@brief GPT-2 Unicode Mapping
"""

import json


def unicode() -> dict[int, str]:
    """Generate a GPT-2 Byte to Unicode map."""
    # https://github.com/openai/gpt-2/blob/master/src/encoder.py#L9
    codepoints = [c for c in range(256)]  # ascii base
    return {cpt: chr(cpt) for cpt in codepoints}


mapping = sorted((k, v) for k, v in unicode().items())
print(json.dumps(mapping, indent=2, ensure_ascii=False))
