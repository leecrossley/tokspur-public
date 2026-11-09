#!/usr/bin/env python3
"""Derive compact normalization DATA from pinned official Unicode 9 UCD files.

No tokenizer/runtime source or generated tables from another engine are inputs.
The C++ normalization algorithm is separately authored in native/src/unicode.cpp.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

PINS = {
    "UnicodeData.txt": "68dfc414d28257b9b5d6ddbb8b466c768c00ebdf6cbf7784364a9b6cad55ee8f",
    "DerivedNormalizationProps.txt": "02d8e12cea7c61a9f3cc5fbf8eaceadf8da23fe83d60c64cf647088830b810cc",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ucd", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sources = {}
    for name, pin in PINS.items():
        data = (args.ucd / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != pin:
            raise RuntimeError(f"Unicode source differs from its recorded pin: {name}")
        sources[name] = data.decode("utf-8")
    excluded = set()
    for line in sources["DerivedNormalizationProps.txt"].splitlines():
        fields = line.split("#")[0].strip().split(";")
        if len(fields) < 2 or fields[1].strip() != "Full_Composition_Exclusion":
            continue
        ends = fields[0].strip().split("..")
        excluded.update(range(int(ends[0], 16), int(ends[-1], 16) + 1))
    combining = {}
    decomposition = {}
    for line in sources["UnicodeData.txt"].splitlines():
        fields = line.split(";")
        point = int(fields[0], 16)
        if int(fields[3]):
            combining[point] = int(fields[3])
        if fields[5] and not fields[5].startswith("<"):
            parts = tuple(int(p, 16) for p in fields[5].split())
            if len(parts) not in (1, 2):
                raise RuntimeError("Unexpected canonical decomposition length")
            decomposition[point] = parts
    ranges = []
    for point, value in sorted(combining.items()):
        if ranges and ranges[-1][1] + 1 == point and ranges[-1][2] == value:
            ranges[-1][1] = point
        else:
            ranges.append([point, point, value])
    composition = sorted(
        ((parts[0] << 21) | parts[1], point)
        for point, parts in decomposition.items()
        if len(parts) == 2 and point not in excluded
    )
    if len({key for key, _ in composition}) != len(composition):
        raise RuntimeError("Ambiguous canonical composition pair")
    lines = [
        "// Generated normalization DATA, not imported tokenizer implementation.",
        "// Regenerate using scripts/generate_unicode9.py and the pinned official UCD.",
        "// Copyright 1991-2016 Unicode, Inc. See UNICODE-LICENSE.txt.",
        "// Unicode 9 semantics are intentionally fixed for the pinned tokenizer.",
    ]
    lines.extend(f"// {name} SHA256 {pin}" for name, pin in PINS.items())
    lines += [
        "#pragma once",
        "#include <cstdint>",
        "namespace tokspur::unicode9_data {",
        "struct ClassRange { uint32_t first; uint32_t last; uint8_t value; };",
        "inline constexpr ClassRange combining[] = {",
    ]
    lines += [
        f"    {{0x{first:x}, 0x{last:x}, {value}}}," for first, last, value in ranges
    ]
    lines += [
        "};",
        "struct Decomposition { uint32_t point; uint32_t first; uint32_t second; };",
        "inline constexpr Decomposition decomposition[] = {",
    ]
    lines += [
        f"    {{0x{point:x}, 0x{parts[0]:x}, 0x{parts[1] if len(parts) == 2 else 0:x}}},"
        for point, parts in sorted(decomposition.items())
    ]
    lines += [
        "};",
        "struct Composition { uint64_t pair; uint32_t point; };",
        "inline constexpr Composition composition[] = {",
    ]
    lines += [f"    {{0x{key:x}ULL, 0x{point:x}}}," for key, point in composition]
    lines += ["};", "} // namespace tokspur::unicode9_data", ""]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines))
    print(
        f"Generated Unicode 9 data: {len(ranges)} combining ranges, {len(decomposition)} decompositions, {len(composition)} compositions"
    )


if __name__ == "__main__":
    main()
