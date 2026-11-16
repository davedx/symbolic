"""Pronounceable nonsense names.

Names are built from a small syllable grammar (onset + vowel + optional coda),
2-3 syllables each. Any string that the grammar can produce *and* that is an
English dictionary word is excluded via a bundled blocklist
(``world/data/name_blocklist.txt``, built by ``scripts/build_name_blocklist.py``).
The blocklist is checked into the repo so name generation is deterministic and
identical on every machine.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import numpy as np

ONSETS = [
    "b", "d", "f", "g", "k", "l", "m", "n", "p", "r", "s", "t", "v", "z",
    "br", "dr", "fl", "gl", "kr", "pl", "sk", "st", "tr", "zh",
]
VOWELS = ["a", "e", "i", "o", "u"]
CODAS = ["", "", "", "n", "r", "l", "s", "k", "m"]  # "" repeated => open syllables more likely

SYLLABLE_RE = "(?:{})(?:{})(?:{})".format(
    "|".join(sorted(ONSETS, key=len, reverse=True)),
    "|".join(VOWELS),
    "|".join(sorted({c for c in CODAS if c}, key=len, reverse=True)) + "|",
)
NAME_RE = re.compile(rf"^(?:{SYLLABLE_RE}){{2,3}}$")

BLOCKLIST_PATH = Path(__file__).parent / "data" / "name_blocklist.txt"


@lru_cache(maxsize=1)
def blocklist() -> frozenset[str]:
    if not BLOCKLIST_PATH.exists():
        return frozenset()
    return frozenset(w.strip() for w in BLOCKLIST_PATH.read_text().split() if w.strip())


def _syllable(rng: np.random.Generator) -> str:
    return (
        ONSETS[rng.integers(len(ONSETS))]
        + VOWELS[rng.integers(len(VOWELS))]
        + CODAS[rng.integers(len(CODAS))]
    )


def generate_names(n: int, rng: np.random.Generator, exclude: set[str] | None = None) -> list[str]:
    """Generate ``n`` unique nonsense names, none in ``exclude`` or the blocklist."""
    taken = set(exclude or ())
    block = blocklist()
    out: list[str] = []
    attempts = 0
    while len(out) < n:
        attempts += 1
        if attempts > 1000 * (n + 10):
            raise RuntimeError("name space exhausted")
        k = 2 if rng.random() < 0.7 else 3
        name = "".join(_syllable(rng) for _ in range(k))
        if name in taken or name in block:
            continue
        taken.add(name)
        out.append(name)
    return out
