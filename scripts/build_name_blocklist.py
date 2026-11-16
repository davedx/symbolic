"""Build world/data/name_blocklist.txt: dictionary words the name grammar can produce.

Run once (output is committed):  uv run python scripts/build_name_blocklist.py [/usr/share/dict/words]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from world.names import BLOCKLIST_PATH, NAME_RE  # noqa: E402


def main() -> None:
    dict_path = Path(sys.argv[1] if len(sys.argv) > 1 else "/usr/share/dict/words")
    words = {w.strip().lower() for w in dict_path.read_text().split()}
    hits = sorted(w for w in words if NAME_RE.match(w))
    BLOCKLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    BLOCKLIST_PATH.write_text("\n".join(hits) + "\n")
    print(f"{len(hits)} blocklisted words written to {BLOCKLIST_PATH}")


if __name__ == "__main__":
    main()
