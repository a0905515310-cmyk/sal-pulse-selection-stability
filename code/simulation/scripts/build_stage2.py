from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sal_stability_stage1.stage2 import build_stage2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and validate only the Stage 2 counter-based random layer."
    )
    parser.add_argument(
        "--generate-golden-vectors",
        action="store_true",
        help=(
            "Explicit one-time creation mode. Refuses to overwrite an existing "
            "rng_test_vectors.csv. The default mode is verification."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    status = build_stage2(generate_golden=args.generate_golden_vectors)
    print(json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if status["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
