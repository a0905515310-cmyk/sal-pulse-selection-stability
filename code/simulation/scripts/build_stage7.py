from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sal_stability_stage1.stage7 import build_stage7


if __name__ == "__main__":
    print(json.dumps(build_stage7(), indent=2, sort_keys=True))
