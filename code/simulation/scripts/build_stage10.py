from __future__ import annotations

import json

from sal_stability_stage1.stage10 import build_stage10


if __name__ == "__main__":
    result = build_stage10()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
