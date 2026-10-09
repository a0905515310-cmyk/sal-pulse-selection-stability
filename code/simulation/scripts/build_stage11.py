from __future__ import annotations

import json

from sal_stability_stage1.stage11 import build_stage11


if __name__ == "__main__":
    result = build_stage11()
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
