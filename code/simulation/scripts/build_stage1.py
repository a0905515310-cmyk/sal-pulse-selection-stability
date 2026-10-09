from __future__ import annotations

import json
from pathlib import Path

from sal_stability_stage1.build import ROOT, STAGE_DIR, build


def main() -> int:
    result = build()
    output = STAGE_DIR / "build_result.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
