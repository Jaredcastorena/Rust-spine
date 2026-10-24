#!/usr/bin/env python3
"""Verify the exact source inventory of the documented Candle backport."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent.parent
vendor = root / "vendor/candle-core"
manifest = json.loads((root / "vendor/candle-core-source.json").read_text())
actual = {
    path.relative_to(vendor).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
    for path in vendor.rglob("*")
    if path.is_file()
}
if actual != manifest["files"]:
    changed = sorted(
        name for name in actual.keys() | manifest["files"].keys()
        if actual.get(name) != manifest["files"].get(name)
    )
    raise SystemExit("Vendored Candle differs from reviewed source: " + ", ".join(changed))
print("Vendored Candle source inventory verified")
