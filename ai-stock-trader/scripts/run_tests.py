"""Run the full test suite and write the readiness evidence file
reports/readiness/tests.json (consumed by the live-trading gate)."""
from __future__ import annotations

import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    out = ROOT / "reports" / "readiness"
    out.mkdir(parents=True, exist_ok=True)
    junit = out / "junit.xml"
    rc = subprocess.call([sys.executable, "-m", "pytest", "-q", f"--junitxml={junit}"], cwd=ROOT)
    suite = ET.parse(junit).getroot()
    suite = suite if suite.tag == "testsuite" else suite.find("testsuite")
    tests, fails, errs = int(suite.get("tests", 0)), int(suite.get("failures", 0)), int(suite.get("errors", 0))
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    rep = {"generated_at": datetime.now(timezone.utc).isoformat(), "passed": rc == 0 and fails + errs == 0,
           "tests": tests, "failures": fails + errs, "skipped": int(suite.get("skipped", 0)), "commit": commit}
    (out / "tests.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))
    return rc


if __name__ == "__main__":
    sys.exit(main())
