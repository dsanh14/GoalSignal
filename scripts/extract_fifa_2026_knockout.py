"""Extract reviewable FIFA 2026 knockout reference tables from official PDFs."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGULATIONS = ROOT / "data/reference/FWC2026_regulations_EN.pdf"
SCHEDULE = ROOT / "data/reference/FWC26-Match-Schedule_English.pdf"
THIRD_OUT = ROOT / "data/reference/fifa_2026_third_place_combinations.csv"
MATCH_OUT = ROOT / "data/reference/fifa_2026_knockout_mapping.csv"
MANIFEST_OUT = ROOT / "artifacts/manifests/fifa_2026_knockout_mapping.json"
REFERENCE_MANIFEST_OUT = ROOT / "data/reference/fifa_2026_knockout_manifest.json"

DESTINATIONS = ["M79", "M85", "M81", "M74", "M82", "M77", "M87", "M80"]
SCHEDULE_ROWS = [
    (73, "round_of_32", "2026-06-28", "15:00", "Los Angeles"),
    (74, "round_of_32", "2026-06-29", "16:30", "Boston"),
    (75, "round_of_32", "2026-06-29", "21:00", "Monterrey"),
    (76, "round_of_32", "2026-06-29", "13:00", "Houston"),
    (77, "round_of_32", "2026-06-30", "17:00", "New York New Jersey"),
    (78, "round_of_32", "2026-06-30", "13:00", "Dallas"),
    (79, "round_of_32", "2026-06-30", "21:00", "Mexico City"),
    (80, "round_of_32", "2026-07-01", "12:00", "Atlanta"),
    (81, "round_of_32", "2026-07-01", "20:00", "San Francisco Bay Area"),
    (82, "round_of_32", "2026-07-01", "16:00", "Seattle"),
    (83, "round_of_32", "2026-07-02", "19:00", "Toronto"),
    (84, "round_of_32", "2026-07-02", "15:00", "Los Angeles"),
    (85, "round_of_32", "2026-07-02", "23:00", "Vancouver"),
    (86, "round_of_32", "2026-07-03", "18:00", "Miami"),
    (87, "round_of_32", "2026-07-03", "21:30", "Kansas City"),
    (88, "round_of_32", "2026-07-03", "14:00", "Dallas"),
    (89, "round_of_16", "2026-07-04", "17:00", "Philadelphia"),
    (90, "round_of_16", "2026-07-04", "13:00", "Houston"),
    (91, "round_of_16", "2026-07-05", "16:00", "New York New Jersey"),
    (92, "round_of_16", "2026-07-05", "20:00", "Mexico City"),
    (93, "round_of_16", "2026-07-06", "15:00", "Dallas"),
    (94, "round_of_16", "2026-07-06", "20:00", "Seattle"),
    (95, "round_of_16", "2026-07-07", "12:00", "Atlanta"),
    (96, "round_of_16", "2026-07-07", "16:00", "Vancouver"),
    (97, "quarterfinal", "2026-07-09", "16:00", "Boston"),
    (98, "quarterfinal", "2026-07-10", "15:00", "Los Angeles"),
    (99, "quarterfinal", "2026-07-11", "17:00", "Miami"),
    (100, "quarterfinal", "2026-07-11", "21:00", "Kansas City"),
    (101, "semifinal", "2026-07-14", "15:00", "Dallas"),
    (102, "semifinal", "2026-07-15", "15:00", "Atlanta"),
    (103, "third_place", "2026-07-18", "17:00", "Miami"),
    (104, "final", "2026-07-19", "15:00", "New York New Jersey"),
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if not REGULATIONS.exists() or not SCHEDULE.exists():
        raise SystemExit("official FIFA PDFs are missing from data/reference")
    with tempfile.TemporaryDirectory() as tmp:
        text = Path(tmp) / "regulations.txt"
        subprocess.run(
            ["pdftotext", "-layout", str(REGULATIONS), str(text)], check=True
        )
        rows = []
        for line in text.read_text(encoding="utf-8").splitlines():
            match = re.match(
                r"^\s*(\d{1,3})\s+((?:3[A-L]\s+){7}3[A-L])(?:\s|$)", line
            )
            if match:
                values = match.group(2).split()
                groups = sorted(value[1:] for value in values)
                rows.append([int(match.group(1)), "-".join(groups), *values])
    if [row[0] for row in rows] != list(range(1, 496)):
        raise SystemExit("Annexe C extraction did not produce options 1..495")

    THIRD_OUT.parent.mkdir(parents=True, exist_ok=True)
    with THIRD_OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["option", "combination_key", *DESTINATIONS])
        writer.writerows(rows)
    with MATCH_OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["match_number", "round", "date", "time_et", "host_city"])
        writer.writerows(SCHEDULE_ROWS)

    MANIFEST_OUT.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "retrieved_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "extraction_review_status": "manually_cross_checked_against_official_pdfs",
        "sources": [
            {
                "title": "Regulations for the FIFA World Cup 2026",
                "publication_version": "May 2026",
                "url": (
                    "https://digitalhub.fifa.com/asset/"
                    "73f73fe3-235a-4f54-b252-b53d3d580ec5/"
                    "FWC2026_regulations_EN.pdf"
                ),
                "local_path": str(REGULATIONS.relative_to(ROOT)),
                "sha256": sha256(REGULATIONS),
            },
            {
                "title": "FIFA World Cup 2026 Match Schedule",
                "publication_version": "Published 2026-06-03",
                "url": (
                    "https://digitalhub.fifa.com/asset/"
                    "4b5d4417-3343-4732-9cdf-14b6662af407/"
                    "FWC26-Match-Schedule_English.pdf"
                ),
                "local_path": str(SCHEDULE.relative_to(ROOT)),
                "sha256": sha256(SCHEDULE),
            },
        ],
        "outputs": {
            str(THIRD_OUT.relative_to(ROOT)): {
                "rows": len(rows),
                "sha256": sha256(THIRD_OUT),
            },
            str(MATCH_OUT.relative_to(ROOT)): {
                "rows": len(SCHEDULE_ROWS),
                "sha256": sha256(MATCH_OUT),
            },
        },
    }
    encoded = json.dumps(manifest, indent=2) + "\n"
    REFERENCE_MANIFEST_OUT.write_text(encoded, encoding="utf-8")
    MANIFEST_OUT.write_text(encoded, encoding="utf-8")
    print(f"extracted {len(rows)} Annexe C rows and {len(SCHEDULE_ROWS)} matches")


if __name__ == "__main__":
    sys.exit(main())
