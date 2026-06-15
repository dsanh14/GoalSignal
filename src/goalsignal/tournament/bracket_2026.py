"""Official, configuration-driven FIFA World Cup 2026 knockout bracket."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

from goalsignal.utils.hashing import sha256_file
from goalsignal.utils.paths import resolve

GROUPS = set("ABCDEFGHIJKL")
THIRD_MATCHES = {74, 77, 79, 80, 81, 82, 85, 87}
ROUND_MATCHES = {
    "round_of_32": list(range(73, 89)),
    "round_of_16": list(range(89, 97)),
    "quarterfinal": list(range(97, 101)),
    "semifinal": [101, 102],
    "third_place": [103],
    "final": [104],
}


@dataclass(frozen=True)
class MatchSlot:
    match_number: int
    round: str
    entrants: tuple[str, str]
    date: str
    time_et: str
    host_city: str


@dataclass
class OfficialBracket:
    matches: dict[int, MatchSlot]
    third_assignments: dict[str, dict[int, str]]
    source_manifest: dict
    config_hash: str
    table_hash: str

    @classmethod
    def load(cls, path: str | Path = "config/tournament_2026.yaml"):
        config_path = resolve(path)
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        schedule_path = resolve(cfg["match_mapping"])
        table_path = resolve(cfg["third_place_table"])
        manifest_path = resolve(cfg["source_manifest"])
        schedule = pd.read_csv(schedule_path).set_index("match_number")
        entrants = {
            **{int(k): tuple(v) for k, v in cfg["round_of_32"].items()},
            **{int(k): tuple(v) for k, v in cfg["advancement"].items()},
        }
        matches = {}
        for number, row in schedule.iterrows():
            matches[int(number)] = MatchSlot(
                int(number),
                str(row["round"]),
                entrants[int(number)],
                str(row["date"]),
                str(row["time_et"]),
                str(row["host_city"]),
            )
        table = pd.read_csv(table_path, dtype=str)
        assignments = {
            row.combination_key: {
                int(column[1:]): getattr(row, column)
                for column in table.columns[2:]
            }
            for row in table.itertuples(index=False)
        }
        obj = cls(
            matches,
            assignments,
            json.loads(manifest_path.read_text(encoding="utf-8")),
            sha256_file(config_path),
            sha256_file(table_path),
        )
        problems = obj.validate()
        if problems:
            raise ValueError("invalid official bracket: " + "; ".join(problems))
        return obj

    def validate(self) -> list[str]:
        problems = []
        if set(self.matches) != set(range(73, 105)):
            problems.append("matches must be exactly M73-M104")
        for round_name, numbers in ROUND_MATCHES.items():
            actual = [n for n, match in self.matches.items() if match.round == round_name]
            if sorted(actual) != numbers:
                problems.append(f"{round_name} match numbers are invalid")
        for number in range(73, 89):
            slots = self.matches[number].entrants
            for slot in slots:
                if slot == "THIRD":
                    if number not in THIRD_MATCHES:
                        problems.append(f"M{number} has an invalid THIRD slot")
                elif not re.fullmatch(r"[12][A-L]", slot):
                    problems.append(f"M{number} has invalid group slot {slot}")
        for number in range(89, 105):
            for slot in self.matches[number].entrants:
                if not re.fullmatch(r"[WL]\d{2,3}", slot):
                    problems.append(f"M{number} has invalid advancement slot {slot}")
                elif int(slot[1:]) >= number:
                    problems.append(f"M{number} depends on a non-prior match")
        if len(self.third_assignments) != math.comb(12, 8):
            problems.append("third-place table must contain 495 combinations")
        for key, mapping in self.third_assignments.items():
            groups = key.split("-")
            values = [value[1:] for value in mapping.values()]
            if len(groups) != 8 or set(groups) - GROUPS:
                problems.append(f"{key}: invalid combination")
            if set(mapping) != THIRD_MATCHES:
                problems.append(f"{key}: required third-place slots not filled")
            if sorted(values) != sorted(groups) or len(values) != len(set(values)):
                problems.append(f"{key}: assignments are not a permutation")
        if not self.source_manifest.get("sources"):
            problems.append("source manifest is empty")
        for source in self.source_manifest.get("sources", []):
            local = resolve(source["local_path"])
            if not local.exists() or sha256_file(local) != source["sha256"]:
                problems.append(f"source hash mismatch: {source.get('title')}")
        return problems

    def resolve_round_of_32(
        self, standings: dict[str, list[str]], best_third_groups: list[str]
    ) -> dict[int, tuple[str, str]]:
        key = "-".join(sorted(best_third_groups))
        if key not in self.third_assignments:
            raise KeyError(f"official third-place combination missing: {key}")
        third = self.third_assignments[key]
        matches = {}
        qualifiers = {
            *(standings[g][0] for g in sorted(GROUPS)),
            *(standings[g][1] for g in sorted(GROUPS)),
            *(standings[g][2] for g in best_third_groups),
        }
        for number in range(73, 89):
            resolved = []
            for slot in self.matches[number].entrants:
                if slot == "THIRD":
                    group = third[number][1:]
                    resolved.append(standings[group][2])
                else:
                    resolved.append(standings[slot[1:]][int(slot[0]) - 1])
            matches[number] = tuple(resolved)
        entrants = [team for pair in matches.values() for team in pair]
        if len(qualifiers) != 32 or len(entrants) != 32 or set(entrants) != qualifiers:
            raise ValueError("Round-of-32 resolution did not produce 32 unique qualifiers")
        return matches
