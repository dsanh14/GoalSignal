"""Artifacts and read-only summaries for full World Cup simulations."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from goalsignal.tournament.bracket_2026 import OfficialBracket
from goalsignal.tournament.full_simulator import STAGES, FullSimulationResult
from goalsignal.utils.paths import resolve

ROUND_FILES = {
    "round_of_32": "wc2026_round_of_32_matchups.csv",
    "round_of_16": "wc2026_round_of_16_matchups.csv",
    "quarterfinal": "wc2026_quarterfinal_matchups.csv",
    "semifinal": "wc2026_semifinal_matchups.csv",
    "third_place": "wc2026_third_place_matchups.csv",
    "final": "wc2026_final_matchups.csv",
}


def advancement_frame(result: FullSimulationResult) -> pd.DataFrame:
    rows = []
    team_group = {team: group for group, teams in result.groups.items() for team in teams}
    for team in result.teams:
        probs = result.advancement_probs[team]
        row = {
            "group": team_group[team],
            "team": team,
            "expected_group_points": result.expected_points[team],
            **{
                f"p_finish_{i + 1}": result.position_probs[team][i]
                for i in range(4)
            },
            "p_best_third": result.best_third_probs[team],
            **{f"p_{stage}": probs[stage] for stage in STAGES},
            "p_finish_third": result.third_place_probs[team],
            "p_finish_fourth": result.fourth_place_probs[team],
        }
        for stage in STAGES:
            row[f"mc_se_{stage}"] = result.mc_standard_error(probs[stage])
        rows.append(row)
    return pd.DataFrame(rows).sort_values("p_champion", ascending=False)


def matchup_frame(
    result: FullSimulationResult, bracket: OfficialBracket, round_name: str
) -> pd.DataFrame:
    rows = []
    for number, slot in bracket.matches.items():
        if slot.round != round_name:
            continue
        for (home, away), count in result.matchup_counts[number].most_common():
            wins = result.winner_counts[number]
            home_wins = wins[(home, away, home)]
            rows.append({
                "match_number": number,
                "round": round_name,
                "date": slot.date,
                "time_et": slot.time_et,
                "host_city": slot.host_city,
                "slot_1_team": home,
                "slot_2_team": away,
                "matchup_probability": count / result.n_sims,
                "conditional_slot_1_win_probability": home_wins / count,
            })
    return pd.DataFrame(rows)


def _modal_bracket(result, bracket):
    matches = []
    for number in sorted(bracket.matches):
        pair, count = result.matchup_counts[number].most_common(1)[0]
        wins = result.winner_counts[number]
        winner = max(pair, key=lambda team: wins[(pair[0], pair[1], team)])
        matches.append({
            "match_number": number,
            "round": bracket.matches[number].round,
            "date": bracket.matches[number].date,
            "host_city": bracket.matches[number].host_city,
            "modal_matchup": list(pair),
            "matchup_probability": count / result.n_sims,
            "modal_conditional_winner": winner,
            "conditional_win_probability": wins[(pair[0], pair[1], winner)] / count,
        })
    return {
        "label": "modal probabilistic summary; no matchup is confirmed",
        "matches": matches,
    }


def write_full_simulation(
    result: FullSimulationResult,
    bracket: OfficialBracket,
    metadata: dict,
    version: str,
) -> Path:
    out = resolve(Path("artifacts/simulations") / version)
    out.mkdir(parents=True, exist_ok=True)
    meta_path = out / "wc2026_tournament_meta.json"
    if meta_path.exists():
        old = json.loads(meta_path.read_text(encoding="utf-8"))
        if old.get("result_store_hash") != metadata["result_store_hash"]:
            raise FileExistsError("stale simulation directory has a different result hash")
    advancement = advancement_frame(result)
    advancement.to_csv(out / "wc2026_team_advancement.csv", index=False)
    advancement[["team", "p_champion", "mc_se_champion"]].to_csv(
        out / "wc2026_champion_probabilities.csv", index=False
    )
    for round_name, filename in ROUND_FILES.items():
        matchup_frame(result, bracket, round_name).to_csv(out / filename, index=False)
    bracket_summary = _modal_bracket(result, bracket)
    (out / "wc2026_bracket.json").write_text(
        json.dumps(bracket_summary, indent=2) + "\n", encoding="utf-8"
    )
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return out


def write_ticket_advisory(
    result: FullSimulationResult,
    bracket: OfficialBracket,
    top_contenders: set[str],
) -> tuple[Path, Path]:
    rows = []
    for number in range(97, 105):
        slot = bracket.matches[number]
        counter = result.matchup_counts[number]
        top_five = counter.most_common(5)
        appearances = {}
        for pair, count in counter.items():
            for team in pair:
                appearances[team] = appearances.get(team, 0) + count
        modal_pair, modal_count = top_five[0]
        rows.append({
            "match_number": number,
            "round": slot.round,
            "date": slot.date,
            "time_et": slot.time_et,
            "host_city": slot.host_city,
            "most_likely_matchup": " vs ".join(modal_pair),
            "matchup_probability": modal_count / result.n_sims,
            "top_five_matchups": "; ".join(
                f"{a} vs {b} ({count / result.n_sims:.3%})"
                for (a, b), count in top_five
            ),
            "team_appearance_probabilities": "; ".join(
                f"{team} ({count / result.n_sims:.3%})"
                for team, count in sorted(
                    appearances.items(), key=lambda item: item[1], reverse=True
                )[:10]
            ),
            "p_at_least_one_top_contender": sum(
                count for pair, count in counter.items() if set(pair) & top_contenders
            ) / result.n_sims,
            "warning": "Speculative probabilities; matchup is not confirmed.",
        })
    frame = pd.DataFrame(rows)
    out = resolve("artifacts/reports")
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "wc2026_ticket_advisory.csv"
    md_path = out / "wc2026_ticket_advisory.md"
    frame.to_csv(csv_path, index=False)
    headers = list(frame.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    md_path.write_text(
        "# World Cup 2026 ticket advisory\n\n"
        "Probabilistic planning aid only. No speculative matchup is confirmed and "
        "this report makes no financial guarantee.\n\n" + "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    return csv_path, md_path
