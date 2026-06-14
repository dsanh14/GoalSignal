"""Knockout-match resolution.

A knockout tie is resolved in stages, each preserved separately: regulation
score (sampled from the goal model's score matrix), extra time (independent
Poisson at one third of the regulation intensities, reflecting 30 of 90
minutes), and a penalty shootout. The shootout baseline is 50/50 — shootout
goals are never added to match goals, and the regulation outcome is recorded
independently of who advances.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class KnockoutOutcome:
    reg_home_goals: int
    reg_away_goals: int
    extra_time: bool
    shootout: bool
    winner: str  # team name that advances


def sample_score(matrix: np.ndarray, rng: np.random.Generator) -> tuple[int, int]:
    flat = rng.choice(matrix.size, p=matrix.ravel() / matrix.sum())
    return int(flat // matrix.shape[1]), int(flat % matrix.shape[1])


def resolve_knockout(
    home: str,
    away: str,
    matrix: np.ndarray,
    lam_home: float,
    lam_away: float,
    rng: np.random.Generator,
    shootout_home_prob: float = 0.5,
    extra_time_factor: float = 1.0 / 3.0,
) -> KnockoutOutcome:
    hg, ag = sample_score(matrix, rng)
    if hg != ag:
        return KnockoutOutcome(hg, ag, False, False, home if hg > ag else away)
    et_h = int(rng.poisson(lam_home * extra_time_factor))
    et_a = int(rng.poisson(lam_away * extra_time_factor))
    if et_h != et_a:
        return KnockoutOutcome(hg, ag, True, False, home if et_h > et_a else away)
    winner = home if rng.random() < shootout_home_prob else away
    return KnockoutOutcome(hg, ag, True, True, winner)
