"""Shared test fixtures.

All data here is SYNTHETIC, using fictional teams (Atlantis, Ruritania,
Freedonia, Sylvania, Zubrowka). It exists to exercise code paths; it is never
a substitute for the user-provided historical dataset.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from goalsignal.data.schemas import DataConfig

RESULTS_CSV = """\
date,home_team,away_team,home_score,away_score,tournament,city,country,neutral
2000-01-01,Atlantis,Ruritania,2,1,Friendly,Poseidonia,Atlantis,FALSE
2000-02-01,Freedonia,Sylvania,0,0,Friendly,Fredville,Freedonia,FALSE
2001-06-10,Atlantis,Freedonia,1,1,Mythic Cup,Neutralia,Zubrowka,TRUE
2001-06-14,Ruritania,Sylvania,3,00,Mythic Cup,Neutralia,Zubrowka,TRUE
2001-06-18,Sylvania,Atlantis,2,1,Mythic Cup,Neutralia,Zubrowka,TRUE
2002-03-01,Old Ruritania,Atlantis,1,0,Friendly,Strelsau,Ruritania,FALSE
2002-03-01,Old Ruritania,Atlantis,1,0,Friendly,Strelsau,Ruritania,FALSE
2003-05-05,Atlantis,Atlantis,1,0,Friendly,Poseidonia,Atlantis,FALSE
2003-07-07,Sylvania,Freedonia,bad,1,Friendly,Sylvia,Sylvania,FALSE
2030-01-01,Atlantis,Ruritania,NA,NA,Mythic Cup,Neutralia,Zubrowka,TRUE
"""

SHOOTOUTS_CSV = """\
date,home_team,away_team,winner,first_shooter
2001-06-10,Atlantis,Freedonia,Atlantis,Freedonia
2001-06-14,Ruritania,Sylvania,Ruritania,
1999-09-09,Ghostland,Phantomia,Ghostland,
2000-01-01,Atlantis,Ruritania,Zubrowka,
"""

GOALSCORERS_CSV = """\
date,home_team,away_team,team,scorer,minute,own_goal,penalty
2000-01-01,Atlantis,Ruritania,Atlantis,A. Triton,10,FALSE,FALSE
2000-01-01,Atlantis,Ruritania,Atlantis,B. Nereus,55,FALSE,TRUE
2000-01-01,Atlantis,Ruritania,Ruritania,R. Rassendyll,80,FALSE,FALSE
2001-06-10,Atlantis,Freedonia,Atlantis,A. Triton,12,FALSE,FALSE
"""

FORMER_NAMES_CSV = """\
current,former,start_date,end_date
Ruritania,Old Ruritania,1990-01-01,2002-12-31
"""


@pytest.fixture
def synthetic_dir(tmp_path: Path) -> Path:
    d = tmp_path / "synthetic_input"
    d.mkdir()
    (d / "results.csv").write_text(RESULTS_CSV, encoding="utf-8")
    (d / "shootouts.csv").write_text(SHOOTOUTS_CSV, encoding="utf-8")
    (d / "goalscorers.csv").write_text(GOALSCORERS_CSV, encoding="utf-8")
    (d / "former_names.csv").write_text(FORMER_NAMES_CSV, encoding="utf-8")
    return d


@pytest.fixture
def synthetic_config(synthetic_dir: Path, tmp_path: Path) -> DataConfig:
    cfg = DataConfig()
    cfg.input.directory = str(synthetic_dir)
    cfg.output.processed_dir = str(tmp_path / "processed")
    cfg.output.reports_dir = str(tmp_path / "reports")
    cfg.output.manifests_dir = str(tmp_path / "manifests")
    cfg.score_scope_policy.knockout_capable_tournament_patterns = ["Mythic Cup"]
    return cfg
