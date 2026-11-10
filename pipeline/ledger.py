"""The live ledger: every forecast is written before kickoff and scored after.

Each run overwrites a fixture's entry with its latest forecast for as long as
kickoff is still in the future, then freezes it. Once the result is in the
data the entry is settled. The file is committed back to the repository, so
the record builds up run after run and cannot be quietly edited after the fact
without it showing in the git history.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import calibration_error


def key(league_id: int, season: str, home: str, away: str) -> str:
    return f"{league_id}|{season}|{home}|{away}"


def load(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {"entries": {}}


def save(path: Path, led: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(led, indent=1, sort_keys=True))


def record(led: dict, k: str, fixture: dict, now: pd.Timestamp, model_version: str) -> None:
    kickoff = pd.Timestamp(fixture["kickoff"])
    e = led["entries"].get(k)
    if e and (e.get("result") or pd.Timestamp(e["kickoff"]) <= now):
        return                                  # frozen
    if kickoff <= now:
        return                                  # never log after kickoff
    led["entries"][k] = {
        "league_id": fixture["league_id"], "home": fixture["home"], "away": fixture["away"],
        "kickoff": fixture["kickoff"], "logged_at": now.isoformat(),
        "model_version": model_version, "outcome": fixture["outcome"], "result": None,
    }


def settle(led: dict, results: pd.DataFrame, season_of) -> int:
    """Attach full-time scores. `results` has league_id, home, away, hg, ag, kickoff."""
    done = 0
    open_ = {k: e for k, e in led["entries"].items() if not e.get("result")}
    if not open_:
        return 0
    idx = {}
    for r in results.dropna(subset=["hg", "ag"]).itertuples():
        idx[key(r.league_id, season_of(r.kickoff), r.home, r.away)] = (int(r.hg), int(r.ag))
    for k, e in open_.items():
        if k in idx:
            e["result"] = {"home": idx[k][0], "away": idx[k][1]}
            done += 1
    return done


def summary(led: dict, league_id: int) -> dict:
    rows = [e for e in led["entries"].values()
            if e["league_id"] == league_id and e.get("result")]
    if not rows:
        return {"n": 0, "note": "No settled predictions yet."}
    P = np.array([[e["outcome"]["home"], e["outcome"]["draw"], e["outcome"]["away"]] for e in rows])
    y = np.array([0 if e["result"]["home"] > e["result"]["away"]
                  else 1 if e["result"]["home"] == e["result"]["away"] else 2 for e in rows])
    ll = float(-np.mean(np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1))))
    out = {"n": len(rows), "log_loss": round(ll, 3),
           "accuracy": round(float(np.mean(P.argmax(1) == y)), 3)}
    if len(rows) >= 50:              # too noisy to mean anything below this
        out["calibration_error"] = round(calibration_error(P, y), 3)
    return out
