"""Bookmaker odds: turn prices into probabilities and score the model against them.

The question this answers: does PitchLab know anything the betting market doesn't?

Decimal odds of 2.50 mean "stake 1, get 2.50 back if it wins", so the price implies
a probability of 1 / 2.50 = 40%. Add up the three outcomes of a match and you get a
bit more than 100% (usually 102-106%). That surplus, the *overround*, is the
bookmaker's margin. To compare like with like we strip it out by scaling the three
implied probabilities so they sum to exactly 1 ("proportional" de-vigging, the
simplest standard method).

Benchmarks, sharpest first (all from football-data.co.uk):
    psc   Pinnacle closing odds: the price at kickoff from the sharpest bookmaker.
          The hardest benchmark there is; beating it is rare.
    avgc  average closing odds across ~40 bookmakers.
    avg   average pre-match odds, collected a day or two before kickoff (Friday
          for weekend games, Tuesday for midweek). This is roughly the price you
          could actually have bet at when the model made its forecast.
"""
from __future__ import annotations

import numpy as np

BENCHMARKS = {"psc": "Pinnacle closing", "avgc": "Market average closing",
              "avg": "Market average pre-match"}
BET_AT = "avg"          # prices used for the betting simulation (see module docstring)
EDGES = (0.02, 0.05, 0.10)
MAIN_EDGE = 0.05        # chosen before looking at results; the others are a sensitivity check


def cols(book: str) -> list[str]:
    return [f"odds_{book}_h", f"odds_{book}_d", f"odds_{book}_a"]


def implied(odds: np.ndarray) -> np.ndarray:
    """(n, 3) decimal odds -> (n, 3) probabilities with the margin removed."""
    inv = 1.0 / odds
    return inv / inv.sum(axis=1, keepdims=True)


def overround(odds: np.ndarray) -> np.ndarray:
    """Bookmaker margin per match, e.g. 0.03 for a 103% book."""
    return (1.0 / odds).sum(axis=1) - 1.0


def valid(odds: np.ndarray) -> np.ndarray:
    """Rows with three usable prices (all present and above 1.0)."""
    return np.isfinite(odds).all(axis=1) & (odds > 1.0).all(axis=1)


def log_losses(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-match log loss: -log(probability given to what actually happened)."""
    return -np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1.0))


def bootstrap_ci(x: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% interval for the mean of x by resampling matches with replacement."""
    if len(x) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(n, len(x)))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def value_bets(P: np.ndarray, odds: np.ndarray, y: np.ndarray, edge: float) -> dict:
    """Flat-stake simulation: back the single best outcome per match when the model
    thinks it is underpriced by more than `edge`.

    Expected value of a 1-unit bet = model probability x odds - 1. A 5% edge means
    the model expects 5p profit per pound staked. Each bet returns odds - 1 if it
    wins and -1 if it loses.
    """
    ev = P * odds - 1.0
    best = ev.argmax(axis=1)
    take = ev[np.arange(len(y)), best] > edge
    rows = np.flatnonzero(take)
    won = best[rows] == y[rows]
    pnl = np.where(won, odds[rows, best[rows]] - 1.0, -1.0)
    n = len(rows)
    lo, hi = bootstrap_ci(pnl) if n else (float("nan"), float("nan"))
    return {"edge": edge, "n_bets": int(n), "hit_rate": float(won.mean()) if n else None,
            "avg_odds": float(odds[rows, best[rows]].mean()) if n else None,
            "profit": float(pnl.sum()), "roi": float(pnl.mean()) if n else None,
            "roi_ci95": [lo, hi] if n else None, "rows": rows, "pnl": pnl}
