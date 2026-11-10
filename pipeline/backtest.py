"""Walk-forward backtest: every match is predicted only from matches before it.

The model is refitted once per block (a week by default) and used for every
match in that block, which is how the live system behaves too.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .ratings import fit, outcome, score_matrix


def _metrics(P: np.ndarray, y: np.ndarray, base: np.ndarray) -> dict:
    eps = 1e-12
    onehot = np.eye(3)[y]
    ll = float(-np.mean(np.log(np.clip(P[np.arange(len(y)), y], eps, 1))))
    bll = float(-np.mean(np.log(np.clip(base[np.arange(len(y)), y], eps, 1))))
    brier = float(np.mean(np.sum((P - onehot) ** 2, axis=1)))
    bbrier = float(np.mean(np.sum((base - onehot) ** 2, axis=1)))
    acc = float(np.mean(P.argmax(1) == y))
    return {"log_loss": round(ll, 5), "brier": round(brier, 5), "accuracy": round(acc, 4),
            "baseline_log_loss": round(bll, 5), "baseline_brier": round(bbrier, 5),
            "skill_vs_baseline": round(1 - ll / bll, 4)}


def reliability(p: np.ndarray, hit: np.ndarray, bins: int = 10) -> list[dict]:
    out = []
    edges = np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if sel.sum() == 0:
            continue
        out.append({"bin_low": round(float(lo), 1), "bin_high": round(float(hi), 1),
                    "n": int(sel.sum()), "predicted": round(float(p[sel].mean()), 4),
                    "observed": round(float(hit[sel].mean()), 4)})
    return out


def calibration_error(P: np.ndarray, y: np.ndarray) -> float:
    """Expected calibration error pooled over home, draw and away."""
    p = P.ravel()
    hit = np.eye(3)[y].ravel()
    rel = reliability(p, hit)
    n = sum(r["n"] for r in rel)
    return round(sum(r["n"] * abs(r["predicted"] - r["observed"]) for r in rel) / n, 4)


def result_index(hg, ag) -> np.ndarray:
    return np.where(hg > ag, 0, np.where(hg == ag, 1, 2))


def run(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> tuple[dict, pd.DataFrame]:
    """Backtest matches with kickoff in [start, end). Returns summary and per-match predictions."""
    played = df.dropna(subset=["hg", "ag"])
    test = played[(played["kickoff"] >= start) & (played["kickoff"] < end)]
    rows = []
    t = start
    step = pd.Timedelta(days=C.BACKTEST_BLOCK_DAYS)
    while t < end:
        block = test[(test["kickoff"] >= t) & (test["kickoff"] < t + step)]
        if len(block):
            train = played[played["kickoff"] < t]
            model = fit(train, "hg", "ag", t, dixon_coles=True)
            if model is not None:
                recent = train[train["kickoff"] > t - pd.Timedelta(days=730)]
                ri = result_index(recent["hg"].to_numpy(), recent["ag"].to_numpy())
                base = np.bincount(ri, minlength=3) / max(len(ri), 1)
                for r in block.itertuples():
                    lh, la = model.rates(r.home, r.away)
                    ph, pd_, pa = outcome(score_matrix(lh, la, model.rho))
                    rows.append({"kickoff": r.kickoff, "home": r.home, "away": r.away,
                                 "hg": r.hg, "ag": r.ag, "p_home": ph, "p_draw": pd_,
                                 "p_away": pa, "b_home": base[0], "b_draw": base[1],
                                 "b_away": base[2]})
        t += step
    preds = pd.DataFrame(rows)
    if preds.empty:
        return {"n_matches": 0}, preds
    P = preds[["p_home", "p_draw", "p_away"]].to_numpy()
    B = preds[["b_home", "b_draw", "b_away"]].to_numpy()
    y = result_index(preds["hg"].to_numpy(), preds["ag"].to_numpy())
    summary = {"n_matches": int(len(preds)), **_metrics(P, y, B),
               "calibration_error": calibration_error(P, y),
               "reliability_home": reliability(P[:, 0], (y == 0).astype(float)),
               "from": preds["kickoff"].min().date().isoformat(),
               "to": preds["kickoff"].max().date().isoformat()}
    return summary, preds
