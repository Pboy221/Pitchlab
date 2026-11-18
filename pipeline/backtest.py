"""Walk-forward backtest: every match is predicted only from matches before it.

The model is refitted once per block (a week by default) and used for every
match in that block, which is how the live system behaves too.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C, odds
from .ratings import fit, outcome, score_matrix

ODDS_COLS = [c for book in odds.BENCHMARKS for c in odds.cols(book)]


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


def vs_bookmakers(preds: pd.DataFrame) -> dict:
    """Score the model against each bookmaker benchmark on the matches that have odds.

    Lower log loss is better. `model_minus_book` < 0 would mean the model beat the
    market; the interval says how sure we can be, given the number of matches.
    """
    out = {}
    y = result_index(preds["hg"].to_numpy(), preds["ag"].to_numpy())
    P = preds[["p_home", "p_draw", "p_away"]].to_numpy()
    for book, label in odds.BENCHMARKS.items():
        if not set(odds.cols(book)) <= set(preds.columns):
            continue
        O = preds[odds.cols(book)].to_numpy(float)
        ok = odds.valid(O)
        if ok.sum() < 50:
            continue
        B = odds.implied(O[ok])
        diff = odds.log_losses(P[ok], y[ok]) - odds.log_losses(B, y[ok])
        lo, hi = odds.bootstrap_ci(diff)
        out[book] = {"label": label, "n_matches": int(ok.sum()),
                     "book_log_loss": round(float(odds.log_losses(B, y[ok]).mean()), 5),
                     "model_log_loss": round(float(odds.log_losses(P[ok], y[ok]).mean()), 5),
                     "model_minus_book": round(float(diff.mean()), 5),
                     "model_minus_book_ci95": [round(lo, 5), round(hi, 5)],
                     "book_brier": round(float(np.mean(np.sum((B - np.eye(3)[y[ok]]) ** 2, 1))), 5),
                     "book_accuracy": round(float(np.mean(B.argmax(1) == y[ok])), 4),
                     "margin": round(float(odds.overround(O[ok]).mean()), 4)}
    return out


def betting(preds: pd.DataFrame, book: str = odds.BET_AT) -> list[dict]:
    """Flat-stake value-bet simulation at several edge thresholds (no per-bet detail)."""
    if not set(odds.cols(book)) <= set(preds.columns):
        return []
    O = preds[odds.cols(book)].to_numpy(float)
    ok = odds.valid(O)
    if ok.sum() < 50:
        return []
    P = preds[["p_home", "p_draw", "p_away"]].to_numpy()[ok]
    y = result_index(preds["hg"].to_numpy(), preds["ag"].to_numpy())[ok]
    res = []
    for e in odds.EDGES:
        r = odds.value_bets(P, O[ok], y, e)
        r.pop("rows"), r.pop("pnl")
        res.append({k: (round(v, 4) if isinstance(v, float) else
                        [round(x, 4) for x in v] if isinstance(v, list) else v)
                    for k, v in r.items()})
    return res


def run(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp,
        train_days: int | None = None) -> tuple[dict, pd.DataFrame]:
    """Backtest matches with kickoff in [start, end). Returns summary and per-match predictions.

    train_days limits each refit to that many days of history (the live site uses
    about three seasons); None uses everything before the block.
    """
    played = df.dropna(subset=["hg", "ag"])
    odds_cols = [c for c in ODDS_COLS if c in played.columns]
    test = played[(played["kickoff"] >= start) & (played["kickoff"] < end)]
    rows = []
    t = start
    step = pd.Timedelta(days=C.BACKTEST_BLOCK_DAYS)
    while t < end:
        block = test[(test["kickoff"] >= t) & (test["kickoff"] < t + step)]
        if len(block):
            train = played[played["kickoff"] < t]
            if train_days:
                train = train[train["kickoff"] >= t - pd.Timedelta(days=train_days)]
            model = fit(train, "hg", "ag", t, dixon_coles=True)
            if model is not None:
                recent = train[train["kickoff"] > t - pd.Timedelta(days=730)]
                ri = result_index(recent["hg"].to_numpy(), recent["ag"].to_numpy())
                base = np.bincount(ri, minlength=3) / max(len(ri), 1)
                for r in block.itertuples():
                    lh, la = model.rates(r.home, r.away)
                    ph, pd_, pa = outcome(score_matrix(lh, la, model.rho))
                    row = {"kickoff": r.kickoff, "home": r.home, "away": r.away,
                           "hg": r.hg, "ag": r.ag, "p_home": ph, "p_draw": pd_,
                           "p_away": pa, "b_home": base[0], "b_draw": base[1],
                           "b_away": base[2]}
                    row.update({c: getattr(r, c) for c in odds_cols})
                    rows.append(row)
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
    vb = vs_bookmakers(preds)
    if vb:
        summary["vs_bookmakers"] = vb
        summary["betting"] = betting(preds)
    return summary, preds
