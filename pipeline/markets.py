"""Turn fitted models into the per-fixture numbers the page shows."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import nbinom, poisson

from . import config as C
from .ratings import TeamModel, fit, outcome, score_matrix


@dataclass
class LeagueModels:
    goals: TeamModel
    corners: TeamModel | None
    cards: TeamModel | None
    corner_phi: float
    card_phi: float
    referees: dict          # name -> (multiplier, matches)


def _dispersion(df, model: TeamModel, hcol, acol, as_of) -> float:
    d = df.dropna(subset=[hcol, acol])
    d = d[d["kickoff"] > as_of - pd.Timedelta(days=730)]
    if d.empty or model is None:
        return 1.0
    mu = np.array([sum(model.rates(h, a)) for h, a in zip(d["home"], d["away"])])
    y = (d[hcol] + d[acol]).to_numpy(float)
    return float(max(1.0, np.sum((y - mu) ** 2) / np.sum(mu)))


def _referees(df, cards: TeamModel | None, as_of) -> dict:
    if cards is None or "referee" not in df:
        return {}
    d = df.dropna(subset=["referee", "hy", "ay"])
    d = d[d["kickoff"] > as_of - pd.Timedelta(days=1100)]
    if d.empty:
        return {}
    exp = np.array([sum(cards.rates(h, a)) for h, a in zip(d["home"], d["away"])])
    obs = (d["hy"] + d["ay"] + d["hr"].fillna(0) + d["ar"].fillna(0)).to_numpy(float)
    k = 3.0 * exp.mean()                  # three matches' worth of shrinkage
    out = {}
    for ref, idx in d.groupby("referee").indices.items():
        out[ref] = ((obs[idx].sum() + k) / (exp[idx].sum() + k), len(idx))
    return out


def fit_league(df: pd.DataFrame, as_of: pd.Timestamp) -> LeagueModels | None:
    df = df[df["kickoff"] < as_of]
    goals = fit(df, "hg", "ag", as_of, dixon_coles=True)
    if goals is None:
        return None
    corners = fit(df, "hc", "ac", as_of)
    cd = df.assign(hcards=df["hy"] + df["hr"].fillna(0), acards=df["ay"] + df["ar"].fillna(0))
    cards = fit(cd, "hcards", "acards", as_of)
    return LeagueModels(
        goals=goals, corners=corners, cards=cards,
        corner_phi=_dispersion(df, corners, "hc", "ac", as_of) if corners else 1.0,
        card_phi=_dispersion(cd, cards, "hcards", "acards", as_of) if cards else 1.0,
        referees=_referees(df, cards, as_of),
    )


def _over(line: float, mean: float, phi: float) -> float:
    k = math.floor(line)
    if phi <= 1.02:
        return float(poisson.sf(k, mean))
    n = mean / (phi - 1)
    return float(nbinom.sf(k, n, n / (n + mean)))


def predict(lm: LeagueModels, home: str, away: str, referee: str | None) -> dict:
    lh, la = lm.goals.rates(home, away)
    m = score_matrix(lh, la, lm.goals.rho)
    ph, pd_, pa = outcome(m)
    g = np.arange(m.shape[0])
    over25 = float(m[(g[:, None] + g[None, :]) > 2.5].sum())
    btts = float(m[1:, 1:].sum())
    flat = [(int(i), int(j), float(m[i, j])) for i in range(7) for j in range(7)]
    flat.sort(key=lambda x: -x[2])
    out = {
        "outcome": {"home": round(ph, 4), "draw": round(pd_, 4), "away": round(pa, 4)},
        "expected_goals": {"home": round(lh, 2), "away": round(la, 2)},
        "btts": round(btts, 4),
        "totals": {"over": round(over25, 4), "under": round(1 - over25, 4)},
        "scorelines": [{"home": i, "away": j, "p": round(p, 4)} for i, j, p in flat[:6]],
    }
    if lm.corners:
        ch, ca = lm.corners.rates(home, away)
        tot = ch + ca
        line = math.floor(tot) + 0.5 if tot - math.floor(tot) >= 0.5 else math.floor(tot) - 0.5
        line = max(line, 6.5)
        ov = _over(line, tot, lm.corner_phi)
        out["corners"] = {"line": line, "over": round(ov, 4), "under": round(1 - ov, 4),
                          "expected_home": round(ch, 2), "expected_away": round(ca, 2),
                          "expected_total": round(tot, 2)}
    else:
        out["corners"] = None
    if lm.cards:
        kh, ka = lm.cards.rates(home, away)
        mult, n = lm.referees.get(referee, (1.0, 0)) if referee else (1.0, 0)
        kh, ka = kh * mult, ka * mult
        tot = kh + ka
        out["team_cards"] = {
            "expected_home": round(kh, 2), "expected_away": round(ka, 2),
            "expected_total": round(tot, 2), "referee_multiplier": round(mult, 3),
            "referee_matches": int(n),
            "over_3_5": round(_over(3.5, tot, lm.card_phi), 4),
            "over_4_5": round(_over(4.5, tot, lm.card_phi), 4),
            "over_5_5": round(_over(5.5, tot, lm.card_phi), 4),
        }
    else:
        out["team_cards"] = None
    return out
