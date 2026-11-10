"""Team strength models.

One generic fit serves goals, corners and cards: a time-weighted Poisson
regression where

    log E[home count] = mu + home_adv + att[home] - def[away]
    log E[away count] = mu            + att[away] - def[home]

with a ridge penalty that pulls every team toward average (this is what keeps
promoted sides with a handful of matches from getting wild ratings). For goals,
a Dixon-Coles rho corrects the Poisson's known under-count of 0-0 and 1-1.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import poisson

from . import config as C


@dataclass
class TeamModel:
    teams: list[str]
    mu: float
    home: float
    att: np.ndarray
    dfn: np.ndarray
    rho: float = 0.0
    idx: dict = field(default_factory=dict)

    def __post_init__(self):
        self.idx = {t: i for i, t in enumerate(self.teams)}
        # unseen teams (just promoted, no matches yet) get a below-average prior
        self._att_new = float(np.quantile(self.att, 0.2)) if len(self.att) else 0.0
        self._def_new = float(np.quantile(self.dfn, 0.2)) if len(self.dfn) else 0.0

    def _ad(self, team):
        i = self.idx.get(team)
        return (self.att[i], self.dfn[i]) if i is not None else (self._att_new, self._def_new)

    def rates(self, home: str, away: str) -> tuple[float, float]:
        ah, dh = self._ad(home)
        aa, da = self._ad(away)
        return (float(np.exp(self.mu + self.home + ah - da)),
                float(np.exp(self.mu + aa - dh)))

    def strength(self, team: str) -> float:
        a, d = self._ad(team)
        return a + d


def _weights(kickoffs: pd.Series, as_of: pd.Timestamp) -> np.ndarray:
    days = (as_of - kickoffs).dt.total_seconds().to_numpy() / 86400.0
    return np.exp(-C.DECAY_PER_DAY * np.clip(days, 0, None))


def fit(df: pd.DataFrame, hcol: str, acol: str, as_of: pd.Timestamp,
        ridge: float = C.RIDGE, dixon_coles: bool = False) -> TeamModel | None:
    d = df.dropna(subset=[hcol, acol])
    if len(d) < 30:
        return None
    teams = sorted(set(d["home"]) | set(d["away"]))
    ix = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    hi = d["home"].map(ix).to_numpy()
    ai = d["away"].map(ix).to_numpy()
    yh = d[hcol].to_numpy(float)
    ya = d[acol].to_numpy(float)
    w = _weights(d["kickoff"], as_of)
    lam = ridge * w.sum() / 380.0     # penalty scales with the effective sample

    def nll(p):
        mu, ha = p[0], p[1]
        att, dfn = p[2:2 + n], p[2 + n:]
        eh = mu + ha + att[hi] - dfn[ai]
        ea = mu + att[ai] - dfn[hi]
        lh, la = np.exp(eh), np.exp(ea)
        f = np.sum(w * (lh - yh * eh)) + np.sum(w * (la - ya * ea))
        f += lam * (att @ att + dfn @ dfn)
        rh, ra = w * (lh - yh), w * (la - ya)
        g = np.empty_like(p)
        g[0] = rh.sum() + ra.sum()
        g[1] = rh.sum()
        g[2:2 + n] = (np.bincount(hi, rh, n) + np.bincount(ai, ra, n)) + 2 * lam * att
        g[2 + n:] = -(np.bincount(ai, rh, n) + np.bincount(hi, ra, n)) + 2 * lam * dfn
        return f, g

    mean = (np.average(yh, weights=w) + np.average(ya, weights=w)) / 2
    p0 = np.zeros(2 + 2 * n)
    p0[0] = np.log(max(mean, 0.05))
    res = minimize(nll, p0, jac=True, method="L-BFGS-B")
    p = res.x
    model = TeamModel(teams, float(p[0]), float(p[1]), p[2:2 + n], p[2 + n:])

    if dixon_coles:
        lh = np.exp(model.mu + model.home + model.att[hi] - model.dfn[ai])
        la = np.exp(model.mu + model.att[ai] - model.dfn[hi])
        low = (yh <= 1) & (ya <= 1)

        def neg(rho):
            t = np.ones(len(yh))
            m00 = low & (yh == 0) & (ya == 0)
            m01 = low & (yh == 0) & (ya == 1)
            m10 = low & (yh == 1) & (ya == 0)
            m11 = low & (yh == 1) & (ya == 1)
            t[m00] = 1 - lh[m00] * la[m00] * rho
            t[m01] = 1 + lh[m01] * rho
            t[m10] = 1 + la[m10] * rho
            t[m11] = 1 - rho
            if np.any(t <= 0):
                return 1e12
            return -np.sum(w * np.log(t))

        model.rho = float(minimize_scalar(neg, bounds=(-0.2, 0.2), method="bounded").x)
    return model


def score_matrix(lh: float, la: float, rho: float = 0.0, k: int = C.MAX_GOALS) -> np.ndarray:
    g = np.arange(k + 1)
    m = np.outer(poisson.pmf(g, lh), poisson.pmf(g, la))
    if rho:
        m[0, 0] *= 1 - lh * la * rho
        m[0, 1] *= 1 + lh * rho
        m[1, 0] *= 1 + la * rho
        m[1, 1] *= 1 - rho
    m = np.clip(m, 0, None)
    return m / m.sum()


def outcome(m: np.ndarray) -> tuple[float, float, float]:
    return (float(np.tril(m, -1).sum()), float(np.trace(m)), float(np.triu(m, 1).sum()))
