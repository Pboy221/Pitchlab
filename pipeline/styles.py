"""Team style profiles and situational shifts.

Eight dimensions from the fields the free data carries (shots, shots on
target, corners, fouls, cards). Each is a z-score against the league.

Situational shifts ask whether a side changes away from home, or against the
top third, MORE than every side does. League-wide swings (ordinary home
advantage, opponent quality) are removed first, then each team/dimension/
context is a Welch t-test, and the whole family is corrected with
Benjamini-Hochberg so chance findings do not get reported.
"""
from __future__ import annotations

import json
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

from . import config as C

DIMENSIONS = json.loads((Path(__file__).with_name("style_dimensions.json")).read_text())
DIM = {d["key"]: d for d in DIMENSIONS}

PHRASE = {
    ("shot_volume", 1): "shoots a lot", ("shot_volume", -1): "few shots",
    ("shot_quality", 1): "picks its shots", ("shot_quality", -1): "shoots from range",
    ("territory", 1): "dominates territory", ("territory", -1): "sits deep",
    ("defensive_exposure", 1): "open at the back", ("defensive_exposure", -1): "hard to break down",
    ("set_piece", 1): "wide and set-piece heavy", ("set_piece", -1): "plays through the middle",
    ("aggression", 1): "fouls a lot", ("aggression", -1): "lets the game flow",
    ("fouls_drawn", 1): "draws fouls", ("fouls_drawn", -1): "rarely fouled",
    ("discipline", 1): "booked easily", ("discipline", -1): "fouls cleanly",
}
MILD = {
    ("shot_volume", 1): "a few more shots", ("shot_volume", -1): "a few fewer shots",
    ("shot_quality", 1): "slightly cleaner chances", ("shot_quality", -1): "slightly more speculative",
    ("territory", 1): "slight territorial edge", ("territory", -1): "slightly deeper",
    ("defensive_exposure", 1): "a little open", ("defensive_exposure", -1): "a little tighter",
    ("set_piece", 1): "a few more corners", ("set_piece", -1): "a few fewer corners",
    ("aggression", 1): "fouls a bit more", ("aggression", -1): "fouls a bit less",
    ("fouls_drawn", 1): "fouled a bit more", ("fouls_drawn", -1): "fouled a bit less",
    ("discipline", 1): "booked a bit more", ("discipline", -1): "booked a bit less",
}


def team_id(league_id: int, name: str) -> int:
    return zlib.crc32(f"{league_id}:{name}".encode()) % 1_000_000_000


def team_rows(df: pd.DataFrame) -> pd.DataFrame:
    """One row per team per match, from that team's point of view."""
    need = ["hs", "as_", "hst", "ast", "hf", "af", "hc", "ac", "hy", "ay"]
    d = df.dropna(subset=need)
    cards_h = d["hy"] + d["hr"].fillna(0)
    cards_a = d["ay"] + d["ar"].fillna(0)
    home = pd.DataFrame({"kickoff": d["kickoff"], "team": d["home"], "opp": d["away"],
                         "away": False, "shots": d["hs"], "sot": d["hst"],
                         "shots_against": d["as_"], "corners": d["hc"],
                         "fouls": d["hf"], "fouls_drawn": d["af"], "cards": cards_h})
    away = pd.DataFrame({"kickoff": d["kickoff"], "team": d["away"], "opp": d["home"],
                         "away": True, "shots": d["as_"], "sot": d["ast"],
                         "shots_against": d["hs"], "corners": d["ac"],
                         "fouls": d["af"], "fouls_drawn": d["hf"], "cards": cards_a})
    r = pd.concat([home, away], ignore_index=True).sort_values("kickoff")
    # per-match dimension values
    r["shot_volume"] = r["shots"]
    r["shot_quality"] = np.where(r["shots"] > 0, 100 * r["sot"] / r["shots"].replace(0, np.nan), np.nan)
    r["territory"] = r["shots"] - r["shots_against"]
    r["defensive_exposure"] = r["shots_against"]
    r["set_piece"] = r["corners"]
    r["aggression"] = r["fouls"]
    r["discipline"] = np.where(r["fouls"] > 0, r["cards"] / r["fouls"].replace(0, np.nan), np.nan)
    return r


def _aggregate(g: pd.DataFrame) -> dict:
    return {
        "shot_volume": g["shots"].mean(),
        "shot_quality": 100 * g["sot"].sum() / max(g["shots"].sum(), 1),
        "territory": (g["shots"] - g["shots_against"]).mean(),
        "defensive_exposure": g["shots_against"].mean(),
        "set_piece": g["corners"].mean(),
        "aggression": g["fouls"].mean(),
        "fouls_drawn": g["fouls_drawn"].mean(),
        "discipline": g["cards"].sum() / max(g["fouls"].sum(), 1),
    }


def _descriptor(z: float) -> str:
    if z >= 1.5: return "very high"
    if z >= 0.75: return "high"
    if z <= -1.5: return "very low"
    if z <= -0.75: return "low"
    return "around average"


def _style_label(dims: list[dict]) -> str:
    ranked = sorted(dims, key=lambda d: -abs(d["z"]))
    strong = [PHRASE[(d["key"], 1 if d["z"] > 0 else -1)] for d in ranked if abs(d["z"]) >= 0.75][:2]
    if strong:
        return ", ".join(strong)
    top = ranked[0]
    if abs(top["z"]) >= 0.4:
        return "balanced, " + MILD[(top["key"], 1 if top["z"] > 0 else -1)]
    return "balanced"


def profiles(rows: pd.DataFrame, teams: list[str], league_id: int) -> dict:
    agg = {}
    n_matches = {}
    for t in teams:
        g = rows[rows["team"] == t].tail(C.STYLE_WINDOW)
        if len(g) >= 5:
            agg[t] = _aggregate(g)
            n_matches[t] = len(g)
    if len(agg) < 4:
        return {"dimensions": DIMENSIONS, "teams": {}, "window": C.STYLE_WINDOW}
    table = pd.DataFrame(agg).T
    z = (table - table.mean()) / table.std(ddof=0).replace(0, 1)
    pct = table.rank(pct=True)
    out = {}
    for t in table.index:
        dims = []
        for d in DIMENSIONS:
            k = d["key"]
            zz = float(z.loc[t, k])
            dims.append({"key": k, "label": d["label"], "unit": d["unit"],
                         "value": round(float(table.loc[t, k]), 3), "z": round(zz, 2),
                         "percentile": round(float(pct.loc[t, k]), 2),
                         "descriptor": _descriptor(zz), "meaning": d["meaning"],
                         "reading": d["high"] if zz >= 0.75 else d["low"] if zz <= -0.75 else ""})
        strong = [x for x in sorted(dims, key=lambda x: -abs(x["z"])) if abs(x["z"]) >= 0.75][:3]
        summary = f"{t}: " + ("; ".join(f"{x['label'].lower()} {x['descriptor']} "
                                         f"({x['value']:.2f} {x['unit']})" for x in strong)
                              if strong else "close to the league average on every measure")
        tid = team_id(league_id, t)
        out[str(tid)] = {"team_id": tid, "name": t, "matches": n_matches[t],
                         "style": _style_label(dims), "summary": summary + ".",
                         "dimensions": dims}
    return {"dimensions": DIMENSIONS, "teams": out, "window": C.STYLE_WINDOW}


def _bh(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted q-values."""
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(q, 0, 1)
    return out


CONTEXTS = {
    "venue": ("away", "at home", "away from home"),
    "opponent": ("vs_top", "against the rest", "against the top third"),
}


def situational(rows: pd.DataFrame, teams: list[str], strength: dict, league_id: int) -> dict:
    """Which teams change their game by fixture beyond the league-wide swing."""
    r = rows[rows["team"].isin(teams)].copy()
    r = r.groupby("team").tail(2 * C.STYLE_WINDOW).copy()
    ranked = sorted(teams, key=lambda t: -strength.get(t, 0))
    top = set(ranked[: max(1, len(ranked) // 3)])
    r["vs_top"] = r["opp"].isin(top)
    keys = [d["key"] for d in DIMENSIONS]

    # league-wide swing for the note
    terr_home = r.loc[~r["away"], "territory"].mean()
    terr_away = r.loc[r["away"], "territory"].mean()
    terr_rest = r.loc[~r["vs_top"], "territory"].mean()
    terr_top = r.loc[r["vs_top"], "territory"].mean()
    note = (f"League-wide, away from home the average side's shot difference falls by "
            f"{terr_home - terr_away:.1f} and against the top third it falls by "
            f"{terr_rest - terr_top:.1f}. That is ordinary home advantage and opponent "
            f"quality, already handled by the result model, so the shifts listed per team "
            f"are what each side does ON TOP of it.")

    tests = []
    for ctx, (col, off_label, on_label) in CONTEXTS.items():
        for k in keys:
            cond_mean = r.groupby(col)[k].transform("mean")
            r[f"res_{k}"] = r[k] - cond_mean
            for t in teams:
                g = r[r["team"] == t]
                on = g.loc[g[col], f"res_{k}"].dropna()
                off = g.loc[~g[col], f"res_{k}"].dropna()
                if len(on) < 6 or len(off) < 6:
                    continue
                p = ttest_ind(on, off, equal_var=False).pvalue
                if np.isnan(p):
                    continue
                tests.append({"team": t, "ctx": ctx, "key": k, "p": float(p),
                              "shift": float(on.mean() - off.mean()), "n": int(len(g))})
    if tests:
        q = _bh(np.array([x["p"] for x in tests]))
        for x, qq in zip(tests, q):
            x["q"] = float(qq)

    out = {}
    for t in teams:
        mine = [x for x in tests if x["team"] == t]
        kept = [x for x in mine if x["q"] <= C.FDR_KEEP]
        weak = [x for x in mine if x["q"] > C.FDR_KEEP and x["p"] < 0.01]
        contexts = {}
        for x in kept:
            contexts.setdefault(x["ctx"], []).append({
                "key": x["key"], "shift": round(x["shift"], 3), "q": round(x["q"], 4),
                "n": x["n"], "confidence": "confirmed" if x["q"] <= C.FDR_CONFIRM else "likely"})
        if kept:
            parts = [_sentence(t, x["ctx"], x["key"], x["shift"], x["n"], on=True) for x in kept[:3]]
            verdict = " ".join(parts)
        elif weak:
            verdict = (f"Weak evidence that {t} change their game with the fixture, but not "
                       f"enough to survive the correction for testing every team. Treat their "
                       f"season averages as the best guide.")
        else:
            verdict = (f"{t} swing with the fixture about as much as any side does. They get "
                       f"worse away and against better teams like everyone else, but nothing in "
                       f"their profile departs from the league norm, so their season averages "
                       f"should hold here.")
        tid = team_id(league_id, t)
        out[str(tid)] = {"team_id": tid, "name": t, "adapts": bool(kept),
                         "adapts_weakly": bool(weak and not kept), "contexts": contexts,
                         "verdict": verdict}
    return {"teams": out, "tests_run": len(tests),
            "tests_kept": sum(1 for x in tests if x["q"] <= C.FDR_KEEP),
            "tests_confirmed": sum(1 for x in tests if x["q"] <= C.FDR_CONFIRM),
            "fdr_q": C.FDR_KEEP, "baseline_note": note, "top_third": sorted(top),
            "window": 2 * C.STYLE_WINDOW}


def _sentence(team, ctx, key, shift, n, on: bool) -> str:
    """Plain-English shift. shift = (in context) - (out of context), league swing removed."""
    d = DIM[key]
    s = shift if on else -shift
    label = d["label"].lower()
    where = {("venue", True): "away from home", ("venue", False): "at home",
             ("opponent", True): "against the top third",
             ("opponent", False): "against the rest"}[(ctx, on)]
    direction = "higher" if s > 0 else "lower"
    return (f"{team} {where}: {label} runs {abs(s):.1f} {d['unit']} {direction} than "
            f"their usual, beyond the league-wide swing ({n} matches).")


def fixture_shifts(sit: dict, league_id: int, team: str, is_away: bool, opp_top: bool) -> list:
    t = sit["teams"].get(str(team_id(league_id, team)))
    if not t:
        return []
    out = []
    for x in t["contexts"].get("venue", []):
        out.append({"phrase": "away from home" if is_away else "at home",
                    "sentence": _sentence(team, "venue", x["key"], x["shift"], x["n"], on=is_away),
                    "n": x["n"], "confidence": x["confidence"]})
    if opp_top:
        for x in t["contexts"].get("opponent", []):
            out.append({"phrase": "against the top third",
                        "sentence": _sentence(team, "opponent", x["key"], x["shift"], x["n"], on=True),
                        "n": x["n"], "confidence": x["confidence"]})
    return out


def matchup(home_prof: dict | None, away_prof: dict | None, home: str, away: str) -> str:
    if not home_prof or not away_prof:
        return ""
    hz = {d["key"]: d["z"] for d in home_prof["dimensions"]}
    az = {d["key"]: d["z"] for d in away_prof["dimensions"]}
    notes = []
    gap = hz["territory"] - az["territory"]
    if abs(gap) >= 1.5:
        dom, other = (home, away) if gap > 0 else (away, home)
        notes.append(f"{dom} should spend most of this in the final third; {other} will "
                     f"defend for long spells.")
    for a, b, za, zb in ((home, away, hz, az), (away, home, az, hz)):
        if za["set_piece"] >= 0.75 and zb["defensive_exposure"] >= 0.75:
            notes.append(f"{a} win plenty of corners and {b} concede plenty of shots — the "
                         f"corner count leans {a}'s way.")
        if za["discipline"] >= 0.75 and zb["fouls_drawn"] >= 0.75:
            notes.append(f"{a} book easily and {b} draw fouls — the card markets lean toward "
                         f"{a}.")
    if hz["aggression"] >= 0.75 and az["aggression"] >= 0.75:
        notes.append("Two sides that foul a lot — expect a stop-start game and a busy referee.")
    if not notes:
        return ("No strong stylistic mismatch between these two. The strength model should "
                "carry this forecast rather than the style read.")
    return " ".join(notes[:2])
