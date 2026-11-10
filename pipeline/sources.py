"""Download and normalise every data source into one shape.

Matches come out as a DataFrame with one row per match:
    league_id, season, kickoff (UTC), home, away, hg, ag,
    hs, as_, hst, ast, hf, af, hc, ac, hy, ay, hr, ar, referee
Stat columns are NaN where a source does not carry them.
"""
from __future__ import annotations

import io
import json
import logging
import time
from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

from . import config as C
from .names import ALIASES, build_map

log = logging.getLogger("pitchlab.sources")
HEADERS = {"User-Agent": "PitchLab/2.0 (+https://github.com) statistical model"}

STAT_COLS = {"HS": "hs", "AS": "as_", "HST": "hst", "AST": "ast", "HF": "hf",
             "AF": "af", "HC": "hc", "AC": "ac", "HY": "hy", "AY": "ay",
             "HR": "hr", "AR": "ar"}


# ------------------------------------------------------------------ helpers
def get(url: str, tries: int = 3, timeout: int = 30) -> bytes | None:
    for i in range(tries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=timeout)
            if r.status_code == 200 and r.content:
                return r.content
            if r.status_code == 404:
                return None
            log.warning("GET %s -> %s", url, r.status_code)
        except requests.RequestException as e:
            log.warning("GET %s failed: %s", url, e)
        time.sleep(2 * (i + 1))
    return None


def season_code(start_year: int) -> str:
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def season_start(d: date) -> int:
    return d.year if d.month >= 7 else d.year - 1


def _read_csv(raw: bytes) -> pd.DataFrame:
    for enc in ("utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc, on_bad_lines="skip")
        except UnicodeDecodeError:
            continue
    return pd.DataFrame()


def _parse_kickoff(df: pd.DataFrame, tz: str) -> pd.Series:
    """Combine Date (+ optional Time) into UTC timestamps."""
    d = df["Date"].astype(str).str.strip()
    iso = d.str.match(r"^\d{4}-\d{2}-\d{2}")
    day = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    day[iso] = pd.to_datetime(d[iso], format="%Y-%m-%d", errors="coerce")
    day[~iso] = pd.to_datetime(d[~iso], dayfirst=True, errors="coerce")
    if "Time" in df.columns:
        t = df["Time"].astype(str).str.strip().where(df["Time"].notna(), "15:00")
    else:
        t = pd.Series("15:00", index=df.index)
    t = t.where(t.str.match(r"^\d{1,2}:\d{2}$"), "15:00")
    local = pd.to_datetime(day.dt.strftime("%Y-%m-%d") + " " + t, errors="coerce")
    return local.dt.tz_localize(ZoneInfo(tz), ambiguous="NaT",
                                nonexistent="shift_forward").dt.tz_convert("UTC")


def _normalise_fd(df: pd.DataFrame, league_id: int, season: str) -> pd.DataFrame:
    df = df.dropna(subset=["HomeTeam", "AwayTeam"]).copy()
    if df.empty:
        return pd.DataFrame()
    out = pd.DataFrame({
        "league_id": league_id,
        "season": season,
        "kickoff": _parse_kickoff(df, C.FD_TIME_ZONE),
        "home": df["HomeTeam"].astype(str).str.strip(),
        "away": df["AwayTeam"].astype(str).str.strip(),
        "hg": pd.to_numeric(df.get("FTHG"), errors="coerce"),
        "ag": pd.to_numeric(df.get("FTAG"), errors="coerce"),
        "referee": df["Referee"].astype(str).str.strip() if "Referee" in df else np.nan,
    })
    for src, dst in STAT_COLS.items():
        out[dst] = pd.to_numeric(df[src], errors="coerce") if src in df else np.nan
    out.loc[out["referee"].isin(["nan", "", "None"]), "referee"] = np.nan
    return out.dropna(subset=["kickoff"])


# ------------------------------------------------------------------ results
def load_results(league: C.League, seasons: list[int], mode: str) -> pd.DataFrame:
    """Played matches for the given season start years.

    mode="live": football-data.co.uk first, GitHub mirror as fallback.
    mode="mirror": GitHub mirror only (used for offline testing).
    """
    frames = []
    for start in seasons:
        code = season_code(start)
        raw = None
        if mode == "live":
            raw = get(C.FD_SEASON_URL.format(season=code, code=league.fd_code))
        if raw is None:
            raw = get(C.MIRROR_URL.format(folder=league.mirror, season=code), tries=1)
        if raw is None:
            log.warning("%s %s: no results file", league.name, code)
            continue
        df = _normalise_fd(_read_csv(raw), league.id, code)
        frames.append(df)
        log.info("%s %s: %d rows", league.name, code, len(df))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    return df.sort_values("kickoff").reset_index(drop=True)


def load_fd_fixtures() -> pd.DataFrame:
    """Upcoming fixtures for every league from football-data.co.uk (about a week out)."""
    raw = get(C.FD_FIXTURES_URL)
    if raw is None:
        return pd.DataFrame()
    df = _read_csv(raw)
    if "Div" not in df:
        return pd.DataFrame()
    frames = []
    for code, g in df.groupby("Div"):
        lg = C.LEAGUE_BY_CODE.get(code)
        if lg is None:
            continue
        frames.append(_normalise_fd(g, lg.id, "fixtures"))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_openfootball(league: C.League, start_year: int, known_teams) -> pd.DataFrame:
    """Full-season schedule from openfootball, names mapped to football-data spelling."""
    s = f"{start_year}-{(start_year + 1) % 100:02d}"
    raw = get(C.OPENFOOTBALL_URL.format(season=s, stem=league.openfootball), tries=2)
    if raw is None:
        return pd.DataFrame()
    matches = json.loads(raw).get("matches", [])
    names = {m["team1"] for m in matches} | {m["team2"] for m in matches}
    # known teams first; the alias table covers newly promoted sides with no results yet
    mapping, missing = build_map(names, sorted(set(known_teams)))
    if missing:
        extra, missing = build_map(missing, sorted(set(ALIASES) - set(known_teams)))
        mapping.update(extra)
    if missing:
        log.warning("%s openfootball names not mapped: %s", league.name, missing)
    tz = ZoneInfo(league.tz)
    rows = []
    for m in matches:
        h, a = mapping.get(m["team1"]), mapping.get(m["team2"])
        if not h or not a:
            continue
        t = m.get("time") or "15:00"
        try:
            local = datetime.strptime(f"{m['date']} {t[:5]}", "%Y-%m-%d %H:%M")
        except ValueError:
            continue
        rows.append({"league_id": league.id, "season": "openfootball",
                     "kickoff": pd.Timestamp(local.replace(tzinfo=tz)).tz_convert("UTC"),
                     "home": h, "away": a, "referee": np.nan})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------- FPL
def load_fpl() -> dict | None:
    raw = get(C.FPL_BOOTSTRAP_URL, tries=2)
    if raw is None:
        return None
    boot = json.loads(raw)
    fx = get(C.FPL_FIXTURES_URL, tries=2)
    boot["_fixtures"] = json.loads(fx) if fx else []
    return boot


def now_utc() -> datetime:
    return datetime.now(timezone.utc)
