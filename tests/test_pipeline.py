"""Offline tests: run before every build so a bad change never reaches the site.

    pip install pytest && pytest -q
"""
import json
import re
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest

from pipeline import backtest, build, config as C, ledger, markets, ratings, sources, styles
from pipeline.names import build_map, match_name


# --------------------------------------------------------------- fixtures
def synthetic_league(league_id=39, seasons=(2024, 2025, 2026), seed=1, upto="2026-11-01"):
    """Three seasons of double round-robin with known team strengths."""
    rng = np.random.default_rng(seed)
    teams = [f"Team{i:02d}" for i in range(20)]
    att = np.linspace(0.45, -0.45, 20)          # Team00 strongest
    rows = []
    for s in seasons:
        day = pd.Timestamp(f"{s}-08-15 14:00", tz="UTC")
        pairs = [(h, a) for h in range(20) for a in range(20) if h != a]
        rng.shuffle(pairs)
        for k, (h, a) in enumerate(pairs):
            ko = day + pd.Timedelta(days=7 * (k // 10))
            lh = np.exp(0.15 + 0.25 + att[h] + att[a] * -1)
            la = np.exp(0.15 + att[a] - att[h])
            hc, ac = rng.poisson(5.5 + 2 * att[h]), rng.poisson(4.5 + 2 * att[a])
            rows.append({"league_id": league_id, "season": sources.season_code(s), "kickoff": ko,
                         "home": teams[h], "away": teams[a],
                         "hg": rng.poisson(lh), "ag": rng.poisson(la),
                         "hs": rng.poisson(13), "as_": rng.poisson(11), "hst": rng.poisson(4.5),
                         "ast": rng.poisson(3.8), "hf": rng.poisson(11), "af": rng.poisson(11.5),
                         "hc": hc, "ac": ac, "hy": rng.poisson(1.8), "ay": rng.poisson(2.1),
                         "hr": 0, "ar": 0, "referee": f"Ref{rng.integers(12)}"})
    df = pd.DataFrame(rows).sort_values("kickoff").reset_index(drop=True)
    df["hst"] = np.minimum(df["hst"], df["hs"])
    df["ast"] = np.minimum(df["ast"], df["as_"])
    df.loc[df["kickoff"] >= pd.Timestamp(upto, tz="UTC"), ["hg", "ag"]] = np.nan
    return df


# ------------------------------------------------------------------ names
@pytest.mark.parametrize("raw,expected", [
    ("Manchester United FC", "Man United"), ("Manchester City FC", "Man City"),
    ("Club Atlético de Madrid", "Ath Madrid"), ("Real Madrid CF", "Real Madrid"),
    ("RCD Espanyol de Barcelona", "Espanol"), ("FC Barcelona", "Barcelona"),
    ("Deportivo Alavés", "Alaves"), ("Paris Saint-Germain FC", "Paris SG"),
    ("Paris FC", "Paris FC"), ("FC Internazionale Milano", "Inter"), ("AC Milan", "Milan"),
    ("Borussia Mönchengladbach", "M'gladbach"), ("1. FC Köln", "FC Koln"),
    ("Spurs", "Tottenham"), ("Man Utd", "Man United"), ("Nott'm Forest", "Nott'm Forest"),
])
def test_name_mapping(raw, expected):
    cands = ["Man United", "Man City", "Ath Madrid", "Real Madrid", "Espanol", "Barcelona",
             "Alaves", "La Coruna", "Paris SG", "Paris FC", "Inter", "Milan", "M'gladbach",
             "Dortmund", "FC Koln", "Tottenham", "Nott'm Forest"]
    assert match_name(raw, cands) == expected


def test_unknown_name_is_left_out():
    mapping, missing = build_map(["Some Unknown Club"], ["Arsenal"])
    assert mapping == {} and missing == ["Some Unknown Club"]


# ------------------------------------------------------------------ model
def test_ratings_recover_strength_order():
    df = synthetic_league()
    now = pd.Timestamp("2026-11-01", tz="UTC")
    m = ratings.fit(df[df["hg"].notna()], "hg", "ag", now, dixon_coles=True)
    assert m.strength("Team00") > m.strength("Team10") > m.strength("Team19")
    assert m.home > 0                                   # home advantage is positive
    assert -0.2 <= m.rho <= 0.2


def test_prediction_is_a_valid_distribution():
    df = synthetic_league()
    lm = markets.fit_league(df[df["hg"].notna()], pd.Timestamp("2026-11-01", tz="UTC"))
    p = markets.predict(lm, "Team00", "Team19", "Ref1")
    o = p["outcome"]
    assert abs(o["home"] + o["draw"] + o["away"] - 1) < 1e-3
    assert o["home"] > o["away"]
    for v in (p["btts"], p["totals"]["over"], p["corners"]["over"], p["team_cards"]["over_3_5"]):
        assert 0 <= v <= 1
    assert p["corners"]["line"] % 1 == 0.5
    assert p["team_cards"]["referee_matches"] > 0
    unseen = markets.predict(lm, "Promoted FC", "Team00", None)   # never-seen team still works
    assert unseen["outcome"]["away"] > unseen["outcome"]["home"]


def test_score_matrix_sums_to_one():
    m = ratings.score_matrix(1.6, 1.1, -0.08)
    assert abs(m.sum() - 1) < 1e-9 and (m >= 0).all()


def test_backtest_beats_baseline_on_structured_data():
    df = synthetic_league()
    summary, preds = backtest.run(df, pd.Timestamp("2025-07-01", tz="UTC"),
                                  pd.Timestamp("2026-11-01", tz="UTC"))
    assert summary["n_matches"] > 300
    assert summary["skill_vs_baseline"] > 0
    assert summary["calibration_error"] < 0.08
    # walk-forward: no prediction may use a later block's data
    assert preds["kickoff"].is_monotonic_increasing


# ----------------------------------------------------------------- ledger
def test_ledger_lifecycle(tmp_path):
    led = {"entries": {}}
    ko = pd.Timestamp("2026-11-20 15:00", tz="UTC")
    fx = {"league_id": 39, "home": "A", "away": "B", "kickoff": ko.isoformat(),
          "outcome": {"home": 0.5, "draw": 0.3, "away": 0.2}}
    k = ledger.key(39, "2627", "A", "B")
    ledger.record(led, k, fx, ko - pd.Timedelta(days=2), "t")
    assert k in led["entries"]
    fx2 = {**fx, "outcome": {"home": 0.1, "draw": 0.1, "away": 0.8}}
    ledger.record(led, k, fx2, ko + pd.Timedelta(minutes=5), "t")   # after kickoff: frozen
    assert led["entries"][k]["outcome"]["home"] == 0.5
    late = ledger.key(39, "2627", "C", "D")
    ledger.record(led, late, {**fx, "home": "C", "away": "D"}, ko + pd.Timedelta(hours=1), "t")
    assert late not in led["entries"]                               # never logged after kickoff
    res = pd.DataFrame([{"league_id": 39, "home": "A", "away": "B", "hg": 2, "ag": 0, "kickoff": ko}])
    assert ledger.settle(led, res, lambda ts: "2627") == 1
    s = ledger.summary(led, 39)
    assert s["n"] == 1 and "calibration_error" not in s             # hidden below 50
    path = tmp_path / "l.json"
    ledger.save(path, led)
    assert ledger.load(path) == led


# ----------------------------------------------------------------- styles
def test_fdr_is_monotone_and_bounded():
    p = np.array([0.001, 0.02, 0.04, 0.3, 0.9])
    q = styles._bh(p)
    assert (q >= p).all() and (q <= 1).all()
    assert list(np.argsort(q)) == list(np.argsort(p))


# --------------------------------------------------------- full build, offline
def test_full_build_offline(tmp_path, monkeypatch):
    frames = {lg.id: synthetic_league(lg.id, seed=lg.id) for lg in C.LEAGUES}
    monkeypatch.setattr(sources, "load_results", lambda lg, seasons, mode: frames[lg.id])
    out = tmp_path / "public"
    data = build.build("mirror", "2026-11-01", out, tmp_path / "ledger.json", None)
    assert len(data["leagues"]) == 5 and data["fixtures"]
    for name in ["index.html", "data.json", "sw.js", "manifest.webmanifest", "404.html",
                 "robots.txt", "sitemap.xml", "og.png", "icons/icon-512.png",
                 "fonts/inter-latin-wght-normal.woff2"]:
        assert (out / name).exists(), name
    page = (out / "index.html").read_text()
    assert "__SITE_URL__" not in page and "/*__PITCHLAB_DATA__*/" not in page
    assert "fonts.googleapis.com" not in page                      # no third-party requests
    json.loads((out / "data.json").read_text())
    json.loads((out / "manifest.webmanifest").read_text())
    f = data["fixtures"][0]
    for key in ("match_id", "kickoff", "outcome", "expected_goals", "scorelines", "corners",
                "team_cards", "style_detail", "totals", "btts"):
        assert key in f, key

    # the page's JavaScript must at least parse
    if shutil.which("node"):
        scripts = re.findall(r"<script>(.*?)</script>", page, re.S)
        js = tmp_path / "page.js"
        js.write_text("\n".join(scripts))
        r = subprocess.run(["node", "--check", str(js)], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
