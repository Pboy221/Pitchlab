"""PitchLab daily build.

    python -m pipeline.build                       # live run (what GitHub Actions does)
    python -m pipeline.build --mode mirror --as-of 2026-03-14   # offline replay for testing

Fetches results and fixtures, fits the models, runs the walk-forward backtest,
logs pre-kickoff forecasts to the ledger, settles finished ones, and writes
site/data.json plus site/index.html.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from . import backtest, config as C, ledger, markets, sources, styles
from .players import PlayerModel

log = logging.getLogger("pitchlab")
ROOT = Path(__file__).resolve().parent.parent

NOTES = {
    "scope": "Match, corner, card and style models run on free match-level data from "
             "football-data.co.uk. Player markets cover the Premier League only and come "
             "from the free Fantasy Premier League feed.",
    "refresh": "Rebuilt automatically twice a day. Each forecast is logged to the ledger "
               "before kickoff and scored once the result is in.",
    "cards": "Referee strictness is estimated with shrinkage from that official's own "
             "matches; where no referee is appointed yet, card numbers use team rates only.",
    "corners": "Fitted only on matches that carry corner data; corner and card totals allow "
               "for the extra spread real counts show over a plain Poisson.",
    "styles": "Style dimensions come from shots, shots on target, corners, fouls and cards. "
              "Situational shifts are measured against the league's own home/away and "
              "opponent-quality swing, so only departures from the norm are reported.",
}


def iso(ts: pd.Timestamp) -> str:
    return pd.Timestamp(ts).tz_convert("UTC").isoformat()


def season_of(ts) -> str:
    return sources.season_code(sources.season_start(pd.Timestamp(ts).date()))


def upcoming_live(played: pd.DataFrame, now: pd.Timestamp, start_year: int) -> pd.DataFrame:
    horizon = now + pd.Timedelta(days=C.UPCOMING_DAYS)
    fd = sources.load_fd_fixtures()
    frames = [fd] if not fd.empty else []
    for lg in C.LEAGUES:
        known = sorted(set(played.loc[played["league_id"] == lg.id, "home"]) |
                       set(played.loc[played["league_id"] == lg.id, "away"]))
        of = sources.load_openfootball(lg, start_year, known)
        if not of.empty:
            frames.append(of)
    if not frames:
        return pd.DataFrame()
    up = pd.concat(frames, ignore_index=True)
    up = up[(up["kickoff"] > now) & (up["kickoff"] <= horizon)]
    # football-data rows first so their kickoff time and referee win the dedupe
    up["pri"] = (up["season"] != "fixtures").astype(int)
    up = up.sort_values(["pri", "kickoff"]).drop_duplicates(["league_id", "home", "away"])
    # drop anything already played (openfootball can lag on postponements)
    done = set(zip(played["league_id"], played["home"], played["away"],
                   played["kickoff"].map(season_of)))
    up = up[[(r.league_id, r.home, r.away, season_of(r.kickoff)) not in done
             for r in up.itertuples()]]
    return up.drop(columns="pri").sort_values("kickoff").reset_index(drop=True)


def build(mode: str, as_of: str | None, out: Path, ledger_path: Path, fpl_file: Path | None) -> dict:
    now = pd.Timestamp(as_of, tz="UTC") + pd.Timedelta(hours=6) if as_of else pd.Timestamp.now(tz="UTC")
    start_year = sources.season_start(now.date())
    seasons = list(range(start_year - C.HISTORY_SEASONS + 1, start_year + 1))
    log.info("Build as of %s, seasons %s, mode %s", now, seasons, mode)

    frames = [sources.load_results(lg, seasons, mode) for lg in C.LEAGUES]
    allres = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    played = allres[(allres["kickoff"] < now) & allres["hg"].notna()].copy()

    if mode == "live" and now.month not in (6, 7):
        # In season, a missing current-season file means the source is down. Fail the
        # run (GitHub emails the owner) and leave yesterday's site up, rather than
        # quietly publishing stale numbers.
        cur = sources.season_code(start_year)
        have = played.loc[played["season"] == cur, "league_id"].nunique()
        if have == 0 and now.month not in (8,):
            raise SystemExit(f"No {cur} results from football-data.co.uk - source down?")
        stale = played.groupby("league_id")["kickoff"].max()
        for lid, last in stale.items():
            if now - last > pd.Timedelta(days=21) and now.month not in (8,):
                log.warning("League %s: latest result is %s - data may be stale", lid, last)

    if mode == "mirror":
        # replay: the matches after `as_of` become "upcoming", scores hidden
        horizon = now + pd.Timedelta(days=C.UPCOMING_DAYS)
        upcoming = allres[(allres["kickoff"] >= now) & (allres["kickoff"] < horizon)].copy()
        upcoming[["hg", "ag"]] = np.nan
        boot = json.loads(fpl_file.read_text()) if fpl_file else None
    else:
        upcoming = upcoming_live(played, now, start_year)
        boot = sources.load_fpl()
        if boot is None:
            log.warning("FPL feed unavailable; player markets skipped this run")

    led = ledger.load(ledger_path)
    settled = ledger.settle(led, played, season_of)
    log.info("Ledger: %d newly settled", settled)

    leagues_out, fixtures_out, recent_out = [], [], []
    for lg in C.LEAGUES:
        df = played[played["league_id"] == lg.id]
        if df.empty:
            log.warning("%s: no data, skipped", lg.name)
            continue
        lm = markets.fit_league(df, now)
        if lm is None:
            log.warning("%s: too little data to fit", lg.name)
            continue

        bt_start = pd.Timestamp(f"{start_year - 1}-07-01", tz="UTC")
        bt, bt_preds = backtest.run(df, bt_start, now)
        log.info("%s backtest: %s matches, skill %.3f", lg.name, bt.get("n_matches"),
                 bt.get("skill_vs_baseline", float("nan")))

        cur = df[df["kickoff"].map(season_of) == sources.season_code(start_year)]
        up = upcoming[upcoming["league_id"] == lg.id] if not upcoming.empty else upcoming
        teams = sorted(set(cur["home"]) | set(cur["away"]) |
                       (set(up["home"]) | set(up["away"]) if len(up) else set()))
        if len(teams) < 10:          # early season: fall back to last season's sides
            teams = sorted(set(df["home"].tail(380)) | set(df["away"].tail(380)))
        rows = styles.team_rows(df)
        strength = {t: lm.goals.strength(t) for t in teams}
        prof = styles.profiles(rows, teams, lg.id)
        sit = styles.situational(rows, teams, strength, lg.id)
        top = set(sit.get("top_third", []))

        pm = PlayerModel(boot, teams) if (lg.id == 39 and boot) else None

        n_fx = 0
        for r in up.itertuples() if len(up) else []:
            ref = r.referee if isinstance(r.referee, str) else None
            pred = markets.predict(lm, r.home, r.away, ref)
            hid, aid = styles.team_id(lg.id, r.home), styles.team_id(lg.id, r.away)
            hp, ap = prof["teams"].get(str(hid)), prof["teams"].get(str(aid))
            k = ledger.key(lg.id, season_of(r.kickoff), r.home, r.away)
            fx = {
                "match_id": zlib.crc32(k.encode()) % 1_000_000_000,
                "kickoff": iso(r.kickoff), "home": r.home, "away": r.away,
                "home_id": hid, "away_id": aid, "referee": ref or "TBC",
                **pred,
                "styles": {"home": hp["style"] if hp else "", "away": ap["style"] if ap else ""},
                "style_detail": {
                    "home": {"team_id": hid, "name": r.home, "style": hp["style"],
                             "top_dimension_keys": [d["key"] for d in sorted(
                                 hp["dimensions"], key=lambda d: -abs(d["z"]))[:3]],
                             "adapts_here": styles.fixture_shifts(sit, lg.id, r.home, False, r.away in top)}
                    if hp else None,
                    "away": {"team_id": aid, "name": r.away, "style": ap["style"],
                             "top_dimension_keys": [d["key"] for d in sorted(
                                 ap["dimensions"], key=lambda d: -abs(d["z"]))[:3]],
                             "adapts_here": styles.fixture_shifts(sit, lg.id, r.away, True, r.home in top)}
                    if ap else None,
                    "matchup": styles.matchup(hp, ap, r.home, r.away),
                },
                "has_players": False, "league_id": lg.id, "league": lg.name,
            }
            if pm and pm.has(r.home) and pm.has(r.away):
                fx.update(pm.fixture(r.home, r.away, pred))
                fx["has_players"] = True
            else:
                fx.update({"scorers": [], "assisters": [], "cards": [], "lineups": {}})
            fixtures_out.append(fx)
            ledger.record(led, k, fx, now, C.MODEL_VERSION)
            n_fx += 1

        # recent results with the forecast that was made for them
        bt_idx = {(p.home, p.away, season_of(p.kickoff)): p for p in bt_preds.itertuples()} \
            if len(bt_preds) else {}
        rec = df[df["kickoff"] >= now - pd.Timedelta(days=C.RECENT_DAYS)]
        for r in rec.itertuples():
            s = season_of(r.kickoff)
            e = led["entries"].get(ledger.key(lg.id, s, r.home, r.away))
            if e:
                o, src = e["outcome"], "ledger"
            elif (r.home, r.away, s) in bt_idx:
                p = bt_idx[(r.home, r.away, s)]
                o, src = {"home": round(p.p_home, 4), "draw": round(p.p_draw, 4),
                          "away": round(p.p_away, 4)}, "replay"
            else:
                o, src = None, None
            recent_out.append({"league_id": lg.id, "kickoff": iso(r.kickoff),
                               "home": r.home, "away": r.away,
                               "score": {"home": int(r.hg), "away": int(r.ag)},
                               "outcome": o, "source": src})

        leagues_out.append({
            "id": lg.id, "name": lg.name, "backtest": bt, "styles": prof,
            "situational": sit, "ledger": ledger.summary(led, lg.id),
            "fixture_count": n_fx,
            "latest_result": iso(df["kickoff"].max()),
        })

    if not leagues_out:
        raise SystemExit("No league could be built - check the data sources.")

    ledger.save(ledger_path, led)
    data = {
        "generated_at": iso(pd.Timestamp.now(tz="UTC")) if mode == "live" else iso(now),
        "model_version": C.MODEL_VERSION,
        "data_source": "free" if mode == "live" else "replay",
        "as_of": iso(now),
        "has_players": any(f["has_players"] for f in fixtures_out),
        "multi": True,
        "leagues": leagues_out,
        "fixtures": sorted(fixtures_out, key=lambda f: f["kickoff"]),
        "recent": sorted(recent_out, key=lambda f: f["kickoff"], reverse=True),
        "model_notes": NOTES,
    }
    out.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    (out / "data.json").write_text(payload)
    template = (ROOT / "site" / "template.html").read_text()
    marker = "/*__PITCHLAB_DATA__*/null"
    if marker not in template:
        raise SystemExit("Template is missing the data marker.")
    (out / "index.html").write_text(template.replace(marker, payload.replace("</", "<\\/")))
    log.info("Wrote %d fixtures, %d recent results, %d leagues", len(fixtures_out),
             len(recent_out), len(leagues_out))
    return data


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", choices=["live", "mirror"], default="live")
    ap.add_argument("--as-of", help="replay date YYYY-MM-DD (mirror mode)")
    ap.add_argument("--out", type=Path, default=ROOT / "public")
    ap.add_argument("--ledger", type=Path, default=ROOT / "data" / "ledger.json")
    ap.add_argument("--fpl-file", type=Path, help="saved FPL bootstrap JSON (testing)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s",
                        stream=sys.stdout)
    build(a.mode, a.as_of, a.out, a.ledger, a.fpl_file)


if __name__ == "__main__":
    main()
