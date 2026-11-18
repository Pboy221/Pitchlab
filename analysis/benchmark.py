"""PitchLab vs the bookmakers: a five-season, five-league walk-forward benchmark.

    python -m analysis.benchmark                          # downloads from football-data.co.uk
    python -m analysis.benchmark --data-dir path/to/csvs  # local copy, no network

Every match from 2021-22 to 2025-26 is predicted using only matches played before
its week (refit weekly on the previous three years, exactly like the live site).
Those probabilities are then compared with three bookmaker benchmarks and run
through a flat-stake betting simulation. Writes reports/benchmark/:
results.json, results.md and three charts.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline import backtest, config as C, odds, sources

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "benchmark"
TEST_SEASONS = range(2021, 2026)          # season start years: 2021-22 ... 2025-26
TRAIN_DAYS = 3 * 365
log = logging.getLogger("pitchlab.benchmark")

# Chart colours: categorical slots 1-2 plus a neutral for the naive baseline.
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
MODEL, BOOK, BASE = "#2a78d6", "#eb6834", "#a3a29c"


# ---------------------------------------------------------------- run
def predict_all(data_dir: Path | None) -> pd.DataFrame:
    seasons = list(range(TEST_SEASONS.start - 3, TEST_SEASONS.stop))
    start = pd.Timestamp(f"{TEST_SEASONS.start}-07-01", tz="UTC")
    end = pd.Timestamp(f"{TEST_SEASONS.stop}-07-01", tz="UTC")
    frames = []
    for lg in C.LEAGUES:
        df = (sources.load_results_dir(lg, seasons, data_dir) if data_dir
              else sources.load_results(lg, seasons, "live"))
        if df.empty:
            raise SystemExit(f"No data for {lg.name}")
        if df[odds.cols("avg")].isna().all().all():
            log.warning("%s: no odds columns - is this the football-data.co.uk file?", lg.name)
        _, preds = backtest.run(df, start, end, train_days=TRAIN_DAYS)
        preds.insert(0, "league", lg.name)
        preds.insert(1, "season", preds["kickoff"].map(
            lambda t: sources.season_code(sources.season_start(t.date()))))
        frames.append(preds)
        log.info("%s: %d matches predicted", lg.name, len(preds))
    return pd.concat(frames, ignore_index=True)


def summarise(p: pd.DataFrame) -> dict:
    """Headline numbers for one group of matches (a league, a season or everything)."""
    y = backtest.result_index(p["hg"].to_numpy(), p["ag"].to_numpy())
    P = p[["p_home", "p_draw", "p_away"]].to_numpy()
    B = p[["b_home", "b_draw", "b_away"]].to_numpy()
    s = {"n_matches": int(len(p)), **backtest._metrics(P, y, B),
         "calibration_error": backtest.calibration_error(P, y)}
    s.update({"vs_bookmakers": backtest.vs_bookmakers(p), "betting": backtest.betting(p)})
    return s


def bets_by_outcome(p: pd.DataFrame, edge: float = odds.MAIN_EDGE) -> dict:
    """Which outcome the simulation backed, and how each did: shows *where* the
    model disagrees with the market."""
    O = p[odds.cols(odds.BET_AT)].to_numpy(float)
    ok = odds.valid(O)
    P = p[["p_home", "p_draw", "p_away"]].to_numpy()[ok]
    y = backtest.result_index(p["hg"].to_numpy(), p["ag"].to_numpy())[ok]
    r = odds.value_bets(P, O[ok], y, edge)
    picked = (P * O[ok] - 1).argmax(1)[r["rows"]]
    out = {}
    for i, name in enumerate(["home", "draw", "away"]):
        sel = picked == i
        out[name] = {"n_bets": int(sel.sum()),
                     "roi": round(float(r["pnl"][sel].mean()), 4) if sel.any() else None}
    return out


def favourite_gap(p: pd.DataFrame) -> dict:
    """Average probability the model and Pinnacle give the market favourite."""
    O = p[odds.cols("psc")].to_numpy(float)
    ok = odds.valid(O)
    B = odds.implied(O[ok])
    P = p[["p_home", "p_draw", "p_away"]].to_numpy()[ok]
    fav = B.argmax(1)
    idx = np.arange(len(fav))
    y = backtest.result_index(p["hg"].to_numpy(), p["ag"].to_numpy())[ok]
    return {"book": round(float(B[idx, fav].mean()), 4),
            "model": round(float(P[idx, fav].mean()), 4),
            "actual_win_rate": round(float((y == fav).mean()), 4)}


# ------------------------------------------------------------- charts
def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def charts(res: dict, preds: pd.DataFrame, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK,
                         "axes.labelcolor": MUTED, "figure.facecolor": SURFACE})

    # 1. log loss by league: dot plot, because the gaps are small next to the totals
    names = [lg.name for lg in C.LEAGUES] + ["All five"]
    groups = [res["leagues"][n] for n in names[:-1]] + [res["overall"]]
    fig, ax = plt.subplots(figsize=(7.5, 3.9), dpi=160)
    _style(ax)
    ys = np.arange(len(names))[::-1]
    for y_, g in zip(ys, groups):
        vals = [g["baseline_log_loss"], g["log_loss"], g["vs_bookmakers"]["avgc"]["book_log_loss"]]
        ax.plot([min(vals), max(vals)], [y_, y_], color=GRID, linewidth=2, zorder=1)
    for key, colour, label in [("base", BASE, "Base rates (always guess the league average)"),
                               ("model", MODEL, "PitchLab"),
                               ("book", BOOK, "Bookmakers (average closing odds)")]:
        xs = [g["baseline_log_loss"] if key == "base" else g["log_loss"] if key == "model"
              else g["vs_bookmakers"]["avgc"]["book_log_loss"] for g in groups]
        ax.scatter(xs, ys, s=64, color=colour, edgecolor=SURFACE, linewidth=2, zorder=3,
                   label=label)
    ax.set_yticks(ys, names)
    ax.get_yticklabels()[-1].set_fontweight("bold")
    ax.set_xlabel("Log loss per match (lower is better)")
    ax.set_title("PitchLab beats base rates but not the bookmakers",
                 loc="left", fontsize=12, fontweight="bold", pad=26)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, frameon=False, fontsize=8,
              handletextpad=0.3, columnspacing=1.2, labelcolor=MUTED)
    fig.text(0.01, 0.01, "Walk-forward backtest, 2021-22 to 2025-26, "
             f"{res['overall']['n_matches']:,} matches.", color=MUTED, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out / "log_loss.png")
    plt.close(fig)

    # 2. calibration: model and Pinnacle on the same matches, all outcomes pooled
    O = preds[odds.cols("psc")].to_numpy(float)
    ok = odds.valid(O)
    y = backtest.result_index(preds["hg"].to_numpy(), preds["ag"].to_numpy())[ok]
    hit = np.eye(3)[y].ravel()
    fig, ax = plt.subplots(figsize=(5.2, 5.0), dpi=160)
    _style(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.plot([0, 0.9], [0, 0.9], color=BASE, linewidth=1.2, linestyle=(0, (4, 3)),
            label="Perfectly calibrated")
    for P, colour, label in [(preds[["p_home", "p_draw", "p_away"]].to_numpy()[ok], MODEL, "PitchLab"),
                             (odds.implied(O[ok]), BOOK, "Pinnacle closing")]:
        rel = backtest.reliability(P.ravel(), hit, bins=18)
        rel = [r for r in rel if r["n"] >= 40]
        ax.plot([r["predicted"] for r in rel], [r["observed"] for r in rel], color=colour,
                linewidth=2, marker="o", markersize=5, markeredgecolor=SURFACE,
                markeredgewidth=1.5, label=label)
    ax.set_xlim(0, 0.9), ax.set_ylim(0, 0.9)
    ax.set_xlabel("Forecast probability")
    ax.set_ylabel("How often it actually happened")
    ax.set_title("Both are well calibrated", loc="left", fontsize=12, fontweight="bold")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", labelcolor=MUTED)
    fig.tight_layout()
    fig.savefig(out / "calibration.png")
    plt.close(fig)

    # 3. cumulative profit of the main betting rule
    O = preds[odds.cols(odds.BET_AT)].to_numpy(float)
    ok = odds.valid(O)
    sub = preds[ok].reset_index(drop=True)
    r = odds.value_bets(sub[["p_home", "p_draw", "p_away"]].to_numpy(), O[ok],
                        backtest.result_index(sub["hg"].to_numpy(), sub["ag"].to_numpy()),
                        odds.MAIN_EDGE)
    when = sub.loc[r["rows"], "kickoff"].to_numpy()
    order = np.argsort(when, kind="stable")
    cum = np.cumsum(r["pnl"][order])
    fig, ax = plt.subplots(figsize=(7.5, 3.6), dpi=160)
    _style(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.grid(axis="x", visible=False)
    ax.axhline(0, color=MUTED, linewidth=1)
    ax.plot(pd.to_datetime(when[order]), cum, color=MODEL, linewidth=2)
    ax.text(0.99, 0.95, f"{cum[-1]:+,.0f} units after {len(cum):,} bets\n"
            f"{r['roi']:+.1%} per bet", transform=ax.transAxes, ha="right", va="top",
            fontsize=9, color=INK)
    ax.set_ylabel("Cumulative profit (1-unit stakes)")
    ax.set_title(f"Betting every {odds.MAIN_EDGE:.0%}+ 'edge' at average odds loses money",
                 loc="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "betting.png")
    plt.close(fig)


# -------------------------------------------------------------- report
def markdown(res: dict) -> str:
    o = res["overall"]
    lines = ["| League | Matches | Base rates | PitchLab | Bookmakers (avg closing) "
             "| Pinnacle closing | PitchLab minus Pinnacle (95% CI) |",
             "|---|---:|---:|---:|---:|---:|---|"]
    for name, g in list(res["leagues"].items()) + [("**All five**", o)]:
        ps, av = g["vs_bookmakers"]["psc"], g["vs_bookmakers"]["avgc"]
        lo, hi = ps["model_minus_book_ci95"]
        lines.append(f"| {name} | {g['n_matches']:,} | {g['baseline_log_loss']:.3f} "
                     f"| {g['log_loss']:.3f} | {av['book_log_loss']:.3f} "
                     f"| {ps['book_log_loss']:.3f} | +{ps['model_minus_book']:.3f} "
                     f"({lo:+.3f} to {hi:+.3f}) |")
    lines += ["", "| Betting rule (flat 1-unit stakes, average pre-match odds) | Bets "
              "| Hit rate | Avg odds | ROI per bet (95% CI) |", "|---|---:|---:|---:|---|"]
    for b in o["betting"]:
        lo, hi = b["roi_ci95"]
        lines.append(f"| Back the best outcome when model edge > {b['edge']:.0%} "
                     f"| {b['n_bets']:,} | {b['hit_rate']:.1%} | {b['avg_odds']:.2f} "
                     f"| {b['roi']:+.1%} ({lo:+.1%} to {hi:+.1%}) |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", type=Path, help="local football-data CSVs (see sources.load_results_dir)")
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s",
                        stream=sys.stdout)
    preds = predict_all(a.data_dir)
    res = {"test_seasons": f"{TEST_SEASONS.start}-{TEST_SEASONS.start % 100 + 1:02d} to "
                           f"{TEST_SEASONS.stop - 1}-{TEST_SEASONS.stop % 100:02d}",
           "model_version": C.MODEL_VERSION, "overall": summarise(preds),
           "leagues": {n: summarise(g) for n, g in preds.groupby("league", sort=False)},
           "seasons": {s: summarise(g) for s, g in preds.groupby("season")},
           "bets_by_outcome": bets_by_outcome(preds), "favourite": favourite_gap(preds)}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "results.json").write_text(json.dumps(res, indent=1, allow_nan=False))
    (a.out / "results.md").write_text(markdown(res))
    charts(res, preds, a.out)
    print(markdown(res))
    print(json.dumps({k: res[k] for k in ("bets_by_outcome", "favourite")}, indent=1))


if __name__ == "__main__":
    main()
