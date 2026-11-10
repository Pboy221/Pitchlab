# PitchLab

Match, player, corner and card forecasts for the Premier League, La Liga,
Serie A, Bundesliga and Ligue 1 — rebuilt automatically twice a day from free
data, with every forecast logged before kickoff and scored afterwards.

## How it updates itself

`.github/workflows/daily.yml` runs on GitHub Actions at 06:17 and 18:17 UTC
(5:17 pm and 5:17 am Sydney in summer). Each run:

1. downloads the latest results and stats from **football-data.co.uk**, upcoming
   fixtures from football-data.co.uk and **openfootball**, and Premier League
   player data from the **Fantasy Premier League** feed;
2. refits the models and re-runs the walk-forward backtest;
3. writes every upcoming forecast to `data/ledger.json`, settles finished
   matches, and commits the ledger back to the repo;
4. publishes the site to **GitHub Pages**.

If football-data.co.uk is down during the season the run fails on purpose,
GitHub emails you, and yesterday's site stays up rather than showing stale
numbers. Nothing needs doing — the next run retries.

## One-time setup

1. Create a **public** GitHub repository (Pages is free for public repos) and
   push this folder to it.
2. Repo **Settings → Pages → Build and deployment → Source: GitHub Actions**.
3. Repo **Actions** tab → *Daily build* → **Run workflow** for the first build.

The site appears at `https://<your-username>.github.io/<repo-name>/`.
GitHub turns scheduled workflows off after 60 days without repository
activity; the ledger commit on every run keeps it active.

## Run it locally

```bash
pip install -r requirements.txt
python -m pipeline.build --out public          # live data
open public/index.html

# offline replay from a past date using the GitHub mirror of football-data
python -m pipeline.build --mode mirror --as-of 2026-03-14 \
    --out /tmp/replay --ledger /tmp/replay-ledger.json
```

## Layout

| Path | What it does |
|---|---|
| `pipeline/sources.py` | downloads and normalises every data source |
| `pipeline/names.py` | maps openfootball and FPL team names onto football-data names |
| `pipeline/ratings.py` | time-weighted Poisson team model with Dixon-Coles correction |
| `pipeline/markets.py` | outcome, scorelines, totals, BTTS, corners, cards (with referee) |
| `pipeline/backtest.py` | walk-forward backtest: log loss, Brier, calibration |
| `pipeline/styles.py` | eight style dimensions and FDR-corrected situational shifts |
| `pipeline/players.py` | Premier League scorer, assist and card probabilities |
| `pipeline/ledger.py` | pre-kickoff forecast ledger, settled after the match |
| `pipeline/build.py` | runs it all and writes `public/` |
| `site/template.html` | the page; data is injected at build time |
| `data/ledger.json` | the forecast record (written by the bot) |

## Model in brief

Goals: Poisson regression on attack, defence and home advantage, weighted so
recent matches count more (decay 0.0015/day) and shrunk toward average (ridge
1.0) so newly promoted sides don't get extreme ratings; Dixon-Coles ρ corrects
low-score draws. Settings were chosen on 2024-26 walk-forward backtests across
all five leagues: 6–9% better log loss than the league's base rates, with
calibration error around 0.02.

Corners and cards use the same team model on their own counts, with a
negative-binomial spread fitted per league. Card totals carry a shrunk
per-referee multiplier where the referee is known.
