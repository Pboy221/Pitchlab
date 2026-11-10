"""Premier League player markets from the free Fantasy Premier League feed.

For each player:
  * availability from FPL status / chance of playing
  * start probability = starts / team matches x availability
  * expected minutes  = minutes / team matches x availability
  * scoring:  the player's share of his team's expected goals (xG per 90,
              shrunk toward his position's average), applied to the team's
              expected goals in THIS match from the result model
  * assists:  same, with expected assists
  * cards:    yellow cards per 90 (shrunk), scaled by the team's expected cards
              in this match (which already carries the referee)
Probabilities are P(at least one) = 1 - exp(-rate).
"""
from __future__ import annotations

import math

import numpy as np

from .names import build_map

POS = {1: "GK", 2: "DEF", 3: "MID", 4: "ATT"}
PSEUDO_MIN_ATTACK = 450.0   # five full matches of prior for xG / xA rates
PSEUDO_MIN_CARDS = 900.0    # ten full matches of prior for card rates


def _f(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def _availability(e) -> float:
    st = e.get("status", "a")
    if st == "a":
        return 1.0
    if st == "d":
        c = e.get("chance_of_playing_next_round")
        return (c if c is not None else 50) / 100.0
    return 0.0


def _name(e) -> str:
    full = f"{e.get('first_name', '')} {e.get('second_name', '')}".strip()
    return full if len(full) <= 20 else e.get("web_name") or full


class PlayerModel:
    def __init__(self, boot: dict, known_teams: list[str]):
        self.ok = False
        if not boot or "elements" not in boot:
            return
        teams = {t["id"]: t["name"] for t in boot.get("teams", [])}
        mapping, self.unmapped = build_map(teams.values(), known_teams)
        self.team_name = {tid: mapping.get(n) for tid, n in teams.items()}

        games = {tid: 0 for tid in teams}
        for fx in boot.get("_fixtures", []):
            if fx.get("finished"):
                games[fx["team_h"]] = games.get(fx["team_h"], 0) + 1
                games[fx["team_a"]] = games.get(fx["team_a"], 0) + 1
        if not any(games.values()):
            done = sum(1 for ev in boot.get("events", []) if ev.get("finished"))
            games = {tid: done for tid in teams}

        els = boot["elements"]
        # position priors per 90 from regulars (900+ minutes)
        prior = {}
        for pos in POS:
            reg = [e for e in els if e["element_type"] == pos and e.get("minutes", 0) >= 900]
            mins = sum(e["minutes"] for e in reg) or 1
            prior[pos] = {
                "xg": 90 * sum(_f(e.get("expected_goals")) or e.get("goals_scored", 0) for e in reg) / mins,
                "xa": 90 * sum(_f(e.get("expected_assists")) or e.get("assists", 0) for e in reg) / mins,
                "yc": 90 * sum(e.get("yellow_cards", 0) for e in reg) / mins,
            }

        self.by_team: dict[str, list[dict]] = {}
        for e in els:
            fd = self.team_name.get(e["team"])
            g = games.get(e["team"], 0)
            if not fd or g == 0:
                continue
            pos = e["element_type"]
            mins = e.get("minutes", 0)
            xg = _f(e.get("expected_goals")) or e.get("goals_scored", 0)
            xa = _f(e.get("expected_assists")) or e.get("assists", 0)
            pr = prior[pos]
            avail = _availability(e)
            p = {
                "player_id": e["id"], "name": _name(e), "team": fd, "position": POS[pos],
                "available": avail > 0,
                "start_probability": round(min(0.99, e.get("starts", 0) / g * avail), 3),
                "exp_min": min(90.0, mins / g) * avail,
                "xg90": 90 * (xg + pr["xg"] * PSEUDO_MIN_ATTACK / 90) / (mins + PSEUDO_MIN_ATTACK),
                "xa90": 90 * (xa + pr["xa"] * PSEUDO_MIN_ATTACK / 90) / (mins + PSEUDO_MIN_ATTACK),
                "yc90": 90 * (e.get("yellow_cards", 0) + pr["yc"] * PSEUDO_MIN_CARDS / 90)
                        / (mins + PSEUDO_MIN_CARDS),
            }
            self.by_team.setdefault(fd, []).append(p)
        # team totals for share calculations
        self.team_xg, self.team_xa, self.team_yc = {}, {}, {}
        for t, ps in self.by_team.items():
            self.team_xg[t] = sum(p["xg90"] * p["exp_min"] / 90 for p in ps) or 1e-6
            self.team_xa[t] = sum(p["xa90"] * p["exp_min"] / 90 for p in ps) or 1e-6
            self.team_yc[t] = sum(p["yc90"] * p["exp_min"] / 90 for p in ps) or 1e-6
        self.ok = bool(self.by_team)

    def has(self, team: str) -> bool:
        return self.ok and team in self.by_team

    def fixture(self, home: str, away: str, pred: dict) -> dict:
        scorers, assisters, cards, lineups = [], [], [], {}
        tc = pred.get("team_cards") or {}
        for side, team in (("home", home), ("away", away)):
            ps = self.by_team.get(team, [])
            team_goals = pred["expected_goals"][side]
            team_cards = tc.get(f"expected_{side}")
            for p in ps:
                if p["exp_min"] < 5:
                    continue
                share_g = p["xg90"] * p["exp_min"] / 90 / self.team_xg[team]
                share_a = p["xa90"] * p["exp_min"] / 90 / self.team_xa[team]
                lam_g = share_g * team_goals
                lam_a = share_a * team_goals * 0.85      # roughly 85% of goals are assisted
                base = {"player_id": p["player_id"], "name": p["name"], "team": team,
                        "side": side}
                scorers.append({**base, "probability": round(1 - math.exp(-lam_g), 4),
                                "expected": round(lam_g, 3),
                                "start_probability": p["start_probability"]})
                assisters.append({**base, "probability": round(1 - math.exp(-lam_a), 4)})
                if team_cards is not None:
                    share_c = p["yc90"] * p["exp_min"] / 90 / self.team_yc[team]
                    lam_c = share_c * team_cards
                    cards.append({**base, "probability": round(1 - math.exp(-lam_c), 4),
                                  "referee_multiplier": tc.get("referee_multiplier", 1.0)})
            gk = sorted([p for p in ps if p["position"] == "GK"], key=lambda p: -p["start_probability"])[:1]
            out = sorted([p for p in ps if p["position"] != "GK"], key=lambda p: -p["start_probability"])[:10]
            lineups[side] = [{"player_id": p["player_id"], "name": p["name"],
                              "position": p["position"], "available": p["available"],
                              "start_probability": p["start_probability"]} for p in gk + out]
        return {
            "scorers": sorted(scorers, key=lambda x: -x["probability"])[:10],
            "assisters": sorted(assisters, key=lambda x: -x["probability"])[:8],
            "cards": sorted(cards, key=lambda x: -x["probability"])[:8],
            "lineups": lineups,
        }
