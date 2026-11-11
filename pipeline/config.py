"""Static configuration: leagues, sources and model settings."""
from dataclasses import dataclass


@dataclass(frozen=True)
class League:
    id: int            # stable id used by the page (matches the original build)
    name: str
    fd_code: str       # football-data.co.uk division code
    mirror: str        # folder name in the datasets/football-datasets GitHub mirror
    openfootball: str  # openfootball/football.json file stem
    tz: str            # local timezone of the league (openfootball times are local)


LEAGUES = [
    League(39, "Premier League", "E0", "premier-league", "en.1", "Europe/London"),
    League(140, "La Liga", "SP1", "la-liga", "es.1", "Europe/Madrid"),
    League(135, "Serie A", "I1", "serie-a", "it.1", "Europe/Rome"),
    League(78, "Bundesliga", "D1", "bundesliga", "de.1", "Europe/Berlin"),
    League(61, "Ligue 1", "F1", "ligue-1", "fr.1", "Europe/Paris"),
]
LEAGUE_BY_CODE = {lg.fd_code: lg for lg in LEAGUES}

# ----------------------------------------------------------------- sources
FD_SEASON_URL = "https://www.football-data.co.uk/mmz4281/{season}/{code}.csv"
FD_FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"
MIRROR_URL = ("https://raw.githubusercontent.com/datasets/football-datasets/"
              "master/datasets/{folder}/season-{season}.csv")
OPENFOOTBALL_URL = ("https://raw.githubusercontent.com/openfootball/football.json/"
                    "master/{season}/{stem}.json")
FPL_BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
FPL_FIXTURES_URL = "https://fantasy.premierleague.com/api/fixtures/"

# football-data.co.uk publishes kickoff times in UK time for every league.
FD_TIME_ZONE = "Europe/London"

# --------------------------------------------------------------------- site
# Public address of the site. Change this one line if you move to a custom domain.
SITE_URL = "https://pboy221.github.io/Pitchlab/"

# ------------------------------------------------------------------- model
MODEL_VERSION = "2.0.0"
HISTORY_SEASONS = 3          # current season plus the two before it
DECAY_PER_DAY = 0.0015       # Dixon-Coles style time decay (tuned on 2024-26 backtests)
RIDGE = 1.0                  # shrinks team ratings toward average (helps promoted sides)
MAX_GOALS = 10

UPCOMING_DAYS = 14           # how far ahead fixtures are shown
RECENT_DAYS = 7              # how far back results are shown
BACKTEST_BLOCK_DAYS = 7      # walk-forward refit cadence

STYLE_WINDOW = 38            # most recent league matches per team used for styles
FDR_KEEP = 0.10              # Benjamini-Hochberg level for a reported shift
FDR_CONFIRM = 0.05           # stricter level marked "confirmed"
