"""Map team names from openfootball and FPL onto football-data.co.uk names.

football-data.co.uk is the canonical source (results, stats, fixtures), so every
other source is translated into its spelling. Matching uses an alias table of
distinctive fragments, longest match wins, so "Paris FC" and "Paris SG", or
"Real Madrid" and "Ath Madrid", never collide.
"""
import re
import unicodedata

# football-data name -> fragments that identify it in a long-form name.
ALIASES = {
    # England
    "Man United": ["manchester united", "man utd", "man united"],
    "Man City": ["manchester city", "man city"],
    "Nott'm Forest": ["nottingham forest", "nott'm forest", "nottm forest"],
    "Wolves": ["wolverhampton", "wolves"],
    "Tottenham": ["tottenham", "spurs"],
    "Brighton": ["brighton"],
    "Newcastle": ["newcastle"],
    "West Ham": ["west ham"],
    "West Brom": ["west bromwich", "west brom"],
    "Leeds": ["leeds"],
    "Leicester": ["leicester"],
    "Sheffield United": ["sheffield united", "sheffield utd"],
    "Sheffield Weds": ["sheffield wednesday"],
    "Ipswich": ["ipswich"],
    "Hull": ["hull city", "hull"],
    "Coventry": ["coventry"],
    "Bournemouth": ["bournemouth"],
    "QPR": ["queens park rangers", "qpr"],
    "Norwich": ["norwich"],
    "Luton": ["luton"],
    "Middlesbrough": ["middlesbrough"],
    "Stoke": ["stoke"],
    # Germany
    "Bayern Munich": ["bayern"],
    "Dortmund": ["borussia dortmund", "dortmund"],
    "M'gladbach": ["monchengladbach", "m'gladbach", "gladbach"],
    "Ein Frankfurt": ["eintracht frankfurt", "ein frankfurt", "frankfurt"],
    "FC Koln": ["koln", "cologne"],
    "Union Berlin": ["union berlin"],
    "Hertha": ["hertha"],
    "Mainz": ["mainz"],
    "Leverkusen": ["leverkusen"],
    "RB Leipzig": ["leipzig"],
    "Freiburg": ["freiburg"],
    "Werder Bremen": ["werder", "bremen"],
    "Hoffenheim": ["hoffenheim"],
    "Stuttgart": ["stuttgart"],
    "Augsburg": ["augsburg"],
    "Schalke 04": ["schalke"],
    "Hamburg": ["hamburger", "hamburg"],
    "Paderborn": ["paderborn"],
    "Elversberg": ["elversberg"],
    "Wolfsburg": ["wolfsburg"],
    "Bochum": ["bochum"],
    "Heidenheim": ["heidenheim"],
    "St Pauli": ["st. pauli", "st pauli", "pauli"],
    "Holstein Kiel": ["holstein kiel", "kiel"],
    "Darmstadt": ["darmstadt"],
    "Greuther Furth": ["greuther", "furth"],
    "Hannover": ["hannover"],
    "Nurnberg": ["nurnberg", "nuremberg"],
    "Karlsruhe": ["karlsruhe"],
    "Kaiserslautern": ["kaiserslautern"],
    "Fortuna Dusseldorf": ["dusseldorf"],
    "Bielefeld": ["bielefeld"],
    # Spain
    "Ath Bilbao": ["athletic club", "athletic bilbao", "ath bilbao"],
    "Ath Madrid": ["atletico", "ath madrid"],
    "Real Madrid": ["real madrid"],
    "Vallecano": ["rayo vallecano", "vallecano"],
    "Betis": ["betis"],
    "Sociedad": ["real sociedad", "sociedad"],
    "Espanol": ["espanyol de barcelona", "espanyol", "espanol"],
    "Celta": ["celta"],
    "La Coruna": ["deportivo la coruna", "la coruna"],
    "Santander": ["racing club de santander", "racing de santander", "santander"],
    "Osasuna": ["osasuna"],
    "Alaves": ["alaves"],
    "Elche": ["elche"],
    "Barcelona": ["fc barcelona", "barcelona"],
    "Getafe": ["getafe"],
    "Levante": ["levante"],
    "Malaga": ["malaga"],
    "Sevilla": ["sevilla"],
    "Valencia": ["valencia"],
    "Villarreal": ["villarreal"],
    "Girona": ["girona"],
    "Mallorca": ["mallorca"],
    "Las Palmas": ["las palmas"],
    "Leganes": ["leganes"],
    "Valladolid": ["valladolid"],
    "Oviedo": ["oviedo"],
    "Cadiz": ["cadiz"],
    "Almeria": ["almeria"],
    "Granada": ["granada"],
    "Sp Gijon": ["sporting gijon", "gijon"],
    "Zaragoza": ["zaragoza"],
    # Italy
    "Milan": ["ac milan", "milan"],
    "Inter": ["internazionale", "inter"],
    "Roma": ["as roma", "roma"],
    "Lazio": ["lazio"],
    "Napoli": ["napoli"],
    "Juventus": ["juventus"],
    "Atalanta": ["atalanta"],
    "Fiorentina": ["fiorentina"],
    "Bologna": ["bologna"],
    "Torino": ["torino"],
    "Udinese": ["udinese"],
    "Genoa": ["genoa"],
    "Sampdoria": ["sampdoria"],
    "Cagliari": ["cagliari"],
    "Como": ["como"],
    "Lecce": ["lecce"],
    "Sassuolo": ["sassuolo"],
    "Parma": ["parma"],
    "Verona": ["hellas verona", "verona"],
    "Empoli": ["empoli"],
    "Monza": ["monza"],
    "Venezia": ["venezia"],
    "Frosinone": ["frosinone"],
    "Salernitana": ["salernitana"],
    "Cremonese": ["cremonese"],
    "Pisa": ["pisa"],
    "Spezia": ["spezia"],
    "Palermo": ["palermo"],
    "Bari": ["bari"],
    # France
    "Paris SG": ["paris saint-germain", "paris saint germain", "paris sg", "psg"],
    "Paris FC": ["paris fc"],
    "Marseille": ["marseille"],
    "Lyon": ["lyonnais", "lyon"],
    "Lens": ["lens"],
    "Lille": ["lille"],
    "Monaco": ["monaco"],
    "Nice": ["nice"],
    "Rennes": ["rennais", "rennes"],
    "Brest": ["brestois", "brest"],
    "Strasbourg": ["strasbourg"],
    "Toulouse": ["toulouse"],
    "Nantes": ["nantes"],
    "Montpellier": ["montpellier"],
    "Reims": ["reims"],
    "St Etienne": ["saint-etienne", "saint etienne", "st etienne"],
    "Auxerre": ["auxerre"],
    "Angers": ["angers"],
    "Le Havre": ["le havre", "havre"],
    "Lorient": ["lorient"],
    "Metz": ["metz"],
    "Troyes": ["troyes"],
    "Le Mans": ["le mans"],
    "Clermont": ["clermont"],
    "Ajaccio": ["ajaccio"],
    "Guingamp": ["guingamp"],
    "Caen": ["caen"],
    "Bordeaux": ["bordeaux"],
}


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = s.lower().replace("&", " and ")
    return re.sub(r"\s+", " ", s).strip()


def _fragments(fd_name: str):
    return ALIASES.get(fd_name, []) + [norm(fd_name)]


def match_name(raw: str, candidates) -> str | None:
    """Return the football-data name among `candidates` that `raw` refers to."""
    n = " " + norm(raw) + " "
    best, best_len, tie = None, 0, False
    for cand in candidates:
        for frag in _fragments(cand):
            # whole-word match so "inter" does not hit "internacional" etc.
            if re.search(r"(?<![a-z])" + re.escape(frag) + r"(?![a-z])", n):
                if len(frag) > best_len:
                    best, best_len, tie = cand, len(frag), False
                elif len(frag) == best_len and cand != best:
                    tie = True
    return None if tie else best


def build_map(raw_names, candidates):
    """Map every raw name onto a candidate; unmatched names are left out."""
    out, missing = {}, []
    for r in sorted(set(raw_names)):
        m = match_name(r, candidates)
        if m:
            out[r] = m
        else:
            missing.append(r)
    return out, missing
