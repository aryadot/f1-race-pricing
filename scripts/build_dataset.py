"""Phase 1: pull and store everything the project needs.

1. Polymarket: every F1 race winner market (2024 to 2026), every driver's
   full hourly price history, and which driver the market resolved to.
2. Jolpica: race results and qualifying for the same seasons.
3. A match table linking each Polymarket event to its Jolpica race and each
   Polymarket driver to a Jolpica driver.

Run from the project root (it reads data/polymarket_f1_events.csv from Phase 0):
    pip install requests pandas pyarrow
    python scripts/build_dataset.py

To redo only the matching step (fast, no downloads):
    python scripts/build_dataset.py --match-only

Takes roughly 15 to 30 minutes. It caches as it goes, so if it stops,
run it again and it picks up where it left off.

Outputs in data/:
    pm_markets.csv          one row per race and driver: token, volume, resolved winner flag
    pm_prices.parquet       hourly prices: event_id, driver, ts, price
    results.csv             Jolpica race results
    qualifying.csv          Jolpica qualifying
    race_match.csv          Polymarket event to Jolpica race
    driver_match.csv        Polymarket driver name to Jolpica driver
"""

import json
import re
import time
import unicodedata
from pathlib import Path

import pandas as pd
import requests

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
JOLPICA = "https://api.jolpi.ca/ergast/f1"
DATA = Path("data")
CACHE = DATA / "pm_cache"
SEASONS = [2024, 2025, 2026]
CHUNK_DAYS = 7  # resolved markets often return nothing for long windows, so ask week by week

# race winner events only: excludes pole, sprint, margin, single driver, and IndyCar markets
RACE_WINNER = re.compile(r"grand prix(?: winner|: driver winner)", re.IGNORECASE)
EXCLUDE = re.compile(r"pole|sprint|margin|chevrolet|^will ", re.IGNORECASE)


def get(url: str, params: dict | None = None, pause: float = 0.25):
    for attempt in range(3):
        try:
            r = requests.get(url, params=params, timeout=30)
            time.sleep(pause)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:  # rate limited: back off and retry
                time.sleep(5 * (attempt + 1))
                continue
            return None
        except requests.RequestException:
            time.sleep(2)
    return None


def utc(x) -> pd.Timestamp:
    ts = pd.Timestamp(x)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def normalize(name: str) -> str:
    """Lowercase, strip accents: 'Pérez' -> 'perez'."""
    text = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    return text.lower().strip()


# ------------------------------------------------------------------ Polymarket
def race_winner_events() -> pd.DataFrame:
    ev = pd.read_csv(DATA / "polymarket_f1_events.csv")
    keep = ev["title"].str.contains(RACE_WINNER) & ~ev["title"].str.contains(EXCLUDE)
    return ev[keep].reset_index(drop=True)


def _is_one(x) -> bool:
    try:
        return float(x) == 1.0
    except (TypeError, ValueError):
        return False


def price_history(token: str, start: pd.Timestamp, end: pd.Timestamp) -> list[dict]:
    points = []
    t = start
    while t < end:
        t2 = min(t + pd.Timedelta(days=CHUNK_DAYS), end)
        res = get(f"{CLOB}/prices-history", {"market": token, "startTs": int(t.timestamp()),
                                              "endTs": int(t2.timestamp()), "fidelity": 60})
        points += (res or {}).get("history", []) if isinstance(res, dict) else []
        t = t2
    return points


def pull_polymarket() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    events = race_winner_events()
    print(f"{len(events)} race winner events")
    market_rows = []
    for i, ev in events.iterrows():
        cache_file = CACHE / f"{ev['event_id']}.json"
        if cache_file.exists():
            cached = json.loads(cache_file.read_text())
        else:
            detail = get(f"{GAMMA}/events/{ev['event_id']}")
            if not detail:
                print(f"  could not fetch event {ev['event_id']}")
                continue
            start = utc(detail.get("startDate") or ev["start"])
            end = utc(detail.get("endDate") or ev["end"]) + pd.Timedelta(days=1)
            cached = {"markets": []}
            for m in detail.get("markets", []):
                if not m.get("clobTokenIds"):
                    continue
                token = json.loads(m["clobTokenIds"])[0]
                final = json.loads(m.get("outcomePrices") or "[null, null]")
                cached["markets"].append({
                    "driver": m.get("groupItemTitle") or m.get("question"),
                    "token": token,
                    "volume": float(m.get("volumeNum") or m.get("volume") or 0),
                    "final_yes_price": final[0],
                    "history": price_history(token, start, end),
                })
            cache_file.write_text(json.dumps(cached))
        n_hist = sum(1 for m in cached["markets"] if m["history"])
        print(f"  [{i + 1}/{len(events)}] {ev['title']}: {len(cached['markets'])} drivers, "
              f"{n_hist} with price history")
        for m in cached["markets"]:
            market_rows.append({"event_id": ev["event_id"], "title": ev["title"], "race_end": ev["end"],
                                "driver": m["driver"], "token": m["token"], "volume": m["volume"],
                                "won": _is_one(m["final_yes_price"]),
                                "n_price_points": len(m["history"])})

    markets = pd.DataFrame(market_rows)
    markets.to_csv(DATA / "pm_markets.csv", index=False)

    price_rows = []
    for f in CACHE.glob("*.json"):
        for m in json.loads(f.read_text())["markets"]:
            for p in m["history"]:
                price_rows.append({"event_id": int(f.stem), "driver": m["driver"],
                                   "ts": pd.to_datetime(p["t"], unit="s", utc=True), "price": p["p"]})
    prices = pd.DataFrame(price_rows).drop_duplicates(["event_id", "driver", "ts"])
    prices.to_parquet(DATA / "pm_prices.parquet", index=False)
    print(f"saved {len(markets)} driver markets and {len(prices):,} price points")


# ------------------------------------------------------------------ Jolpica
def jolpica_table(season: int, kind: str) -> list[dict]:
    """kind is 'results' or 'qualifying'; pages through all rows."""
    rows, offset = [], 0
    while True:
        data = get(f"{JOLPICA}/{season}/{kind}.json", {"limit": 100, "offset": offset}, pause=0.4)
        if not data:
            break
        mr = data["MRData"]
        for race in mr["RaceTable"]["Races"]:
            key = "Results" if kind == "results" else "QualifyingResults"
            for r in race.get(key, []):
                d, c = r["Driver"], r["Constructor"]
                row = {"season": season, "round": int(race["round"]), "race_name": race["raceName"],
                       "circuit_id": race["Circuit"]["circuitId"], "date": race["date"],
                       "driver_id": d["driverId"], "code": d.get("code"),
                       "given_name": d["givenName"], "family_name": d["familyName"],
                       "constructor": c["constructorId"], "position": r.get("position")}
                if kind == "results":
                    row.update({"grid": r.get("grid"), "status": r.get("status"),
                                "laps": r.get("laps")})
                else:
                    row.update({"q1": r.get("Q1"), "q2": r.get("Q2"), "q3": r.get("Q3")})
                rows.append(row)
        offset += 100
        if offset >= int(mr["total"]):
            break
    return rows


def pull_jolpica() -> None:
    for kind in ["results", "qualifying"]:
        rows = []
        for s in SEASONS:
            rows += jolpica_table(s, kind)
        pd.DataFrame(rows).to_csv(DATA / f"{kind}.csv", index=False)
        print(f"saved {len(rows)} {kind} rows")


# ------------------------------------------------------------------ matching
# Polymarket end dates are resolution deadlines, often a week after the race,
# so races are matched by name within the market's open window, not by date alone.
RACE_KEYWORDS = {
    "brazil": ["sao paulo", "brazil"], "brazlian": ["sao paulo", "brazil"],
    "mexic": ["mexico"], "united states": ["united states"], "las vegas": ["las vegas"],
    "abu dhabi": ["abu dhabi"], "australia": ["australian"], "china": ["chinese"],
    "chinese": ["chinese"], "japan": ["japanese"], "bahrain": ["bahrain"],
    "saudi": ["saudi"], "miami": ["miami"], "italy": ["italian", "emilia romagna"],
    "italian": ["italian"], "monaco": ["monaco"], "catalunya": ["barcelona", "spanish"],
    "spanish": ["spanish"], "canad": ["canadian"], "austria": ["austrian"],
    "british": ["british"], "belgian": ["belgian"], "hungar": ["hungarian"],
    "dutch": ["dutch"], "azerbaijan": ["azerbaijan"], "singapore": ["singapore"],
    "qatar": ["qatar"],
}
DRIVER_ALIASES = {"george russel": "russell"}  # typo in one Polymarket market
PLACEHOLDER = re.compile(r"^(other|driver [a-z])$", re.IGNORECASE)


def match_races() -> pd.DataFrame:
    events = race_winner_events()
    results = pd.read_csv(DATA / "results.csv")
    races = results[["season", "round", "race_name", "date"]].drop_duplicates()
    races["date"] = pd.to_datetime(races["date"])
    races["name_norm"] = races["race_name"].map(normalize)

    rows = []
    for _, e in events.iterrows():
        title = normalize(e["title"])
        opens = pd.to_datetime(e["start"], format="ISO8601", utc=True).tz_localize(None).normalize()
        closes = pd.to_datetime(e["end"], format="ISO8601", utc=True).tz_localize(None).normalize()
        names = next((v for k, v in RACE_KEYWORDS.items() if k in title), [])
        in_window = races[(races["date"] >= opens - pd.Timedelta(days=1))
                          & (races["date"] <= closes + pd.Timedelta(days=1))]
        hit = in_window[in_window["name_norm"].apply(lambda n: any(x in n for x in names))]
        best = hit.iloc[0] if len(hit) == 1 else None
        rows.append({"event_id": e["event_id"], "title": e["title"],
                     "season": best["season"] if best is not None else None,
                     "round": best["round"] if best is not None else None,
                     "race_name": best["race_name"] if best is not None else None,
                     "n_candidates": len(hit)})
    return pd.DataFrame(rows)


def match_drivers(names) -> pd.DataFrame:
    results = pd.read_csv(DATA / "results.csv")
    drivers = results.drop_duplicates("driver_id")
    by_family = {normalize(r.family_name): r.driver_id for r in drivers.itertuples()}
    rows = []
    for n in names:
        key = normalize(n)
        if PLACEHOLDER.match(key):
            rows.append({"pm_driver": n, "driver_id": None, "placeholder": True})
            continue
        found = DRIVER_ALIASES.get(key)
        if found is None:
            found = next((by_family[t] for t in reversed(key.split()) if t in by_family), None)
        rows.append({"pm_driver": n, "driver_id": found, "placeholder": False})
    return pd.DataFrame(rows)


def match() -> None:
    race_match = match_races()
    race_match.to_csv(DATA / "race_match.csv", index=False)
    markets = pd.read_csv(DATA / "pm_markets.csv")
    driver_match = match_drivers(markets["driver"].dropna().unique())
    driver_match.to_csv(DATA / "driver_match.csv", index=False)

    unmatched_races = race_match[race_match["season"].isna()]["title"].tolist()
    print(f"races matched: {len(race_match) - len(unmatched_races)} of {len(race_match)}")
    if unmatched_races:
        print(f"unmatched races: {unmatched_races}")
    real = driver_match[~driver_match["placeholder"]]
    unmatched = real[real["driver_id"].isna()]["pm_driver"].tolist()
    print(f"drivers matched: {len(real) - len(unmatched)} of {len(real)} "
          f"({driver_match['placeholder'].sum()} placeholder outcomes skipped)")
    if unmatched:
        print(f"unmatched names (send me these): {unmatched}")


if __name__ == "__main__":
    import sys
    if "--match-only" in sys.argv:
        match()
        sys.exit()
    print("1. Polymarket")
    pull_polymarket()
    print("\n2. Jolpica")
    pull_jolpica()
    print("\n3. Matching")
    match()
