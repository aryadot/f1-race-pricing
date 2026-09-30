"""Phase 0: check what data actually exists before designing around it.

Answers three questions:
  1. Which F1 Grand Prix winner markets exist on Polymarket, and do they
     have usable price history? (This decides the market comparison sample.)
  2. Does FastF1 load practice, qualifying, and race laps with tyre data?
  3. Does the Jolpica API return full race results?

Run from the project root:
    pip install fastf1 requests pandas
    python scripts/check_data.py

Writes data/polymarket_f1_events.csv and prints a coverage summary.
"""

import json
import re
import time
from pathlib import Path

import pandas as pd
import requests

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
JOLPICA = "https://api.jolpi.ca/ergast/f1"
OUT = Path("data")
OUT.mkdir(exist_ok=True)

GP_PATTERN = re.compile(r"grand prix|\bgp\b", re.IGNORECASE)
WINNER_PATTERN = re.compile(r"winner|win", re.IGNORECASE)


def get(url: str, params: dict | None = None) -> dict | list | None:
    """GET with a polite pause; returns None instead of crashing on errors."""
    try:
        r = requests.get(url, params=params, timeout=30)
        time.sleep(0.2)
        if r.status_code != 200:
            print(f"  {r.status_code} from {url} {params or ''}")
            return None
        return r.json()
    except requests.RequestException as e:
        print(f"  request failed: {e}")
        return None


# ---------------------------------------------------------------- Polymarket
def find_f1_events() -> list[dict]:
    """Collect F1 events through tag filters and search, deduplicated by id."""
    found: dict[str, dict] = {}

    for slug in ["f1", "formula-1", "formula1"]:
        for closed in ["true", "false"]:
            offset = 0
            while True:
                batch = get(f"{GAMMA}/events", {"tag_slug": slug, "closed": closed,
                                                "limit": 100, "offset": offset})
                if not batch:
                    break
                for ev in batch:
                    found[str(ev["id"])] = ev
                if len(batch) < 100:
                    break
                offset += 100

    for query in ["Grand Prix winner", "F1 Grand Prix", "Formula 1 winner"]:
        res = get(f"{GAMMA}/public-search", {"q": query})
        for ev in (res or {}).get("events", []) if isinstance(res, dict) else []:
            found.setdefault(str(ev["id"]), ev)

    return [ev for ev in found.values()
            if GP_PATTERN.search(ev.get("title", "")) and WINNER_PATTERN.search(ev.get("title", ""))]


def price_history_points(token_id: str, start: str | None, end: str | None) -> int:
    """Number of price points available for one outcome token."""
    hist = get(f"{CLOB}/prices-history", {"market": token_id, "interval": "max", "fidelity": 60})
    points = (hist or {}).get("history", []) if isinstance(hist, dict) else []
    if not points and start and end:
        # resolved markets sometimes need an explicit time window
        s = int(pd.Timestamp(start).timestamp())
        e = int(pd.Timestamp(end).timestamp()) + 86400
        hist = get(f"{CLOB}/prices-history", {"market": token_id, "startTs": s,
                                               "endTs": e, "fidelity": 60})
        points = (hist or {}).get("history", []) if isinstance(hist, dict) else []
    return len(points)


def summarize_events(events: list[dict]) -> pd.DataFrame:
    rows = []
    for ev in events:
        markets = ev.get("markets", [])
        # the favourite (highest volume market) is the best test of history depth
        top = max(markets, key=lambda m: float(m.get("volume") or 0), default=None)
        n_points = 0
        if top and top.get("clobTokenIds"):
            yes_token = json.loads(top["clobTokenIds"])[0]
            n_points = price_history_points(yes_token, ev.get("startDate"), ev.get("endDate"))
        rows.append({
            "event_id": ev["id"],
            "title": ev.get("title"),
            "start": ev.get("startDate"),
            "end": ev.get("endDate"),
            "closed": ev.get("closed"),
            "volume_usd": round(float(ev.get("volume") or 0)),
            "n_outcomes": len(markets),
            "top_outcome": (top or {}).get("groupItemTitle") or (top or {}).get("question"),
            "top_history_points": n_points,
        })
    return pd.DataFrame(rows).sort_values("end")


# ---------------------------------------------------------------- FastF1
def check_fastf1(year: int = 2024, gp: str = "Bahrain") -> None:
    import fastf1

    cache = OUT / "fastf1_cache"
    cache.mkdir(exist_ok=True)
    fastf1.Cache.enable_cache(str(cache))
    for ident in ["FP2", "Q", "R"]:
        s = fastf1.get_session(year, gp, ident)
        s.load(laps=True, telemetry=False, weather=False, messages=False)
        laps = s.laps
        has_tyres = {"Compound", "TyreLife"}.issubset(laps.columns)
        print(f"  {year} {gp} {ident}: {len(laps)} laps, "
              f"{laps['Driver'].nunique()} drivers, tyre data: {has_tyres}")


# ---------------------------------------------------------------- Jolpica
def check_jolpica(year: int = 2024) -> None:
    data = get(f"{JOLPICA}/{year}/results.json", {"limit": 100})
    if not data:
        return
    total = int(data["MRData"]["total"])
    races = data["MRData"]["RaceTable"]["Races"]
    print(f"  {year}: {total} result rows available; first page covers {len(races)} races")


if __name__ == "__main__":
    print("1. Polymarket F1 Grand Prix winner markets")
    events = find_f1_events()
    table = summarize_events(events) if events else pd.DataFrame()
    if table.empty:
        print("  none found; send me this output so we can adjust the search")
    else:
        table.to_csv(OUT / "polymarket_f1_events.csv", index=False)
        usable = table[table["top_history_points"] > 20]
        print(f"  {len(table)} events found, {len(usable)} with price history")
        print(f"  date range: {table['end'].min()} to {table['end'].max()}")
        print(table[["title", "end", "volume_usd", "n_outcomes", "top_history_points"]]
              .to_string(index=False))

    print("\n2. FastF1 session data")
    check_fastf1()

    print("\n3. Jolpica results")
    check_jolpica()
