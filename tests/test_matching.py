import pandas as pd
import pytest

import build_dataset as bd


@pytest.fixture
def match_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(bd, "DATA", tmp_path)
    pd.DataFrame([
        {"season": 2025, "round": 7, "race_name": "Emilia Romagna Grand Prix", "date": "2025-05-18",
         "driver_id": "russell", "family_name": "Russell"},
        {"season": 2025, "round": 16, "race_name": "Italian Grand Prix", "date": "2025-09-07",
         "driver_id": "perez", "family_name": "Pérez"},
        {"season": 2025, "round": 21, "race_name": "São Paulo Grand Prix", "date": "2025-11-09",
         "driver_id": "max_verstappen", "family_name": "Verstappen"},
    ]).to_csv(tmp_path / "results.csv", index=False)
    pd.DataFrame([
        # an "Italy" market in May means Imola, not Monza
        {"event_id": 1, "title": "Italy Grand Prix Winner",
         "start": "2025-05-05T00:00:00Z", "end": "2025-05-19T12:00:00Z"},
        # misspelled title, and an end date a week after the race
        {"event_id": 2, "title": "Brazlian Grand Prix Winner",
         "start": "2025-10-31T00:00:00Z", "end": "2025-11-16T17:00:00Z"},
        {"event_id": 3, "title": "Italian Grand Prix: Pole Winner",
         "start": "2025-09-01T00:00:00Z", "end": "2025-09-07T00:00:00Z"},
    ]).to_csv(tmp_path / "polymarket_f1_events.csv", index=False)
    return tmp_path


def test_pole_markets_are_excluded(match_dir):
    titles = bd.race_winner_events()["title"].tolist()
    assert "Italian Grand Prix: Pole Winner" not in titles


def test_races_match_by_name_within_the_open_window(match_dir):
    m = bd.match_races().set_index("event_id")
    assert m.loc[1, "race_name"] == "Emilia Romagna Grand Prix"
    assert m.loc[2, "race_name"] == "São Paulo Grand Prix"


def test_driver_names_placeholders_and_typos(match_dir):
    d = bd.match_drivers(["Verstappen", "Max Verstappen", "Sergio Pérez", "George Russel",
                          "Other", "Driver A"]).set_index("pm_driver")
    assert d.loc["Verstappen", "driver_id"] == d.loc["Max Verstappen", "driver_id"] == "max_verstappen"
    assert d.loc["Sergio Pérez", "driver_id"] == "perez"
    assert d.loc["George Russel", "driver_id"] == "russell"
    assert d.loc["Other", "placeholder"] and d.loc["Driver A", "placeholder"]
