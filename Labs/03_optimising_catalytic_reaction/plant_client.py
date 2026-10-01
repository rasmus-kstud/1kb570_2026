"""Client for Laboration 3's "black-box" plant.

Fill in the three values below before you start (your instructor gives
you SERVER_URL and TOKEN at the start of the session; TEAM is your
team name as registered by the instructor).

    from plant_client import run_experiment, leaderboard

    result = run_experiment(T=[25.0, 30.0], c=[2.0, 2.5], rpm=[500.0, 500.0], name="first-try")
    result   # -> DataFrame with columns T, c, rpm, rate

Every call prints how many experiments your team has used so far --
that count is tracked on the server, not by you, so it's the number
that goes on your answer sheet.

Each batch is cached under its `name` in CACHE_FILE: calling
run_experiment again with a name that's already cached returns the saved
results without contacting the plant, so re-running a notebook doesn't
use up more experiments.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

SERVER_URL = "http://<INSTRUCTOR-IP-GOES-HERE>:5000"
TEAM = "your-team-name"
TOKEN = "your-team-token"

CACHE_FILE = Path(__file__).parent / "experiment_cache.json"


def _load_cache():
    if CACHE_FILE.exists():
        with open(CACHE_FILE) as f:
            return json.load(f)
    return {}


def run_experiment(T, c, rpm, *, name):
    """Run one batch of experiments on the plant, or load it from the cache.

    Parameters
    ----------
    T : list of float
        Temperatures (degC), one per run.
    c : list of float
        Concentrations (mM), one per run, same length as T.
    rpm : list of float
        Stirring rates, one per run, same length as T.
    name : str
        Name of this batch. If it's already in CACHE_FILE, the saved results
        are returned and no experiments are run.

    Returns
    -------
    pandas.DataFrame with columns T, c, rpm, rate.
    """
    T, c, rpm = list(T), list(c), list(rpm)

    cache = _load_cache()
    if name in cache:
        cached = pd.DataFrame(cache[name])
        if len(cached) != len(T) or not np.allclose(cached[["T", "c", "rpm"]].values,
                                                    np.column_stack([T, c, rpm])):
            raise ValueError(f"Batch '{name}' is cached with different settings -- "
                             "use a new name if the design changed.")
        print(f"'{name}': loaded {len(cached)} cached run(s), no new experiments.")
        return cached

    payload = {"team": TEAM, "token": TOKEN, "T": T, "c": c, "rpm": rpm}

    # The plant needs a brief reset between shutdown windows for the same
    # team (a real constraint on the server, not just this client) -- if two
    # calls land too close together, wait it out and retry automatically
    # rather than surfacing an error for something that isn't a mistake.
    for _ in range(5):
        resp = requests.post(f"{SERVER_URL}/experiment", json=payload, timeout=10)
        if resp.status_code == 429:
            time.sleep(0.6)
            continue
        break

    if resp.status_code != 200:
        raise RuntimeError(f"plant server rejected the request: {resp.json().get('error', resp.text)}")

    data = resp.json()
    print(f"Team '{TEAM}': {data['n_this_call']} experiment(s) this call, "
          f"{data['team_total']} total so far.")
    result = pd.DataFrame({"T": T, "c": c, "rpm": rpm, "rate": data["rate"]})

    cache[name] = result.to_dict(orient="list")
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2, default=float)
    return result


def leaderboard():
    """Fetch the current leaderboard (experiment count + best rate per team)."""
    resp = requests.get(f"{SERVER_URL}/leaderboard", timeout=10)
    resp.raise_for_status()
    return pd.DataFrame(resp.json()["leaderboard"])
