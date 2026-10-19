"""
VIDEO GAME LAUNCH STUDY (version 2)

Question: after a big video game launches, does the company's stock
usually do worse than the overall market?

What this version adds:
  1. More games from more companies (Take-Two, EA, Ubisoft)
  2. A "normal" comparison: how the stock did in ordinary periods
  3. A "luck check": how often random dates look this good or bad
  4. A flag showing whether an earnings report landed after the launch

How to run (in Terminal):
    cd ~/Downloads
    python3 launch_event_study.py
"""

import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# SETTINGS (the only part you'd ever need to edit)
# ---------------------------------------------------------------------------

# Each company's stock, and the market it should be compared against.
# US stocks are compared to SPY (the S&P 500). Ubisoft trades in Paris,
# so it is compared to the French market (^FCHI) to keep currencies matched.
COMPANIES = {
    "Take-Two":   {"stock": "TTWO",   "market": "SPY"},
    "EA":         {"stock": "EA",     "market": "SPY"},
    "Ubisoft":    {"stock": "UBI.PA", "market": "^FCHI"},
}

# Game launches: (company, game, US release date, note)
# A note marks a launch where something unrelated moved the stock.
# Launches with a note are shown but left out of the averages.
LAUNCHES = [
    ("Take-Two", "GTA IV",                 "2008-04-29", "EA takeover bid in same period"),
    ("Take-Two", "Red Dead Redemption",    "2010-05-18", ""),
    ("Take-Two", "L.A. Noire",             "2011-05-17", ""),
    ("Take-Two", "Max Payne 3",            "2012-05-15", ""),
    ("Take-Two", "GTA V",                  "2013-09-17", ""),
    ("Take-Two", "Red Dead Redemption 2",  "2018-10-26", ""),
    ("EA",       "Mass Effect 3",          "2012-03-06", ""),
    ("EA",       "Star Wars Battlefront",  "2015-11-17", ""),
    ("EA",       "Battlefield 1",          "2016-10-21", ""),
    ("EA",       "Star Wars Battlefront II", "2017-11-17", ""),
    ("EA",       "Battlefield V",          "2018-11-20", ""),
    ("EA",       "Anthem",                 "2019-02-22", ""),
    ("EA",       "Battlefield 2042",       "2021-11-19", ""),
    ("Ubisoft",  "Watch Dogs",             "2014-05-27", ""),
    ("Ubisoft",  "Assassin's Creed Unity", "2014-11-11", ""),
    ("Ubisoft",  "Assassin's Creed Origins", "2017-10-27", ""),
    ("Ubisoft",  "Assassin's Creed Odyssey", "2018-10-05", ""),
    ("Ubisoft",  "Assassin's Creed Valhalla", "2020-11-10", ""),
]

# Time windows, counted in trading days. Day 0 = launch day.
WINDOWS = {
    "Before (-60 to -1)":   (-60, -1),
    "Launch week (0 to 5)": (0, 5),
    "Month after (0 to 20)": (0, 20),
    "3 months after (0 to 60)": (0, 60),
}

LUCK_CHECK_ROUNDS = 5000
OUT_DIR = Path(__file__).resolve().parent
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# STEP 1: download prices
# ---------------------------------------------------------------------------

_price_cache = {}
_earnings_problem = []


def fetch_one(ticker):
    """
    Download one ticker's daily closing prices. Tries up to 3 times,
    because Yahoo sometimes refuses requests that come in too fast.
    """
    if ticker in _price_cache:
        return _price_cache[ticker]
    reason = "no data returned"
    for attempt in range(3):
        try:
            df = yf.download(ticker, start="2007-01-01", auto_adjust=True,
                             progress=False)
            if df is not None and not df.empty:
                close = df["Close"]
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]
                close = close.dropna()
                if len(close) > 300:
                    _price_cache[ticker] = close
                    return close
        except Exception as e:
            reason = str(e)[:80]
        time.sleep(3)
    print(f"  Could not download {ticker} ({reason})")
    _price_cache[ticker] = None
    return None


def download_prices(stock, market):
    """Daily closing prices for one stock and its market, lined up by date."""
    s, m = fetch_one(stock), fetch_one(market)
    if s is None or m is None:
        return None
    prices = pd.concat([s, m], axis=1, keys=["stock", "market"]).dropna()
    return prices if len(prices) > 300 else None


def download_earnings_dates(stock):
    """Past earnings report dates, if Yahoo has them. Returns [] if not."""
    try:
        df = yf.Ticker(stock).get_earnings_dates(limit=100)
        if df is None or df.empty:
            return []
        return sorted(pd.to_datetime(df.index).tz_localize(None).normalize())
    except Exception as e:
        if not _earnings_problem:
            _earnings_problem.append(str(e)[:120])
        return []


# ---------------------------------------------------------------------------
# STEP 2: measure each window
# ---------------------------------------------------------------------------

def excess_series(prices, start, end):
    """
    For every possible launch day, the stock's return minus the market's
    return over the window. Position i = 'if launch were on day i'.
    """
    s, m = prices["stock"].values, prices["market"].values
    n = len(s)
    out = np.full(n, np.nan)
    for i in range(n):
        a, b = i + start - 1, i + end
        if a >= 0 and b < n:
            out[i] = (s[b] / s[a] - 1) - (m[b] / m[a] - 1)
    return out


def launch_position(prices, date):
    pos = prices.index.searchsorted(pd.Timestamp(date))
    return pos if pos < len(prices) else None


def earnings_in_window(prices, pos, earnings, days=60):
    if not earnings:
        return "unknown"
    end_pos = min(pos + days, len(prices) - 1)
    start, end = prices.index[pos], prices.index[end_pos]
    return "yes" if any(start <= d <= end for d in earnings) else "no"


# ---------------------------------------------------------------------------
# STEP 3: run everything
# ---------------------------------------------------------------------------

def run(download=download_prices, get_earnings=download_earnings_dates):
    rng = np.random.default_rng(RANDOM_SEED)
    data, skipped = {}, []

    for company, info in COMPANIES.items():
        prices = download(info["stock"], info["market"])
        if prices is None:
            skipped.append(f"{company} ({info['stock']})")
            continue
        series = {w: excess_series(prices, *r) for w, r in WINDOWS.items()}
        data[company] = {"prices": prices, "series": series,
                         "earnings": get_earnings(info["stock"])}

    rows = []
    for company, game, date, note in LAUNCHES:
        if company not in data:
            continue
        d = data[company]
        pos = launch_position(d["prices"], date)
        if pos is None:
            continue
        row = {"Company": company, "Game": game, "Date": date,
               "Note": note, "Used": note == "", "_pos": pos}
        for w in WINDOWS:
            row[w] = d["series"][w][pos]
        row["Earnings within 3 months"] = earnings_in_window(
            d["prices"], pos, d["earnings"])
        rows.append(row)

    results = pd.DataFrame(rows)
    return data, results, skipped, rng


def normal_periods(data, companies, used):
    """Ordinary (non-launch) windows for the same companies and years."""
    pools = {w: [] for w in WINDOWS}
    for company in companies:
        d = data[company]
        positions = used[used["Company"] == company]["_pos"]
        if positions.empty:
            continue
        lo = max(positions.min() - 250, 0)
        hi = min(positions.max() + 250, len(d["prices"]))
        for w in WINDOWS:
            vals = d["series"][w][lo:hi]
            pools[w].extend(vals[~np.isnan(vals)])
    return {w: np.array(v) for w, v in pools.items()}


def luck_check(data, used, window, rng):
    """
    Pick random 'fake launch dates' (same number per company, same years)
    thousands of times. Returns the % of tries that look as extreme as
    the real launches. Small % = less likely to be luck.
    """
    actual = used[window].mean()
    sims = np.zeros(LUCK_CHECK_ROUNDS)
    for company, group in used.groupby("Company"):
        d = data[company]
        series = d["series"][window]
        lo = max(group["_pos"].min() - 250, 0)
        hi = min(group["_pos"].max() + 250, len(series))
        valid = np.arange(lo, hi)
        valid = valid[~np.isnan(series[valid])]
        picks = rng.choice(valid, size=(LUCK_CHECK_ROUNDS, len(group)))
        sims += series[picks].sum(axis=1)
    sims /= len(used)
    if actual < 0:
        return (sims <= actual).mean()
    return (sims >= actual).mean()


# ---------------------------------------------------------------------------
# STEP 4: print results in plain English
# ---------------------------------------------------------------------------

def pct(x):
    return "n/a" if pd.isna(x) else f"{x:+.1%}"


def print_group(title, data, used, rng):
    if used.empty:
        return
    companies = used["Company"].unique()
    normal = normal_periods(data, companies, used)
    print(f"\n{title}  ({len(used)} launches)")
    print("=" * 96)
    print(f"{'Window':<26}{'Launch avg':>12}{'Normal avg':>12}"
          f"{'Launches beat mkt':>20}{'Normal beat mkt':>17}{'Luck check':>12}")
    print("-" * 96)
    for w in WINDOWS:
        col = used[w].dropna()
        nrm = normal[w]
        luck = luck_check(data, used.dropna(subset=[w]), w, rng)
        print(f"{w:<26}{pct(col.mean()):>12}{pct(nrm.mean()):>12}"
              f"{f'{(col > 0).sum()}/{len(col)}':>20}"
              f"{f'{(nrm > 0).mean():.0%}':>17}{f'{luck:.1%}':>12}")


def print_launch_table(results):
    print("\nEVERY LAUNCH (stock minus market)")
    print("=" * 96)
    table = results[["Company", "Game", "Date"] + list(WINDOWS)
                    + ["Earnings within 3 months", "Note"]].copy()
    for w in WINDOWS:
        table[w] = table[w].map(pct)
    table.columns = ["Company", "Game", "Date", "Before", "Launch wk",
                     "Month", "3 months", "Earnings?", "Note"]
    print(table.to_string(index=False))


def print_guide():
    print("""
HOW TO READ THIS
- Launch avg: how the stock did vs. the market after launches.
- Normal avg: how the same stock did vs. the market in ordinary periods.
  If these two are close, launches are not special.
- Launches beat mkt: how many launches beat the market (e.g. 2/20).
- Normal beat mkt: how often ordinary periods beat the market.
- Luck check: how often RANDOM dates look this extreme.
  Under 5% = probably a real pattern. Over 20% = could easily be luck.
""")


def plot_paths(data, used, out_file):
    days = np.arange(-60, 61)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    groups = {"All companies": used,
              "Take-Two only": used[used["Company"] == "Take-Two"]}
    colors = {"All companies": "#1f4e79", "Take-Two only": "#c0392b"}
    for label, group in groups.items():
        paths = []
        for _, r in group.iterrows():
            p = data[r["Company"]]["prices"]
            base = r["_pos"] - 61
            if base < 0 or r["_pos"] + 60 >= len(p):
                continue
            seg = p.iloc[base + 1: r["_pos"] + 61]
            rel = seg / p.iloc[base] - 1
            paths.append((rel["stock"] - rel["market"]).values)
        if paths:
            ax.plot(days, np.mean(paths, axis=0) * 100, lw=2.4,
                    color=colors[label], label=f"{label} ({len(paths)} launches)")
    ax.axvline(0, color="black", ls="--", lw=1, label="Launch day")
    ax.axhline(0, color="gray", lw=0.6)
    ax.set_xlabel("Trading days from launch")
    ax.set_ylabel("Stock minus market (%)")
    ax.set_title("Game publisher stocks vs. the market around big launches")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_file, dpi=150)
    plt.close(fig)


def main(**kwargs):
    data, results, skipped, rng = run(**kwargs)
    if results.empty:
        raise SystemExit("No data downloaded. Check your internet connection.")

    used = results[results["Used"]]
    print_guide()
    print_group("ALL COMPANIES", data, used, rng)
    print_group("TAKE-TWO ONLY", data, used[used["Company"] == "Take-Two"], rng)
    print_launch_table(results)

    if skipped:
        print("\nSkipped (no price data available): " + ", ".join(skipped))
    if (results["Earnings within 3 months"] == "unknown").any():
        print("Some earnings dates were not available from Yahoo (shown as 'unknown').")
        if _earnings_problem:
            print("  Reason: " + _earnings_problem[0])
    print("Launches with a Note are shown but left out of the averages.")

    results.drop(columns=["_pos"]).to_csv(OUT_DIR / "launch_study_results.csv", index=False)
    plot_paths(data, used, OUT_DIR / "launch_study_chart.png")
    print(f"\nSaved: {OUT_DIR / 'launch_study_results.csv'}")
    print(f"Saved: {OUT_DIR / 'launch_study_chart.png'}")


if __name__ == "__main__":
    main()
