"""
FUND STRATEGIES: BACKTEST

Tests two bot strategies on big, popular funds (no single stocks):

  1. MONTHLY MOMENTUM
     Once a month, hold the 3 funds that rose the most over the past year.
     Any of those 3 that isn't beating cash gets swapped for bonds.

  2. BUY THE DIP
     Every day, look for stock funds in a long-term uptrend that just had
     a sharp 2-day drop. Buy, then sell on the bounce (or after 10 days).
     At most 3 funds at once.

Both are compared to simply holding the S&P 500 (SPY) and to a cautious
60% stocks / 40% bonds mix.

How to run (in Terminal):
    cd ~/Downloads
    python3 fund_strategies_backtest.py
"""

import pickle
import time
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# THE RULES (all decided before testing, not tuned to the results)
# ---------------------------------------------------------------------------
STARTING_MONEY = 2000
TRADING_COST = 0.0005       # 0.05% per buy and per sell (big funds trade cheaply)

FUNDS = {
    "SPY": "US large companies", "QQQ": "Nasdaq 100 (tech heavy)",
    "IWM": "US small companies", "EFA": "International developed",
    "EEM": "Emerging markets", "XLK": "Tech sector", "XLF": "Financials sector",
    "XLE": "Energy sector", "XLV": "Healthcare sector", "XLI": "Industrials sector",
    "XLP": "Consumer staples", "XLY": "Consumer discretionary",
    "XLU": "Utilities sector", "TLT": "Long-term US bonds", "GLD": "Gold",
    "VNQ": "Real estate",
}
BONDS = "AGG"               # where momentum goes when things look weak
CASH = "SHY"                # short-term Treasuries, stands in for cash interest

# Momentum settings
MOMENTUM_HOLD = 3           # hold the top 3 funds
LOOKBACK_DAYS = 252         # 12 months of trading days

# Buy-the-dip settings (stock funds only; bonds and gold behave differently)
DIP_FUNDS = [f for f in FUNDS if f not in ("TLT", "GLD")]
DIP_MAX_HOLD = 3            # at most 3 funds at once
DIP_TREND_DAYS = 200        # only buy above the 200-day average (uptrend)
DIP_RSI_BELOW = 10          # "sharp drop" = 2-day RSI under 10
DIP_EXIT_AVG = 5            # sell when price closes above its 5-day average
DIP_MAX_DAYS = 10           # or after 10 trading days, whichever comes first

OUT_DIR = Path(__file__).resolve().parent
DATA_FILE = OUT_DIR / "fund_data_cache.pkl"


# ---------------------------------------------------------------------------
# STEP 1: DOWNLOAD DAILY PRICES
# ---------------------------------------------------------------------------

def fetch_one(ticker):
    for attempt in range(3):
        try:
            df = yf.download(ticker, start="2004-01-01", auto_adjust=True,
                             progress=False)
            if df is not None and not df.empty:
                out = {}
                for f in ("Open", "Close"):
                    s = df[f]
                    if isinstance(s, pd.DataFrame):
                        s = s.iloc[:, 0]
                    out[f] = s
                return out
        except Exception:
            pass
        time.sleep(3)
    return None


def download_all(tickers):
    opens, closes = {}, {}
    for n, t in enumerate(tickers, 1):
        print(f"  Downloading {t} ({n} of {len(tickers)})...")
        got = fetch_one(t)
        if got is None:
            raise SystemExit(f"Could not download {t}. Check your internet and run again.")
        opens[t], closes[t] = got["Open"], got["Close"]
        time.sleep(1)
    o, c = pd.DataFrame(opens), pd.DataFrame(closes)
    for df in (o, c):
        if getattr(df.index, "tz", None) is not None:
            df.index = df.index.tz_localize(None)
    return {"Open": o.sort_index(), "Close": c.sort_index()}


def load_data(download=download_all):
    if DATA_FILE.exists():
        with open(DATA_FILE, "rb") as f:
            saved = pickle.load(f)
        if saved.get("day") == date.today().isoformat():
            print("Using data already downloaded today.")
            return saved["data"]
    print("Downloading daily prices for 18 funds (about a minute)...")
    data = download(list(FUNDS) + [BONDS, CASH])
    with open(DATA_FILE, "wb") as f:
        pickle.dump({"day": date.today().isoformat(), "data": data}, f)
    return data


# ---------------------------------------------------------------------------
# STEP 2: STRATEGY 1, MONTHLY MOMENTUM
# ---------------------------------------------------------------------------

def momentum_backtest(data, start_i):
    o, c = data["Open"], data["Close"]
    days = c.index
    past_year = c / c.shift(LOOKBACK_DAYS) - 1
    month_end = pd.Series(days.month, index=days).shift(-1) != days.month

    cash = float(STARTING_MONEY)
    shares = {}
    pending, values, log = None, {}, []

    for i in range(start_i, len(days)):
        # Carry out last month-end's decision at today's open
        if pending is not None:
            total = cash + sum(n * o[t].iloc[i] for t, n in shares.items())
            new_shares, traded = {}, 0.0
            for t, w in pending.items():
                new_shares[t] = total * w / o[t].iloc[i]
            for t in set(shares) | set(new_shares):
                diff = new_shares.get(t, 0) - shares.get(t, 0)
                traded += abs(diff) * o[t].iloc[i]
            shares = new_shares
            cash = total - sum(n * o[t].iloc[i] for t, n in shares.items())
            cash -= traded * TRADING_COST
            pending = None

        if i > 0:
            cash *= c[CASH].iloc[i] / c[CASH].iloc[i - 1]
        values[days[i]] = cash + sum(n * c[t].iloc[i] for t, n in shares.items())

        # Month-end decision, executed at the next day's open
        if (month_end.iloc[i] or i == start_i) and i + 1 < len(days):
            ranks = past_year[list(FUNDS)].iloc[i].dropna().sort_values(ascending=False)
            top = list(ranks.index[:MOMENTUM_HOLD])
            cash_return = past_year[CASH].iloc[i]
            picks = [t if ranks[t] > cash_return else BONDS for t in top]
            weights = {}
            for t in picks:
                weights[t] = weights.get(t, 0) + 1 / len(picks)
            pending = weights
            log.append({"Month": days[i].date(), "Holds": ", ".join(picks)})

    return pd.Series(values), pd.DataFrame(log)


# ---------------------------------------------------------------------------
# STEP 3: STRATEGY 2, BUY THE DIP
# ---------------------------------------------------------------------------

def rsi2(close):
    change = close.diff()
    up = change.clip(lower=0).ewm(alpha=1 / 2, adjust=False).mean()
    down = (-change.clip(upper=0)).ewm(alpha=1 / 2, adjust=False).mean()
    return 100 - 100 / (1 + up / down.replace(0, np.nan))


def dip_backtest(data, start_i):
    o, c = data["Open"], data["Close"]
    days = c.index
    trend = c[DIP_FUNDS].rolling(DIP_TREND_DAYS).mean()
    exit_avg = c[DIP_FUNDS].rolling(DIP_EXIT_AVG).mean()
    rsi = c[DIP_FUNDS].apply(rsi2)

    cash = float(STARTING_MONEY)
    held, to_sell, to_buy = {}, [], []
    values, trades, days_invested = {}, [], 0

    for i in range(start_i, len(days)):
        # Morning: carry out yesterday's decisions at today's open
        for t in to_sell:
            p = held.pop(t)
            price = o[t].iloc[i]
            cash += p["shares"] * price * (1 - TRADING_COST)
            trades.append({"Fund": t, "Bought": days[p["i"]].date(), "Sold": days[i].date(),
                           "Return": price * (1 - TRADING_COST) / (p["price"] * (1 + TRADING_COST)) - 1,
                           "Days held": i - p["i"]})
        if to_buy:
            total = cash + sum(p["shares"] * o[t].iloc[i] for t, p in held.items())
            for t in to_buy:
                spend = min(total / DIP_MAX_HOLD, cash)
                if spend < 1:
                    break
                price = o[t].iloc[i]
                held[t] = {"shares": spend / (price * (1 + TRADING_COST)), "price": price, "i": i}
                cash -= spend
        to_sell, to_buy = [], []

        if i > 0:
            cash *= c[CASH].iloc[i] / c[CASH].iloc[i - 1]
        values[days[i]] = cash + sum(p["shares"] * c[t].iloc[i] for t, p in held.items())
        days_invested += bool(held)

        if i + 1 >= len(days):
            break

        # Evening: decide what to do tomorrow, using today's close
        for t, p in held.items():
            if c[t].iloc[i] > exit_avg[t].iloc[i] or i - p["i"] >= DIP_MAX_DAYS:
                to_sell.append(t)
        open_slots = DIP_MAX_HOLD - (len(held) - len(to_sell))
        if open_slots > 0:
            today = rsi.iloc[i]
            ok = (c[DIP_FUNDS].iloc[i] > trend.iloc[i]) & (today < DIP_RSI_BELOW)
            candidates = today[ok].drop([t for t in held if t in today.index], errors="ignore")
            to_buy = list(candidates.sort_values().index[:open_slots])

    current = [{"Fund": t, "Bought": days[p["i"]].date(),
                "Buy price": round(p["price"], 2), "Latest": round(c[t].iloc[-1], 2)}
               for t, p in held.items()]
    stats = {"days_invested": days_invested / (len(days) - start_i)}
    return pd.Series(values), pd.DataFrame(trades), pd.DataFrame(current), stats


# ---------------------------------------------------------------------------
# STEP 4: RESULTS IN PLAIN ENGLISH
# ---------------------------------------------------------------------------

def scorecard(v):
    years = (v.index[-1] - v.index[0]).days / 365.25
    yearly = v.groupby(v.index.year).last()
    prev = pd.concat([pd.Series([v.iloc[0]], index=[yearly.index[0] - 1]), yearly.iloc[:-1]])
    return {
        "end": v.iloc[-1],
        "cagr": (v.iloc[-1] / v.iloc[0]) ** (1 / years) - 1,
        "drop": (v / v.cummax() - 1).min(),
        "worst_year": (yearly.values / prev.values - 1).min(),
        "yearly": pd.Series(yearly.values / prev.values - 1, index=yearly.index),
    }


def pct(x):
    return "n/a" if pd.isna(x) else f"{x:+.1%}"


def main(**kwargs):
    data = load_data(**kwargs)
    c = data["Close"]
    needed = list(FUNDS) + [BONDS, CASH]
    have_year = c[needed].notna().rolling(LOOKBACK_DAYS + 1).sum().eq(LOOKBACK_DAYS + 1).all(axis=1)
    start_i = int(np.argmax(have_year.values))
    if not have_year.iloc[start_i]:
        raise SystemExit("Not enough history in the data. Screenshot this and send it to Claude.")
    data = {k: v.ffill() for k, v in data.items()}
    c = data["Close"]
    print(f"Testing from {c.index[start_i].date()} to {c.index[-1].date()}...")

    mom, mom_log = momentum_backtest(data, start_i)
    dip, dip_trades, dip_now, dip_stats = dip_backtest(data, start_i)

    period = c.index[start_i:]
    spy = c["SPY"].loc[period]
    mix_daily = 0.6 * c["SPY"].pct_change() + 0.4 * c[BONDS].pct_change()
    mix = (1 + mix_daily.loc[period].fillna(0)).cumprod()

    lines = {
        "Monthly momentum": mom,
        "Buy the dip": dip,
        "Hold S&P 500 (SPY)": STARTING_MONEY * spy / spy.iloc[0],
        "60% stocks / 40% bonds": STARTING_MONEY * mix / mix.iloc[0],
    }
    cards = {k: scorecard(v) for k, v in lines.items()}

    print(f"\nRESULTS: ${STARTING_MONEY:,} invested on {period[0].date()}, tracked to {period[-1].date()}")
    print("=" * 92)
    print(f"{'':<26}{'Ending value':>15}{'Avg per year':>15}{'Worst drop':>14}{'Worst year':>14}")
    print("-" * 92)
    for k, s in cards.items():
        end_value = "${:,.0f}".format(s["end"])
        print(f"{k:<26}{end_value:>15}"
              f"{pct(s['cagr']):>15}{pct(s['drop']):>14}{pct(s['worst_year']):>14}")

    print("\nYEAR BY YEAR")
    print("-" * 74)
    print(f"{'Year':<8}{'Momentum':>14}{'Buy the dip':>14}{'SPY':>14}{'60/40':>14}")
    for y in cards["Monthly momentum"]["yearly"].index:
        if (period.year == y).sum() < 20:
            continue
        row = [cards[k]["yearly"].get(y, np.nan) for k in lines]
        print(f"{y:<8}" + "".join(f"{pct(x):>14}" for x in row))

    half = period[len(period) // 2]
    print("\nFIRST HALF vs SECOND HALF (does it hold up over time?)")
    print("-" * 74)
    for label, part in (("First half", period[period < half]), ("Second half", period[period >= half])):
        parts = "   ".join(f"{name} {pct(scorecard(lines[k].loc[part])['cagr'])}/yr"
                           for k, name in zip(lines, ["momentum", "dip", "SPY", "60/40"]))
        print(f"{label:<12}{part[0].date()} to {part[-1].date()}   {parts}")

    print("\nMONTHLY MOMENTUM DETAILS")
    print("-" * 74)
    in_bonds = mom_log["Holds"].str.contains(BONDS).mean()
    changes = (mom_log["Holds"] != mom_log["Holds"].shift()).sum() - 1
    print(f"Months with at least one slot in bonds: {in_bonds:.0%}")
    print(f"Months where holdings changed:          {changes} of {len(mom_log)}")
    print(f"Latest month-end pick ({mom_log['Month'].iloc[-1]}): {mom_log['Holds'].iloc[-1]}")

    print("\nBUY THE DIP DETAILS")
    print("-" * 74)
    if dip_trades.empty:
        print("No trades. That's almost certainly a data problem. Screenshot this and send it to Claude.")
    else:
        wins = dip_trades[dip_trades["Return"] > 0]
        losses = dip_trades[dip_trades["Return"] <= 0]
        years = len(period) / 252
        print(f"Total trades:        {len(dip_trades)}  (about {len(dip_trades) / years:.0f} per year)")
        print(f"Winners:             {len(wins) / len(dip_trades):.0%}")
        print(f"Average winner:      {pct(wins['Return'].mean())}")
        print(f"Average loser:       {pct(losses['Return'].mean())}")
        print(f"Average time held:   {dip_trades['Days held'].mean():.1f} trading days")
        print(f"Time invested:       {dip_stats['days_invested']:.0%} of days (rest earns cash interest)")
    print("Holding right now:   " + (", ".join(dip_now["Fund"]) if not dip_now.empty else "nothing (in cash)"))

    print("""
HOW TO READ THIS
- Avg per year: average yearly growth. Compare each strategy to SPY.
- Worst drop: the biggest fall from a high point. Smaller is safer.
- 60/40 is the classic cautious mix. If a strategy can't beat 60/40 on
  growth OR on safety, it isn't earning its complexity.
- Unlike the stock tests, these funds never disappear, so there is no
  "missing losers" problem. These results are not tilted toward the bot.
- Trading costs: 0.05% per buy and sell. Taxes are NOT included.
  Frequent trading in a regular (non-retirement) account would cost more.
""")

    pd.DataFrame(lines).to_csv(OUT_DIR / "fund_strategies_daily_values.csv")
    dip_trades.to_csv(OUT_DIR / "fund_dip_trades.csv", index=False)
    mom_log.to_csv(OUT_DIR / "fund_momentum_picks.csv", index=False)
    save_chart(lines, OUT_DIR / "fund_strategies_chart.png")
    print(f"Saved chart: {OUT_DIR / 'fund_strategies_chart.png'}")
    print(f"Saved monthly picks: {OUT_DIR / 'fund_momentum_picks.csv'}")


def save_chart(lines, out_file):
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                                      gridspec_kw={"height_ratios": [3, 1.3]})
    colors = {"Monthly momentum": "#1f4e79", "Buy the dip": "#27ae60",
              "Hold S&P 500 (SPY)": "#7f8c8d", "60% stocks / 40% bonds": "#c0392b"}
    for k, v in lines.items():
        top.plot(v.index, v, lw=1.8, color=colors[k], label=k)
        bottom.plot(v.index, (v / v.cummax() - 1) * 100, lw=1.1, color=colors[k])
    top.set_ylabel("Value of $2,000 ($)")
    top.set_title("Fund strategies vs. holding the market")
    top.legend(frameon=False)
    top.grid(alpha=0.25)
    bottom.set_ylabel("Drop from peak (%)")
    bottom.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_file, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
