"""
WEEKLY 52-WEEK-HIGH BREAKOUT: BACKTEST (Phase 1)

What this does:
  Pretends it's every Friday from 2011 to today, applies the bot's rules
  to small-company stocks, and tracks what $2,000 would have become.
  Then it compares that to simply holding the market.

How to run (in Terminal):
    cd ~/Downloads
    python3 weekly_breakout_backtest.py          (small companies, S&P 600)
    python3 weekly_breakout_backtest.py large    (large companies, S&P 500)

The first run downloads about 600 stocks and takes a few minutes.
After that, it saves the data and reruns in seconds (same day).
"""

import io
import pickle
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import yfinance as yf

# ---------------------------------------------------------------------------
# THE RULES (all decided before testing, not tuned to the results)
# ---------------------------------------------------------------------------
STARTING_MONEY = 2000
MAX_POSITIONS = 10          # hold at most 10 stocks, equal amounts
MIN_PRICE = 5.0             # skip stocks under $5 (pump-and-dump territory)
MIN_DAILY_DOLLARS = 2_000_000   # skip stocks trading under $2M a day
NEAR_HIGH = 0.95            # only buy within 5% of the 52-week high
STOP_LOSS = 0.10            # sell if down 10% from purchase (checked weekly)
TRADING_COST = 0.0025       # 0.25% lost per buy and per sell (price gap)
START_DATE = "2010-01-01"

# Which list of companies to test. Only the stock list changes; the rules are identical.
UNIVERSES = {
    "small": {"name": "small companies (S&P 600)",
              "url": "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies",
              "compare": "IJR", "compare_name": "Hold small caps (IJR)"},
    "large": {"name": "large companies (S&P 500)",
              "url": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
              "compare": "RSP", "compare_name": "Hold equal-weight (RSP)"},
}
CHOICE = sys.argv[1].lower() if len(sys.argv) > 1 else "small"
if CHOICE not in UNIVERSES:
    raise SystemExit("Type 'small' or 'large' after the file name, or nothing for small.")
U = UNIVERSES[CHOICE]

OUT_DIR = Path(__file__).resolve().parent
DATA_FILE = OUT_DIR / f"bot_data_cache_{CHOICE}.pkl"


# ---------------------------------------------------------------------------
# STEP 1: GET THE LIST OF COMPANIES
# ---------------------------------------------------------------------------

def get_small_cap_list():
    """Current index members (S&P 600 or S&P 500), from Wikipedia."""
    url = U["url"]
    html = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30).text
    for table in pd.read_html(io.StringIO(html)):
        for col in table.columns:
            if "symbol" in str(col).lower() or "ticker" in str(col).lower():
                tickers = table[col].astype(str).str.strip().str.replace(".", "-", regex=False)
                tickers = [t for t in tickers if t and t != "nan"]
                if len(tickers) > 300:
                    return sorted(set(tickers))
    raise SystemExit("Could not read the company list from Wikipedia. Try again in a minute.")


# ---------------------------------------------------------------------------
# STEP 2: DOWNLOAD WEEKLY PRICES
# ---------------------------------------------------------------------------

def download_batch(tickers):
    for attempt in range(3):
        try:
            df = yf.download(tickers, start=START_DATE, interval="1wk",
                             auto_adjust=True, progress=False, threads=True)
            if df is not None and not df.empty:
                return df
        except Exception:
            pass
        time.sleep(5)
    return None


def download_all(tickers):
    """Weekly Open, Close and Volume for every ticker, in batches of 100."""
    fields = {"Open": [], "Close": [], "Volume": []}
    batches = [tickers[i:i + 100] for i in range(0, len(tickers), 100)]
    for n, batch in enumerate(batches, 1):
        print(f"  Downloading batch {n} of {len(batches)}...")
        df = download_batch(batch)
        if df is None:
            print(f"  Batch {n} failed, skipping those stocks.")
            continue
        for f in fields:
            part = df[f]
            if isinstance(part, pd.Series):
                part = part.to_frame(batch[0])
            fields[f].append(part)
        time.sleep(2)
    return {f: pd.concat(v, axis=1) for f, v in fields.items()}


def drop_unfinished_week(df):
    """Remove this week's bar if the week isn't over yet."""
    last_friday = date.today() - timedelta(days=(date.today().weekday() - 4) % 7)
    if date.today().weekday() == 4:
        last_friday -= timedelta(days=7)  # Friday's bar isn't final until after close
    cutoff = pd.Timestamp(last_friday) - pd.Timedelta(days=4)  # that week's Monday
    return df[df.index <= cutoff]


def load_data(get_list=get_small_cap_list, download=download_all):
    if DATA_FILE.exists():
        with open(DATA_FILE, "rb") as f:
            saved = pickle.load(f)
        if saved.get("day") == date.today().isoformat():
            print("Using data already downloaded today.")
            return saved["stocks"], saved["market"]

    print(f"Getting the list of {U['name']}...")
    tickers = get_list()
    print(f"Found {len(tickers)} companies. Downloading weekly prices (a few minutes)...")
    stocks = download(tickers)
    print(f"Downloading market benchmarks (SPY and {U['compare']})...")
    market = download(["SPY", U["compare"]])

    stocks = {f: drop_unfinished_week(v.sort_index()) for f, v in stocks.items()}
    market = {f: drop_unfinished_week(v.sort_index()) for f, v in market.items()}
    with open(DATA_FILE, "wb") as f:
        pickle.dump({"day": date.today().isoformat(), "stocks": stocks, "market": market}, f)
    return stocks, market


# ---------------------------------------------------------------------------
# STEP 3: CHART READINGS FOR EVERY STOCK, EVERY WEEK
# ---------------------------------------------------------------------------

def line_up_weeks(fields):
    """
    Yahoo sometimes labels the same week with different dates for
    different stocks. This relabels every week by its Friday so all
    stocks line up on the same rows.
    """
    idx = fields["Close"].index
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    friday = idx.to_period("W-FRI").end_time.normalize()
    out = {}
    for name, df in fields.items():
        df = df.copy()
        df.index = friday
        if name == "Close":
            out[name] = df.groupby(level=0).last()
        elif name == "Open":
            out[name] = df.groupby(level=0).first()
        else:
            out[name] = df.groupby(level=0).sum(min_count=1)
    return out


def chart_readings(stocks):
    close, vol = stocks["Close"], stocks["Volume"]
    return {
        "close": close,
        "open": stocks["Open"],
        "high52": close.rolling(52, min_periods=52).max(),
        "ma10": close.rolling(10, min_periods=10).mean(),
        "ma40": close.rolling(40, min_periods=40).mean(),
        "daily_dollars": (close * vol).rolling(10, min_periods=10).mean() / 5,
    }


# ---------------------------------------------------------------------------
# STEP 4: REPLAY EVERY WEEK
# ---------------------------------------------------------------------------

def run_backtest(r, market):
    close, opn = r["close"], r["open"]
    weeks = close.index
    spy = market["Close"]["SPY"].reindex(weeks).ffill()
    spy_ma40 = spy.rolling(40, min_periods=40).mean()

    first = r["high52"].notna().sum(axis=1).gt(50).idxmax()
    start_i = weeks.get_loc(first)

    cash, positions, trades = float(STARTING_MONEY), {}, []
    values = {}
    last_price = close.ffill()

    def value_at(i):
        return cash + sum(p["shares"] * last_price[t].iloc[i] for t, p in positions.items())

    values[weeks[start_i]] = value_at(start_i)

    for i in range(start_i, len(weeks) - 1):
        c, nxt_open = close.iloc[i], opn.iloc[i + 1]

        # --- Sell decisions (made at Friday close, executed next Monday open)
        for t in list(positions):
            p = positions[t]
            price = c[t]
            reason = None
            if pd.isna(price):
                reason = "data stopped"
            elif price < r["ma10"][t].iloc[i]:
                reason = "trend broke"
            elif price <= p["buy_price"] * (1 - STOP_LOSS):
                reason = "stop loss"
            if reason:
                fill = nxt_open[t] if pd.notna(nxt_open[t]) else last_price[t].iloc[i]
                cash += p["shares"] * fill * (1 - TRADING_COST)
                trades.append({"Stock": t, "Bought": p["date"].date(), "Buy price": p["buy_price"],
                               "Sold": weeks[i + 1].date(), "Sell price": fill,
                               "Return": fill * (1 - TRADING_COST) / (p["buy_price"] * (1 + TRADING_COST)) - 1,
                               "Weeks held": i + 1 - p["i"], "Why sold": reason})
                del positions[t]

        # --- Buy decisions (only when the overall market is in an uptrend)
        open_slots = MAX_POSITIONS - len(positions)
        market_ok = pd.notna(spy_ma40.iloc[i]) and spy.iloc[i] > spy_ma40.iloc[i]
        if open_slots > 0 and market_ok:
            ok = ((c >= MIN_PRICE)
                  & (r["daily_dollars"].iloc[i] >= MIN_DAILY_DOLLARS)
                  & (c >= NEAR_HIGH * r["high52"].iloc[i])
                  & (c > r["ma10"].iloc[i])
                  & (r["ma10"].iloc[i] > r["ma40"].iloc[i])
                  & nxt_open.notna())
            closeness = (c / r["high52"].iloc[i])[ok]
            closeness = closeness.drop([t for t in positions if t in closeness.index])
            picks = closeness.sort_values(ascending=False).index[:open_slots]
            slot_size = value_at(i) / MAX_POSITIONS
            for t in picks:
                spend = min(slot_size, cash)
                if spend < 1:
                    break
                buy_price = nxt_open[t]
                positions[t] = {"shares": spend / (buy_price * (1 + TRADING_COST)),
                                "buy_price": buy_price, "date": weeks[i + 1], "i": i + 1}
                cash -= spend

        values[weeks[i + 1]] = value_at(i + 1)

    current = []
    for t, p in positions.items():
        current.append({"Stock": t, "Bought": p["date"].date(),
                        "Buy price": round(p["buy_price"], 2),
                        "Latest price": round(last_price[t].iloc[-1], 2)})
    return pd.Series(values), pd.DataFrame(trades), pd.DataFrame(current)


# ---------------------------------------------------------------------------
# STEP 5: SCORECARD
# ---------------------------------------------------------------------------

def scorecard(v):
    years = (v.index[-1] - v.index[0]).days / 365.25
    yearly = v.groupby(v.index.year).last()
    prev = pd.concat([pd.Series([v.iloc[0]], index=[yearly.index[0] - 1]), yearly.iloc[:-1]])
    yearly_ret = yearly.values / prev.values - 1
    return {
        "Ending value": v.iloc[-1],
        "Avg yearly return": (v.iloc[-1] / v.iloc[0]) ** (1 / years) - 1,
        "Worst drop from a peak": (v / v.cummax() - 1).min(),
        "Worst calendar year": yearly_ret.min(),
        "yearly": pd.Series(yearly_ret, index=yearly.index),
    }


def money(x):
    return f"${x:,.0f}"


def pct(x):
    return "n/a" if pd.isna(x) else f"{x:+.1%}"


def print_results(strategy, market, trades, current):
    spy = market["Close"]["SPY"].reindex(strategy.index).ffill()
    ijr = market["Close"][U["compare"]].reindex(strategy.index).ffill()
    lines = {
        "The bot": strategy,
        "Hold S&P 500 (SPY)": STARTING_MONEY * spy / spy.iloc[0],
        U["compare_name"]: STARTING_MONEY * ijr / ijr.iloc[0],
    }
    cards = {k: scorecard(v) for k, v in lines.items()}

    print(f"\nRESULTS: ${STARTING_MONEY:,} invested on {strategy.index[0].date()}, "
          f"tracked to {strategy.index[-1].date()}")
    print("=" * 92)
    print(f"{'':<26}{'Ending value':>15}{'Avg per year':>15}{'Worst drop':>14}{'Worst year':>14}")
    print("-" * 92)
    for k, s in cards.items():
        print(f"{k:<26}{money(s['Ending value']):>15}{pct(s['Avg yearly return']):>15}"
              f"{pct(s['Worst drop from a peak']):>14}{pct(s['Worst calendar year']):>14}")

    print("\nYEAR BY YEAR")
    print("-" * 60)
    print(f"{'Year':<8}{'The bot':>14}{'SPY':>14}{U['compare']:>14}")
    for y in cards["The bot"]["yearly"].index:
        if (strategy.index.year == y).sum() < 4:
            continue  # skip a year with only a week or two of data
        row = [cards[k]["yearly"].get(y, np.nan) for k in lines]
        print(f"{y:<8}" + "".join(f"{pct(x):>14}" for x in row))

    half = strategy.index[len(strategy) // 2]
    print("\nFIRST HALF vs SECOND HALF (does it hold up over time?)")
    print("-" * 60)
    for label, part in (("First half", strategy[:half]), ("Second half", strategy[half:])):
        s_bot = scorecard(part)["Avg yearly return"]
        s_spy = scorecard(lines["Hold S&P 500 (SPY)"][part.index])["Avg yearly return"]
        s_ijr = scorecard(lines[U["compare_name"]][part.index])["Avg yearly return"]
        print(f"{label:<14}{part.index[0].date()} to {part.index[-1].date()}   "
              f"bot {pct(s_bot)}/yr   SPY {pct(s_spy)}/yr   {U['compare']} {pct(s_ijr)}/yr")

    if not trades.empty:
        wins = trades[trades["Return"] > 0]
        losses = trades[trades["Return"] <= 0]
        print("\nTRADES")
        print("-" * 60)
        print(f"Total trades:        {len(trades)}  (about {len(trades) / (len(strategy) / 52):.0f} per year)")
        print(f"Winners:             {len(wins) / len(trades):.0%}")
        print(f"Average winner:      {pct(wins['Return'].mean())}")
        print(f"Average loser:       {pct(losses['Return'].mean())}")
        print(f"Average time held:   {trades['Weeks held'].mean():.0f} weeks")
        print("Why trades ended:    " + ", ".join(
            f"{k} {v}" for k, v in trades["Why sold"].value_counts().items()))

    print("\nWHAT THE RULES WOULD HOLD RIGHT NOW (practice info only, not a recommendation)")
    print("-" * 60)
    print(current.to_string(index=False) if not current.empty
          else "Nothing. The bot is fully in cash (market trend filter is off or no stocks qualify).")

    print("""
IMPORTANT: WHY THESE RESULTS LOOK BETTER THAN REALITY
- The stock list is TODAY's index members. Companies that went bankrupt,
  got bought out, or shrank out of the index are missing, so the bot
  never "sees" those losers. For large companies this is even stronger:
  many were added to the S&P 500 only AFTER their stock had already risen.
- Stops are only checked on Fridays, so real losses can be bigger than 10%.
- Trading costs are estimated at 0.25% per buy and sell.
The honest test is Phase 2: practice trading with fake money going forward.
""")
    return lines


def save_chart(lines, out_file):
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                                      gridspec_kw={"height_ratios": [3, 1.3]})
    colors = {"The bot": "#1f4e79", "Hold S&P 500 (SPY)": "#7f8c8d",
              U["compare_name"]: "#c0392b"}
    for k, v in lines.items():
        top.plot(v.index, v, lw=2 if k == "The bot" else 1.4, color=colors[k], label=k)
        bottom.plot(v.index, (v / v.cummax() - 1) * 100, lw=1.2, color=colors[k])
    top.set_ylabel("Value of $2,000 ($)")
    top.set_title(f"Weekly 52-week-high bot on {U['name']} vs. holding the market")
    top.legend(frameon=False)
    top.grid(alpha=0.25)
    bottom.set_ylabel("Drop from peak (%)")
    bottom.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_file, dpi=150)
    plt.close(fig)


def main(**kwargs):
    stocks, market = load_data(**kwargs)
    stocks, market = line_up_weeks(stocks), line_up_weeks(market)
    usable = stocks["Close"].notna().sum().gt(52).sum()
    print(f"Stocks with at least a year of price history: {usable}")
    if usable < 50:
        raise SystemExit("Too few stocks have usable data. Screenshot this and send it to Claude.")
    print("Replaying every week (takes about a minute)...")
    strategy, trades, current = run_backtest(chart_readings(stocks), market)
    if trades.empty and current.empty:
        print("\nWARNING: the bot made zero trades. That's almost certainly a problem")
        print("with the data, not the strategy. Screenshot this and send it to Claude.")
    lines = print_results(strategy, market, trades, current)
    trades.to_csv(OUT_DIR / f"bot_backtest_trades_{CHOICE}.csv", index=False)
    save_chart(lines, OUT_DIR / f"bot_backtest_chart_{CHOICE}.png")
    print(f"Saved every trade: {OUT_DIR / f'bot_backtest_trades_{CHOICE}.csv'}")
    print(f"Saved chart:       {OUT_DIR / f'bot_backtest_chart_{CHOICE}.png'}")


if __name__ == "__main__":
    main()
