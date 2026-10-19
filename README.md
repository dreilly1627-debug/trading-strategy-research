# Trading Strategy Research

Testing whether simple, rule-based trading strategies actually beat buying and holding the market, using honest backtests on real data.

**The short version:** I tested three ideas. Two failed and were dropped without tweaking the rules to rescue them. One (monthly momentum across funds) passed, with an important caveat about what it really offers.

| Study | Idea | Result |
|---|---|---|
| 1. Game launch study | Do video game publisher stocks lag the market after big launches? | Pattern found for Take-Two, but too few examples to trade on |
| 2. Weekly breakout bot | Buy stocks near their 52-week high, sell when the trend breaks | **Failed** on both small and large companies |
| 3. Fund strategies | Monthly momentum and "buy the dip" across 16 funds | **Momentum passed**, buy the dip failed |

---

## Ground rules

These applied to every test:

1. **Rules were set before testing and never tuned to the results.** If a strategy failed, it was dropped, not adjusted until it looked good.
2. **Every strategy was compared to a simple benchmark,** usually just holding the S&P 500 (SPY). A strategy has to beat doing nothing.
3. **No peeking ahead.** Decisions use only information available at the time, and trades happen at the next day's or week's opening price.
4. **Trading costs are included** (0.25% per trade for stocks, 0.05% for funds).
5. **Weaknesses are stated plainly,** especially survivorship bias (explained below).

---

## Study 1: Game launch study

**Question:** GTA 6 launches in November 2026. Historically, how did game publisher stocks perform around major launches?

**Method:** Measured each stock's return minus its market's return before and after 18 major launches (Take-Two and Ubisoft), compared against the same stock's returns in ordinary periods. A "luck check" repeated the test on 5,000 sets of random dates to see how often chance alone produces a result this extreme.

**Findings:**
- **Take-Two lagged the market in the 3 months after all 5 major Rockstar launches** (GTA V, Red Dead Redemption 1 and 2, L.A. Noire, Max Payne 3), by about 19% on average. In ordinary 3-month periods, the same stock *beat* the market 65% of the time. Random dates almost never looked this bad (luck check: under 1%).
- **Ubisoft showed no such pattern.** Its post-launch results averaged out to roughly zero.
- **GTA IV was excluded** from the averages because EA's takeover bid for Take-Two happened in the same window.

**Why I didn't trade it:**
- Only 5 usable examples, and they were the same data that suggested the idea. A pattern can't confirm itself.
- Every Take-Two launch was followed by an earnings report within 3 months, so the launch and the earnings reaction can't be separated.
- EA and Activision couldn't serve as fresh tests: both stopped trading after being acquired, and their price history is no longer available.

---

## Study 2: Weekly 52-week-high breakout bot

**Idea:** Stocks trading near their highest price of the past year tend to keep rising (a documented effect in academic research). Every Friday, the bot buys up to 10 stocks within 5% of their 52-week high and in an uptrend, but only when the overall market is also trending up. It sells when a stock's weekly price closes below its 10-week average or falls 10%.

**Results ($2,000 starting, Dec 2010 to Oct 2026):**

| | Small companies (S&P 600) | Large companies (S&P 500) | Hold S&P 500 |
|---|---|---|---|
| Ending value | $2,761 | $6,500 | $16,175 |
| Average per year | +2.1% | +7.7% | +14.1% |
| Worst drop from a peak | -51.9% | -25.8% | -28.6% |

**Verdict: failed.** It lost to simply holding the market on both groups of stocks.

**What I learned:**
- **The edge decayed.** On large companies, the bot roughly matched the market from 2011 to 2018 (+12.8% vs +12.5% a year), then fell far behind from 2018 to 2026 (+2.9% vs +15.8%). This is common with well-known strategies.
- **It got shaken out constantly.** Nearly every trade ended because the stock dipped below its 10-week average, so the bot kept selling and rebuying and missed strong market years.
- **The real results would be worse.** The test uses *today's* index members, so companies that went bankrupt or were acquired are missing (survivorship bias). This tilts results in the bot's favor, and it still lost.

---

## Study 3: Fund strategies

To avoid survivorship bias entirely, this study uses 16 large, long-running funds that don't disappear: US and international stock markets, 8 US sectors, long-term bonds, gold, and real estate.

**Monthly momentum:** on the last trading day of each month, hold the 3 funds with the best 12-month return. Any of those that isn't beating cash is swapped for bonds.

**Buy the dip:** buy stock funds in a long-term uptrend after a sharp 2-day drop, and sell on the bounce or after 10 days. At most 3 funds at once.

**Results ($2,000 starting, Nov 2005 to Oct 2026):**

| | Ending value | Per year | Worst drop | Worst year |
|---|---|---|---|---|
| **Monthly momentum** | **$17,848** | **+11.0%** | **-23.5%** | **-5.3%** |
| Buy the dip | $5,253 | +4.7% | -18.6% | -11.5% |
| Hold S&P 500 | $18,086 | +11.1% | -55.2% | -36.8% |
| 60% stocks / 40% bonds | $10,568 | +8.3% | -34.7% | -20.0% |

**Monthly momentum: passed, as crash insurance rather than a market-beater.**
- It matched the S&P 500's growth with less than half the worst drop, and beat the classic 60/40 mix on both growth and safety.
- It protected capital in crashes: **-3.3% in 2008** (S&P 500: -36.8%) and **+13.0% in 2022** (S&P 500: -18.2%).
- **The caveat:** most of its advantage came from avoiding 2008. Starting in 2009 instead, $1 grew to $6.26 with momentum vs. $11.69 holding the S&P 500. In long bull markets it trails, and it beat the S&P 500 in only 11 of 22 years.

**Buy the dip: failed.** It won 65% of its trades but still lagged even the 60/40 mix, because its average loss (-1.7%) was larger than its average win (+1.2%).

---

## Limitations

- **Taxes are not included.** Momentum changed holdings in most months, and buy the dip traded about 60 times a year. In a taxable account, that would meaningfully reduce returns.
- **One test period.** Results depend on the start date, as the 2009 comparison above shows.
- **Backtests are not live trading.** Real fills, delays, and behavior (like abandoning a strategy during a drawdown) can differ.

## A data bug worth mentioning

The first run of the weekly breakout bot reported zero trades. The cause: Yahoo Finance labeled the same week with different dates across download batches, which broke every stock's price history into gaps. I fixed it by aligning every week to its Friday, and added checks that now warn if data is missing or if a strategy makes zero trades. Lesson: a result that looks too clean (or too empty) deserves suspicion before interpretation.

## Next step

Practice-trade monthly momentum with a paper (fake money) brokerage account for several months, then compare live results to the backtest.

---

## How to run the studies

Requires Python 3.9+.

```
pip install -r requirements.txt

python studies/1_game_launch_study/launch_event_study.py
python studies/2_weekly_breakout/weekly_breakout_backtest.py         # small companies
python studies/2_weekly_breakout/weekly_breakout_backtest.py large   # large companies
python studies/3_fund_strategies/fund_strategies_backtest.py
```

Each script downloads free price data from Yahoo Finance, prints results in plain English, and saves a chart next to the script.

*For research and education only. Not investment advice.*
