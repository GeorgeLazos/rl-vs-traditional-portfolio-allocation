# RL vs Traditional Portfolio Allocation

![Python 3.12](https://img.shields.io/badge/python-3.12-blue) ![PyTorch 2.13](https://img.shields.io/badge/PyTorch-2.13-ee4c2c) ![Licence: MIT](https://img.shields.io/badge/licence-MIT-green)

An end-to-end study of whether deep reinforcement learning can beat classical portfolio allocation, built as my MSc dissertation at Queen Mary University of London.

I developed a pipeline that collects and aligns over 20 years of point-in-time market data, engineers 75 features from it, and trains PPO agents to allocate capital across 119 assets. The agents were then tested against four classical strategies on four and a half years of data they had never seen, with every strategy trading through the same simulator at the same costs.

### At a glance

- **119 assets in four classes:** 98 S&P 100 equities, 6 bond ETFs, 5 commodity ETFs, 10 REITs, plus cash
- **75 point-in-time features:** 57 per asset and 18 market-wide, all computed only from data known on the day
- **Three PPO designs:** one flat agent, a two-level hierarchy, and a Markowitz-anchored tilt, each run on a monthly and a daily clock
- **Four benchmarks:** equal weight (1/N), Markowitz, minimum variance and risk parity
- **One simulator for everything:** agents and benchmarks trade through the same engine with the same costs
- **Test split touched once:** the code refuses to run it without an explicit flag, and logs every agent test run

## Result

No freely learned agent beat Markowitz. The best one reached a test Sharpe ratio of **1.18**, level with the naive 1/N portfolio, against **1.41** for Markowitz. The anchored tilt matched Markowitz, but by construction rather than by learning anything new.

<!-- Fill in from reports/baselines_report.md and the agent run reports. -->

## Result

No freely learned agent beat Markowitz. The strongest, the monthly monolithic agent, matched 1/N at a Sharpe ratio of 1.18, against 1.41 for Markowitz. The two tilt agents were the only learned models to match the benchmark, at 1.43 and 1.48, but the plain Markowitz weights traded under the tilt agents' own conventions score almost the same, so most of that edge comes from execution rather than learning. The monthly hierarchy was the most defensive learned strategy, with the shallowest drawdown of any learned model.

| Strategy | Return | Volatility | Sharpe | Sortino | Max drawdown | Calmar | Turnover |
|---|---|---|---|---|---|---|---|
| Tilt, daily | 29.2% | 15.0% | 1.48 (0.01) | 2.65 | -10.1% | 2.89 | 3.7 |
| Tilt, monthly | 28.5% | 15.2% | 1.43 (0.01) | 2.51 | -11.5% | 2.48 | 3.4 |
| Markowitz | 28.2% | 15.2% | 1.41 | 2.49 | -11.5% | 2.46 | 3.4 |
| Monolithic, monthly | 21.4% | 13.3% | 1.18 (0.14) | 2.29 | -16.3% | 1.32 | 1.5 |
| 1/N | 20.0% | 12.2% | 1.18 | 2.32 | -14.6% | 1.37 | 0.3 |
| Hierarchy, daily | 15.2% | 9.4% | 1.06 (0.04) | 2.27 | -10.5% | 1.45 | 0.9 |
| Risk parity | 14.8% | 9.4% | 1.03 | 2.25 | -10.4% | 1.42 | 0.3 |
| Monolithic, daily | 17.5% | 12.4% | 1.00 (0.07) | 2.02 | -14.5% | 1.21 | 4.2 |
| Hierarchy, monthly | 13.8% | 8.9% | 0.99 (0.12) | 2.21 | -8.9% | 1.55 | 1.0 |
| Minimum variance | 7.2% | 4.1% | 0.60 | 2.56 | -4.4% | 1.64 | 0.9 |

Test window 2023 to June 2026, after trading costs. Return and volatility are annualised and turnover is the fraction of the portfolio traded per year. Learned models are three-seed means, with the standard deviation across seeds in brackets.

<!-- Add one or two key figures, e.g. docs/equity_test.png and docs/drawdown_test.png -->
<!-- ![Test-period equity curves](docs/equity_test.png) -->

## Why this question

Mean-variance optimisation (Markowitz, 1952) builds portfolios from estimated returns and covariances. Those estimates are noisy, so the optimiser tends to amplify their errors (Michaud, 1989), and out of sample it often fails to beat simple equal weighting (DeMiguel et al., 2009).

Reinforcement learning treats allocation as a sequence of decisions and learns a policy directly from market data, with no return forecast in between. Recent hierarchical RL frameworks report strong results. This project tests that claim under strict conditions: realistic costs, point-in-time data and a test set that no design decision was allowed to see.

## How it works

The pipeline has five layers. Each one reads only from the layer below it, and every stage imports its dates, universe and cost rates from `config/`, so there is one source of truth for each.

| Layer | Package | What it does |
|---|---|---|
| 0. Collection | `collectors/` | Fetches raw prices, corporate actions, fundamentals and macro data. Every saved response is recorded in a SQLite ledger, so reruns skip finished work and retry failures. |
| 1. Dataset | `dataset/` | Aligns everything to one trading calendar and builds point-in-time Parquet tables for prices, macro context and fundamentals. |
| 2. Features | `features/` | Builds the 75 features declared in `config/feature_registry.py`. Checks fail if the code and the registry disagree in either direction. |
| 3. Baselines and engine | `portfolio/` | Runs the four benchmarks through a shared simulator and computes the study's metrics. |
| 4. RL agents | `portfolio/models/` | Trains and evaluates the PPO agents in an environment that wraps the same simulator. |

### Data

| Data | Source |
|---|---|
| Equity and REIT prices, corporate actions, quarterly fundamentals | Sharadar (paid subscription) |
| ETF prices, VIX, independent price cross-check | yfinance |
| Macro series with their full revision history | FRED and ALFRED |
| Independent fundamentals cross-check | SEC EDGAR |

Macro series are stored as every published vintage, so the model sees the value that was known on each date rather than a figure revised years later. Fundamentals are timed by their filing date for the same reason.

### Features

The 57 asset features cover momentum, mean reversion, volatility, liquidity, value, growth, quality, factor exposure and sector. The 18 market features cover fear, interest rates, the economy, cross-asset signals and market direction. A feature stays blank until it has enough history for an honest first value, and those blanks are never filled in.

### Benchmarks

All four rebalance monthly. Markowitz, minimum variance and risk parity share one estimation block (a 252-day window with Ledoit-Wolf covariance shrinkage), so the only difference between them is the objective.

- **Equal weight (1/N):** every tradeable asset gets the same weight
- **Markowitz:** long-only maximum Sharpe portfolio against the cash rate
- **Minimum variance:** lowest-risk portfolio, ignoring return forecasts
- **Risk parity:** inverse-volatility weights

### Agents

Every agent uses PPO with a Gaussian policy. A masked softmax turns the policy's output into weights, so assets that aren't tradeable yet get exactly zero. The reward is the differential Sharpe ratio, minus a penalty on turnover, and a small no-trade band stops the agent from paying costs for tiny adjustments.

- **Flat:** one network allocates across all 119 assets and cash
- **Hierarchy:** four sleeve agents each allocate within one asset class. Once frozen, a top-level allocator splits capital between the four sleeves and cash every month
- **Anchored tilt:** starts as the Markowitz portfolio and can only tilt away from it, within a fixed bound, where the reward justifies it

### Costs

Trades cost 10 bp for equities and REITs, 5 bp for bond and commodity ETFs and 1 bp for cash. Results are also reported at 0, 5, 10 and 25 bp, so no conclusion rests on one cost assumption.

## Evaluation discipline

| Split | Period | Used for |
|---|---|---|
| Train | 2005 to 2018 | Fitting, with walk-forward folds over 2012 to 2017 for model selection |
| Validation | 2019 to 2021 | Checkpoint selection |
| Test | 2022 to June 2026 | One final pass, every strategy together |

- The split boundaries live in one file, `config/splits.py`, which every stage imports
- Training is seeded end to end, and deterministic evaluation reproduces a backtest exactly
- The test split only runs with `--acknowledge-single-test-pass`, and every agent test run is logged

## Repository structure

```
config/             universe, splits, costs, feature registry, one config file per agent
collectors/         source adapters and the SQLite ledger
dataset/            trading calendar and point-in-time tables
features/           asset and market feature builders, with checks
portfolio/          engine, benchmarks, metrics, plots and reports
portfolio/models/   PPO environment, networks, training, allocator, evaluation
```

## Getting started

Requires Python 3.12 and a Sharadar subscription for the equity and REIT data. FRED and SEC EDGAR access is free.

```bash
pip install -r requirements.txt
cp .env.example .env        # add your SHARADAR_KEY, FRED_KEY and SEC_USER_AGENT
```

Run the pipeline in order:

```bash
# 1. Collect raw data
python -m collectors.collect_all

# 2. Build the processed tables and features
python -m dataset.build_all
python -m features.build_all

# 3. Run the benchmarks on train and validation
python -m portfolio.run --splits train val

# 4. Train the agents (repeat for bonds, commodities, reits, flat and tilt)
python -m portfolio.models.train --sleeve equities

# 5. Build, check and train the hierarchy's allocator
python -m portfolio.models.allocate --build
python -m portfolio.models.allocate --check
python -m portfolio.models.allocate --promote

# 6. The final test pass, once
python -m portfolio.run --splits test --acknowledge-single-test-pass
python -m portfolio.models.evaluate --run agent_runs/<sleeve>/<run_id> --window test --acknowledge-single-test-pass
```

Self-tests and checks:

```bash
python -m features.selftest
python -m portfolio.models.selftest
python -m portfolio.checks --all
python -m portfolio.models.checks --all
```

## Limitations

- **Survivorship bias:** the equity universe is the S&P 100 as constituted in 2025, which favours firms that survived and grew. This inflates every strategy, benchmarks included.
- **Paid data:** the equity and REIT data needs a Sharadar subscription, so the pipeline can't be fully reproduced for free.
- **Allocator approximation:** the hierarchy's allocator trains on the frozen sleeves' return series and charges class-level costs on share changes, which approximates the underlying trades.

## Author

Georgios Lazos, MSc Artificial Intelligence and Machine Learning in Sciences, Queen Mary University of London. Supervised by Dr Linus Wunderlich.

Dissertation: *Assessing performance of Reinforcement Learning against Traditional Allocation Methods* (2026).

## Licence

MIT. See [LICENSE](LICENSE).
