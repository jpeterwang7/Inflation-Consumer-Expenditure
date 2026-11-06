# U.S. Inflation Trends and Consumer Prices

Using SQL, Python, and Tableau to examine CPI trends, category-level price changes, and forecast inflation through 2027.

## Data Sources

All four datasets come from the U.S. Bureau of Labor Statistics (BLS), based on the **March 2026 Consumer Price Index (CPI) release** - the government's monthly measure of average price change for a basket of consumer goods and services.

| Source | What it contains | Used for |
|---|---|---|
| `Jan_2018-Mar_2026.csv` | Monthly headline CPI values, January 2018-March 2026, in long format (`MonthYear`, `CPI_Value`). Output of `Jan_2018-Mar_2026.sql`, which reshapes BLS's original wide-format table (one column per month) into rows. | Overall CPI trend and year-over-year inflation rate (Figures 1-2) |
| `cpi_weighted.xlsx` | BLS's expenditure-category breakdown: each category's relative importance (spending weight) alongside its year-over-year price change. | Broad categories (Level 3 in BLS's hierarchy, e.g. "Food," "Shelter") with the largest price increases over the past year (Figure 3) |
| `news-release-table2-202603.xlsx` | BLS's detailed CPI table (Table 2) from the March 2026 news release, covering more granular categories (Level 5+ - specific goods and services rather than broad groups). | Sharpest individual price increases and declines (Figure 4) |
| `cpi_clean.xlsx` | Historical monthly CPI data back to 1913, in wide format (one column per month). | Training and backtest data for the forecasting model (Figure 6) |
| BLS Public Data API (`api.bls.gov`) | Live pull of 8 monthly CPI-U series, 2019-2026, indexed to Jan 2019 = 100: 7 grouped into essential/nonessential, plus All Items kept separate as an overall-inflation baseline. Fetched directly in `main.py`. | Essential vs. nonessential comparison, exported as `cpi_tableau.csv` for Tableau (Figure 5) |

## Repository Structure

```
.
├── Data/
│   ├── Jan_2018-Mar_2026.csv
│   ├── cpi_weighted.xlsx
│   ├── news-release-table2-202603.xlsx
│   └── cpi_clean.xlsx
├── main.py
├── arima_model_comparison.py
├── Jan_2018-Mar_2026.sql
└── README.md
```

## Requirements

- Python 3.9+, plus: `pip install pandas numpy matplotlib plotly statsmodels scipy openpyxl requests`
- Tableau Desktop or Public (Figure 5 only)

## Workflow

1. **Reshape historical CPI data (SQL).** BLS publishes historical CPI as one row per year with a column per month. `Jan_2018-Mar_2026.sql` uses `UNPIVOT` to turn that into a simple two-column time series (`MonthYear`, `CPI_Value`), filtered to January 2018-March 2026. This long format makes it easy to plot as a trend and to calculate rolling changes in Python.

2. **Figure 1 - CPI trend.** A line chart of the headline CPI index from 2018 to 2026, showing the overall trajectory of consumer prices, including the sharp acceleration during 2021-2022.

3. **Figure 2 - Inflation rate.** Converts the CPI level into a year-over-year inflation rate (`CPI.pct_change(12) * 100`) - the figure usually reported as "inflation" in the news. Viewed next to Figure 1, it shows that the CPI level keeps climbing even as the inflation *rate* comes down, which is part of why lower inflation doesn't always feel like relief.

4. **Figure 3 - Categories with the largest increases.** Ranks BLS's Level 3 expenditure categories by their March 2025-March 2026 year-over-year change and charts the 10 with the steepest increases, showing which broad areas of spending are driving inflation.

5. **Figure 4 - Where inflation is still lingering.** Drills into more detailed, lower-level categories to show the 8 largest increases and 8 largest declines, excluding energy (already covered in Figure 3). This captures more specific, product-level price swings that get lost in the broad category view.

6. **Figure 5 - Essential vs. nonessential goods.** `main.py` pulls 8 category series (Food at Home, Food Away, Energy, Shelter, Medical Care, Used Cars, Apparel, and All Items) from the BLS public API and indexes each to Jan 2019 = 100. Seven of them are grouped into `Essential Goods` (Food at Home, Energy, Shelter, Medical Care) and `Nonessential Goods` (Food Away, Used Cars, Apparel); All Items is kept separate as an `Overall Inflation` baseline, not classified as essential or nonessential, so the chart can show how the two groups are moving relative to headline CPI rather than in isolation. The three resulting series are exported as `cpi_tableau.csv` and imported into **Tableau** to build the comparison. The 2026-2027 estimate bands are generated in Tableau itself, on top of that file, using the built-in forecast feature. This comparison speaks to why household budgets can stay squeezed even as headline inflation eases.

7. **Figure 6 - Seasonal ARIMA forecast.** Forecasts monthly CPI through May 2027 from the full historical series (January 1990-present). CPI is modeled in logs, so monthly changes are percentage changes, and an Augmented Dickey-Fuller test on the differenced series confirms stationarity. October 2025 has no BLS release (federal government shutdown), so it is kept as a missing month that the model skips rather than dropped; otherwise September and November 2025 would be treated as consecutive months. The model, SARIMA(0,1,1)(1,0,1)12 with drift, was chosen with the backtest described in [Forecast Model Selection](#forecast-model-selection). It is fit on the complete dataset and produces the forecast with a 95% prediction interval.

Figures 1-4 and 6 render as interactive Plotly charts (`fig.show(renderer="browser")`); none are saved to disk currently - add `fig.write_html(...)` / `fig.write_image(...)` calls if you need static exports.

## Forecast Model Selection

The forecast originally used ARIMA(4,1,1) on the CPI level, selected with a single 80/20 train/test split. That setup had four weaknesses:

- **No drift term.** Nothing in the model says prices tend to rise, so long forecasts flatten out. Trained on 1990-2018, it predicted CPI of about 250 for 2026; the actual value was 333.
- **No seasonality.** The BLS series is not seasonally adjusted: prices rise about 0.5% in an average March and fall in December. The model left that pattern in its errors (Ljung-Box test, p < 0.001).
- **CPI points instead of percentages.** A 1-point rise meant more in 1990 (CPI ≈ 130) than in 2026 (CPI ≈ 330).
- **A weak test.** The 80/20 split scored one 87-month forecast, which mostly rewarded whichever model happened to anticipate the 2021-22 surge, rather than accuracy over the 13-month horizon actually being forecast.

`arima_model_comparison.py` replaces that test with a rolling backtest. Starting in December 2009, each model is fit only on the data available at that point, forecasts the next 13 months (the length of the real forecast, April 2026 to May 2027), and is scored against actual CPI. The starting point then moves forward one month and the process repeats, 184 times through March 2025.

| Model | Why it's tested | 1-month RMSE | 12-month RMSE | 95% interval coverage (12 months) |
|---|---|---|---|---|
| ARIMA(4,1,1), CPI level, no drift | Original model (baseline) | 0.77 | 7.23 | 87% |
| Random walk with drift, log CPI | Simplest reasonable benchmark: prices grow at their long-run average rate | 0.88 | 5.15 | 81% |
| ARIMA(4,1,1) with drift, log CPI | Original order with the drift and scale problems fixed | 0.74 | 5.12 | 89% |
| ARIMA(0,2,1), log CPI | Lets the inflation rate itself wander with no fixed average (a standard inflation benchmark) | 0.84 | 9.41 | 100% (intervals far too wide) |
| SARIMA(0,1,1)(0,1,1)12, log CPI | Adds the month-of-year pattern | **0.65** | 5.14 | 87% |
| **SARIMA(0,1,1)(1,0,1)12 with drift, log CPI** | Combines seasonality with drift | 0.69 | **4.73** | 90% |

RMSE is the typical forecast miss in CPI points (lower is better). The seasonal models' orders were chosen by AIC on 1990-2009 data only, before any backtest scoring.

**Selected model: SARIMA(0,1,1)(1,0,1)12 with drift.** It has the lowest 12-month error and the lowest error averaged over all 13 horizons, and among the competitive models its 95% intervals were the most reliable. A Diebold-Mariano test shows it is significantly more accurate than the original model at 1 month (p = 0.04) and nearly so at 12 months (p = 0.06). The seasonal model without drift is slightly better at 1-3 months, but the two are not statistically distinguishable.

The selected model forecasts CPI of about 344.1 for May 2027 (95% interval 333.6-355.0), with year-over-year inflation easing from 3.8% in April 2026 to about 3.1% by April 2027. Even this model's 95% intervals contained the actual value only about 90% of the time at 12 months in the backtest, so the band in Figure 6 likely understates the true uncertainty.

## How to Run

1. Run `Jan_2018-Mar_2026.sql` against the BLS historical CPI-U table and export the result to `Data/Jan_2018-Mar_2026.csv`; the other three files go in `Data/`.
2. `pip install` the requirements above, then run `python main.py` from the repository root.
3. `main.py` also writes `cpi_tableau.csv` for Figure 5 - open it in Tableau to reproduce the comparison.
4. Optional: `python arima_model_comparison.py` reruns the forecast model backtest (about 30 seconds on a multi-core machine) and saves the results to `model_comparison_output/`.
