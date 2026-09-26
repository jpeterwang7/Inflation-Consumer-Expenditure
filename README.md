# U.S. Inflation Trends and Consumer Prices

Using SQL, Python, and Tableau to examine CPI trends, category-level price changes, and forecast inflation through 2027.

## Data Sources

All four datasets come from the U.S. Bureau of Labor Statistics (BLS), based on the **March 2026 Consumer Price Index (CPI) release** - the government's monthly measure of average price change for a basket of consumer goods and services.

| Source | What it contains | Used for |
|---|---|---|
| `Jan_2018-Mar_2026.csv` | Monthly headline CPI values, January 2018-March 2026, in long format (`MonthYear`, `CPI_Value`). Output of `Jan_2018-Mar_2026.sql`, which reshapes BLS's original wide-format table (one column per month) into rows. | Overall CPI trend and year-over-year inflation rate (Figures 1-2) |
| `cpi_weighted.xlsx` | BLS's expenditure-category breakdown: each category's relative importance (spending weight) alongside its year-over-year price change. | Broad categories (Level 3 in BLS's hierarchy, e.g. "Food," "Shelter") with the largest price increases over the past year (Figure 3) |
| `news-release-table2-202603.xlsx` | BLS's detailed CPI table (Table 2) from the March 2026 news release, covering more granular categories (Level 5+ - specific goods and services rather than broad groups). | Sharpest individual price increases and declines (Figure 4) |
| `cpi_clean.xlsx` | Historical monthly CPI data back to 1913, in wide format (one column per month). | Training data for the ARIMA forecasting model (Figure 6) |
| BLS Public Data API (`api.bls.gov`) | Live pull of 8 monthly CPI-U series, 2019-2026, indexed to Jan 2019 = 100: 7 grouped into essential/nonessential, plus All Items kept separate as an overall-inflation baseline. Fetched directly in `main.py`. | Essential vs. nonessential comparison, exported as `cpi_tableau.csv` for Tableau (Figure 5) |

## Repository Structure

```
.
├── data/
│   ├── Jan_2018-Mar_2026.csv
│   ├── cpi_weighted.xlsx
│   ├── news-release-table2-202603.xlsx
│   └── cpi_clean.xlsx
├── main.py
├── Jan_2018-Mar_2026.sql
└── README.md
```

> `main.py` currently reads the data files from the project root, not `data/` - add a `data/` prefix to its four read calls to match this structure.

## Requirements

- Python 3.9+, plus: `pip install pandas numpy matplotlib plotly statsmodels scikit-learn openpyxl requests`
- Tableau Desktop or Public (Figure 5 only)

## Workflow

1. **Reshape historical CPI data (SQL).** BLS publishes historical CPI as one row per year with a column per month. `Jan_2018-Mar_2026.sql` uses `UNPIVOT` to turn that into a simple two-column time series (`MonthYear`, `CPI_Value`), filtered to January 2018-March 2026. This long format makes it easy to plot as a trend and to calculate rolling changes in Python.

2. **Figure 1 - CPI trend.** A line chart of the headline CPI index from 2018 to 2026, showing the overall trajectory of consumer prices, including the sharp acceleration during 2021-2022.

3. **Figure 2 - Inflation rate.** Converts the CPI level into a year-over-year inflation rate (`CPI.pct_change(12) * 100`) - the figure usually reported as "inflation" in the news. Viewed next to Figure 1, it shows that the CPI level keeps climbing even as the inflation *rate* comes down, which is part of why lower inflation doesn't always feel like relief.

4. **Figure 3 - Categories with the largest increases.** Ranks BLS's Level 3 expenditure categories by their March 2025-March 2026 year-over-year change and charts the 10 with the steepest increases, showing which broad areas of spending are driving inflation.

5. **Figure 4 - Where inflation is still lingering.** Drills into more detailed, lower-level categories to show the 8 largest increases and 8 largest declines, excluding energy (already covered in Figure 3). This captures more specific, product-level price swings that get lost in the broad category view.

6. **Figure 5 - Essential vs. nonessential goods.** `main.py` pulls 8 category series (Food at Home, Food Away, Energy, Shelter, Medical Care, Used Cars, Apparel, and All Items) from the BLS public API and indexes each to Jan 2019 = 100. Seven of them are grouped into `Essential Goods` (Food at Home, Energy, Shelter, Medical Care) and `Nonessential Goods` (Food Away, Used Cars, Apparel); All Items is kept separate as an `Overall Inflation` baseline, not classified as essential or nonessential, so the chart can show how the two groups are moving relative to headline CPI rather than in isolation. The three resulting series are exported as `cpi_tableau.csv` and imported into **Tableau** to build the comparison. The 2026-2027 estimate bands are generated in Tableau itself, on top of that file, using the built-in forecast feature. This comparison speaks to why household budgets can stay squeezed even as headline inflation eases.

7. **Figure 6 - ARIMA forecast.** Uses the full historical monthly CPI series (1990-present) to forecast prices through May 2027. The process: an 80/20 train/test split, differencing plus an Augmented Dickey-Fuller test to confirm stationarity, fitting several candidate `(p,d,q)` ARIMA specifications, and selecting the one with the lowest out-of-sample RMSE (cross-checked against AIC/BIC). That model is refit on the complete dataset and used to forecast monthly CPI with a 95% confidence interval.

Figures 1-4 and 6 render as interactive Plotly charts (`fig.show(renderer="browser")`); none are saved to disk currently - add `fig.write_html(...)` / `fig.write_image(...)` calls if you need static exports.

## How to Run

1. Run `Jan_2018-Mar_2026.sql` and export the result to `data/Jan_2018-Mar_2026.csv`; add the other three files to `data/`.
2. `pip install` the requirements above, then `python main.py`.
3. `main.py` also writes `cpi_tableau.csv` for Figure 5 - open it in Tableau to reproduce the comparison.
