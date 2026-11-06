#Import needed libraries
import json
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller

#Consistent color pallete
BLUE = "#2F5D8A"
ORANGE = "#E8754F"

#Figure 1: CPI line graph (Jan 2018 - Mar 2026)
df1 = pd.read_csv("Data/Jan_2018-Mar_2026.csv")
df1["MonthYear"] = pd.to_datetime(df1["MonthYear"], format="%b %Y")
df1["CPI_Value"] = pd.to_numeric(df1["CPI_Value"])

fig = px.line(df1, x="MonthYear", y="CPI_Value")
fig.update_traces(line=dict(color=BLUE, width=3))

    #Title and font size
fig.update_layout(
    title=dict(
        text="U.S. Consumer Price Index, 2018–2026",
        font=dict(size=26)
    ),
    xaxis_title="Year", 
    yaxis_title="Consumer Price Index",
    template="plotly_white",
    font=dict(size=16)
    )
    
fig.show(renderer="browser")

#Figure 2: Inflation Rate
df2 = pd.read_csv("Data/Jan_2018-Mar_2026.csv")
df2["MonthYear"] = pd.to_datetime(df2["MonthYear"])
df2["YoY Inflation"] = df2["CPI_Value"].pct_change(12) * 100 #inflation formula

fig = px.line(df2, x="MonthYear", y="YoY Inflation")
fig.update_traces(line=dict(color=BLUE, width=3))

    #Title and font size
fig.update_layout(
    title=dict(
        text="U.S. Inflation Rate, 2019–2026",
        font=dict(size=24)
    ),
    xaxis_title="Year",
    yaxis_title="Inflation Rate (%)",
    template="plotly_white",
    font=dict(size=16)
    )

    #Ensuring tickers do not overlap
fig.update_xaxes(dtick="M12",tickformat="%b %Y",tickangle=0)

fig.show(renderer="browser")

#Figure 3: Top 10 categories with largest price increases
df3 = pd.read_excel("Data/cpi_weighted.xlsx", header=None)
df3 = df3.iloc[1:, [0, 1, 29]]
df3.columns = ["level", "category", "YoY"]

df3["level"] = pd.to_numeric(df3["level"], errors="coerce")
df3["YoY"] = pd.to_numeric(df3["YoY"], errors="coerce")

    #Keep only level 3 categories
df3 = df3[df3["level"] == 3].copy()
df3 = df3.dropna(subset=["category", "YoY"])

    #Clean category names
df3["category"] = df3["category"].astype(str).str.strip()

    #Select Top 10
top10 = (df3.nlargest(10, "YoY").sort_values("YoY", ascending=False))

    # Create chart
fig = px.bar(top10, x="category", y="YoY", text="YoY")

    #Ensuring texts do not overlap with axis marks
fig.update_traces(marker_color=BLUE,texttemplate="%{text:.1f}%",textposition="outside")

    #Title and font size
fig.update_layout(
    title=dict(
        text="Top 10 Categories With the Largest Price Increases"
             "<br><sup>Year-over-year percentage change, March 2025–March 2026</sup>",
        font=dict(size=24)
    ),
    xaxis_title="Expenditure Category",
    yaxis_title="Price Change (%)",
    template="plotly_white"
)

fig.update_xaxes(tickangle=-25)
fig.show(renderer="browser")

# Figure 4: Lingering inflation

    #Selects rows and columns
df4 = pd.read_excel("Data/news-release-table2-202603.xlsx", header=None)
df4 = df4.iloc[6:, [0, 1, 2, 3]]
df4.columns = ["level", "category", "weight", "YoY"]

    #Convert to numeric columns
df4["level"] = pd.to_numeric(df4["level"],errors="coerce")
df4["weight"] = pd.to_numeric(df4["weight"], errors="coerce")
df4["YoY"] = pd.to_numeric(df4["YoY"], errors="coerce")

    #Keep detailed categories at level 5 or above
df4 = df4[
    (df4["level"] >= 5) &
    df4["category"].notna() &
    df4["YoY"].notna() &
    df4["weight"].notna()
    ].copy()

    #Remove footnote markers from names
df4["category"] = (
    df4["category"]
    .astype(str)
    .str.replace(r"\(\d+\)", "", regex=True)
    .str.strip()
    )

    #Exclude energy categories already covered in previous graphs
energy_terms="gasoline|fuel oil|energy|utility gas|electricity"

df4 = df4[~df4["category"].str.contains(energy_terms,case=False,na=False)]

    #8 largest increases + 8 largest declines
increases = df4.nlargest(8, "YoY")
declines = df4.nsmallest(8, "YoY")

chart = pd.concat([declines, increases]).sort_values("YoY")

    #Making colors different for declines vs increases
colors = [BLUE if x < 0 else ORANGE for x in chart["YoY"]]

    #Creating figure
fig = go.Figure()
fig.add_trace(
    go.Bar(
        x=chart["YoY"],
        y=chart["category"],
        orientation="h",
        marker_color=colors,
        text=[f"{x:+.1f}%" for x in chart["YoY"]],
        textposition="inside",
        insidetextanchor="end",
        textfont=dict(color="white")
        )
    )

    #Adding figure title, subtitle, and axis titles
fig.update_layout(
    template="plotly_white",
    title=dict(
        text=(
            "Where Inflation Is Still Lingering"
            "<br><sup>Largest year-over-year price increases and declines by detailed category · March 2026</sup>"
        ),
        font=dict(size=24)
        ),
    xaxis_title="Year-over-Year Price Change (%)",
    yaxis_title="",
    showlegend=False
    )

#Verticle zero reference line
fig.add_vline(x=0, line_width=1.5)
#Add percentage symbols onto x axis numbers
fig.update_xaxes(ticksuffix="%")

fig.show(renderer="browser")

#Figure 5: Essential vs. Non-essential goods
    #Pull indexed CPI-U series by category from the BLS public API
series = {
    "All Items":    "CUUR0000SA0",
    "Food at Home": "CUUR0000SAF11",
    "Food Away":    "CUUR0000SEFV",
    "Energy":       "CUUR0000SA0E",
    "Shelter":      "CUUR0000SAH1",
    "Medical Care": "CUUR0000SAM",
    "Used Cars":    "CUUR0000SETA02",
    "Apparel":      "CUUR0000SAA",
    }

payload = json.dumps({
    "seriesid": list(series.values()),
    "startyear": "2019",
    "endyear": "2026",
    })

response = requests.post(
    "https://api.bls.gov/publicAPI/v2/timeseries/data/",
    data=payload,
    headers={"Content-type": "application/json"}
    )
bls_data = response.json()

    #Reshape the API response into a long dataframe
id_to_name = {v: k for k, v in series.items()}
records = []
for s in bls_data["Results"]["series"]:
    name = id_to_name[s["seriesID"]]
    for obs in s["data"]:
        if obs["value"] == "-":  #skip missing values
            continue
        records.append({
            "category": name,
            "year": int(obs["year"]),
            "month": int(obs["period"].replace("M", "")),
            "value": float(obs["value"]),
            })

df5 = pd.DataFrame(records)
df5["date"] = pd.to_datetime(df5[["year", "month"]].assign(day=1))
df5 = df5.sort_values(["category", "date"]).reset_index(drop=True)

    #Index each category to January 2019 = 100 so categories are comparable
base = df5[df5["date"] == "2019-01-01"].set_index("category")["value"]
df5["indexed"] = df5.apply(lambda r: r["value"] / base[r["category"]] * 100, axis=1)

    #Group categories into essential vs. nonessential goods
group_map = {
    "Food at Home": "Essential Goods",
    "Energy":       "Essential Goods",
    "Shelter":      "Essential Goods",
    "Medical Care": "Essential Goods",
    "Food Away":    "Nonessential Goods",
    "Used Cars":    "Nonessential Goods",
    "Apparel":      "Nonessential Goods",
    "All Items":    "Overall Inflation",
    }
df5["group"] = df5["category"].map(group_map)

    #Average the indexed values within each group, by month
tableau_df = (
    df5.groupby(["group", "date"])["indexed"]
    .mean()
    .reset_index()
    .rename(columns={"group": "Category", "date": "Date", "indexed": "Index"})
    )

    #Export for Tableau — the essential vs. nonessential chart and the
    #2026-2027 estimate bands are built in Tableau from this file
tableau_df.to_csv("cpi_tableau.csv", index=False)
print("Saved to cpi_tableau.csv")

#Figure 6: Seasonal ARIMA forecast
    #Load and reshape data
df = pd.read_excel("Data/cpi_clean.xlsx").melt(
    id_vars="Year",
    var_name="Month",
    value_name="CPI"
    )

    #Month map that will be used to date each entry
    #Since ARIMA model is a time series model
month_map = {
    "Jan.":1, "Feb.":2, "Mar.":3, "Apr.":4,
    "May":5, "Jun.":6, "Jul.":7, "Aug.":8,
    "Sep.":9, "Oct.":10, "Nov.":11, "Dec.":12
    }

    #Establishing data and CPI value columns
df["Month"] = df["Month"].map(month_map)
df["Time"] = pd.to_datetime(dict(year=df["Year"], month=df["Month"], day=1))
df["CPI"] = pd.to_numeric(df["CPI"], errors="coerce")

    #Monthly series from 1990 through the latest release
    #Oct 2025 has no BLS release (government shutdown), so it stays a missing
    #month that the model skips instead of treating Sep and Nov as back to back
cpi = df.set_index("Time")["CPI"].sort_index()
cpi = cpi["1990-01-01":cpi.last_valid_index()].asfreq("MS")

    #Model log CPI so monthly changes are percentage changes (inflation rates)
log_cpi = np.log(cpi)

    #Check stationarity
    #Difference log CPI once to remove the long term upward trend
log_diff = log_cpi.diff().dropna()

    #Run the Augmented Dickey-Fuller test
    #A p-value below 0.05 suggests stationarity
adf_pvalue = adfuller(log_diff)[1]
print("ADF p-value:", round(adf_pvalue, 4))

    #Selected model: SARIMA(0,1,1)(1,0,1)12 with drift
    #Chosen with a rolling backtest of six candidates (see arima_model_comparison.py)
    #Drift = average monthly inflation, seasonal terms = month-of-year price pattern
final_model = ARIMA(
    log_cpi,
    order=(0, 1, 1),
    seasonal_order=(1, 0, 1, 12),
    trend="t"
    ).fit()
print(final_model.summary())

future_dates = pd.date_range(
    cpi.index.max() + pd.offsets.MonthBegin(),
    "2027-05-01",
    freq="MS"
    )

    #Forecast CPI for each future month
    #Convert from log scale back to CPI points
forecast = final_model.get_forecast(steps=len(future_dates))
ci = np.exp(forecast.conf_int())
forecast_df = pd.DataFrame({
    "Time": future_dates,
    "Forecast": np.exp(forecast.predicted_mean.values),
    "Lower": ci.iloc[:, 0].values,
    "Upper": ci.iloc[:, 1].values
    })
print(forecast_df.round(2).to_string(index=False))


    #Plot final forecast
plot_df = cpi["2018-01-01":].dropna()
fig = go.Figure()
fig.add_scatter(
    x=plot_df.index, y=plot_df.values,
    name="Actual CPI", line=dict(color=BLUE, width=4)
    )

    #Adding 95% confidence interval for forecast portion
fig.add_scatter(
    x=pd.concat([forecast_df["Time"], forecast_df["Time"][::-1]]),
    y=pd.concat([forecast_df["Upper"], forecast_df["Lower"][::-1]]),
    fill="toself", fillcolor="lightgray",
    line=dict(width=0), name="95% Confidence Interval"
    )

    
    #Forecast line
fig.add_scatter(
    x=forecast_df["Time"], y=forecast_df["Forecast"],
    name="Forecast", line=dict(color=ORANGE, width=4)
    )

    #Axis titles
fig.update_layout(
    template="plotly_white",
    title="<b>Where Are Prices Headed?</b><br><sup>Seasonal ARIMA forecast of U.S. CPI through May 2027</sup>",
    yaxis_title="Consumer Price Index",
    legend=dict(orientation="h")
    )

fig.update_xaxes(tickformat="%Y", dtick="M12", showgrid=False)
fig.update_yaxes(gridcolor="#EAEAEA", zeroline=False)

fig.show(renderer="browser")
