#Import needed libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error

#Consistent color pallete
BLUE = "#2F5D8A"
ORANGE = "#E8754F"

#Figure 1: CPI line graph (Jan 2018 - Mar 2026)
df1 = pd.read_csv("Jan_2018-Mar_2026.csv")
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
df2 = pd.read_csv("Jan_2018-Mar_2026.csv")
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
df3 = pd.read_excel("cpi_weighted.xlsx", header=None)
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
df4 = pd.read_excel("news-release-table2-202603.xlsx", header=None)
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
    #Using historical

#Figure 6: ARIMA Model
    #Load and reshape data
df = pd.read_excel("cpi_clean.xlsx").melt(
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

    #Cleaning data
df = (df[["Time", "CPI"]].dropna().sort_values("Time"))
    #Filtering dates and resetting index after
df = df[df["Time"] >= "1990-01-01"].reset_index(drop=True)


    #80% Train/ 20% test split
split = int(len(df) * 0.8)
train = df.iloc[:split]
test = df.iloc[split:]

    #Check stationarity
    #Difference the CPI series once to remove the long term upward trend
diff = train["CPI"].diff().dropna()

    #Run the Augmented Dickey-Fuller test
    #A p-value below 0.05 suggests stationarity
adf_pvalue = adfuller(diff)[1]
print("ADF p-value:", round(adf_pvalue, 4))

    #Compare reasonable ARIMA models
    #Define different combinations of (p, d, q) to test
candidates = [
    (1,1,1), (2,1,1), (3,1,1),
    (4,1,1), (4,1,0), (3,1,2)
    ]

    #Fit and evaluate each ARIMA model
results = []
for order in candidates:
    model = ARIMA(train["CPI"], order=order).fit()
    pred = model.forecast(steps=len(test))
    rmse = np.sqrt(mean_squared_error(test["CPI"], pred)) #Prediction error
    results.append([order, model.aic, model.bic, rmse]) #storing results
results = pd.DataFrame(results,columns=["ARIMA", "AIC", "BIC", "RMSE"])
print(results.sort_values("RMSE").to_string(index=False))

    #Select the ARIMA model with the lowest RMSE
best_order = results.loc[results["RMSE"].idxmin(), "ARIMA"]
print("\nSelected model:", best_order)

    #Refit the selected model using the full CPI dataset
final_model = ARIMA(df["CPI"], order=best_order).fit()

future_dates = pd.date_range(
    df["Time"].max() + pd.offsets.MonthBegin(),
    "2027-05-01",
    freq="MS"
    )

    #Forecast CPI for each future month
forecast = final_model.get_forecast(steps=len(future_dates))
ci = forecast.conf_int()
forecast_df = pd.DataFrame({
    "Time": future_dates,
    "Forecast": forecast.predicted_mean.values,
    "Lower": ci.iloc[:, 0].values,
    "Upper": ci.iloc[:, 1].values
    })


    #Plot final forecast
plot_df = df[df["Time"] >= "2018-01-01"]
fig = go.Figure()
fig.add_scatter(
    x=plot_df["Time"], y=plot_df["CPI"],
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
    title="<b>Where Are Prices Headed?</b><br><sup>ARIMA forecast of U.S. CPI through May 2027</sup>",
    yaxis_title="Consumer Price Index",
    legend=dict(orientation="h")
    )

fig.update_xaxes(tickformat="%Y", dtick="M12", showgrid=False)
fig.update_yaxes(gridcolor="#EAEAEA", zeroline=False)

fig.show(renderer="browser")
