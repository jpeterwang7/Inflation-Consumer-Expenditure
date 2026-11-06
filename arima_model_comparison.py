"""
ARIMA model comparison for the CPI forecast (Figure 6).

Baseline: ARIMA(4,1,1) on the CPI level with no constant (current main.py spec).
Challengers:
  M1  Random walk + drift on log CPI          -> naive benchmark
  M2  ARIMA(4,1,1) + drift on log CPI          -> same dynamics, fixed specification
  M3  ARIMA(0,2,1) on log CPI (IMA(1,1) infl.) -> lets the inflation *level* shift
  M4  SARIMA(0,1,1)(0,1,1)12 on log CPI        -> handles the seasonality in NSA CPI
Decision step:
  M5  SARIMA(0,1,1)(1,0,1)12 + drift (log)     -> combines M4's seasonality with M2's drift
      (orders for M4/M5 were chosen by AIC on 1990-2009 only, before any scoring)

Evaluation: expanding-window, rolling-origin forecasts (origins Jan 2010 - Mar 2025),
horizons 1-13 months (the length of the real forecast, Apr 2026 -> May 2027).

Usage: python arima_model_comparison.py [path/to/cpi_clean.xlsx] [output_dir]
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import sys
import json
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox

H = 13                       # forecast horizon (Apr 2026 -> May 2027)
FIRST_ORIGIN = "2009-12-01"  # last training month of the first rolling window
LAST_ORIGIN = "2025-03-01"   # last origin whose 13-month forecast can be scored
START = "1990-01-01"

# name: (use_log, order, seasonal_order, trend)
MODELS = {
    "B0 ARIMA(4,1,1) level, no drift [current]": (False, (4, 1, 1), (0, 0, 0, 0), "n"),
    "M1 Random walk + drift (log)":              (True,  (0, 1, 0), (0, 0, 0, 0), "t"),
    "M2 ARIMA(4,1,1) + drift (log)":             (True,  (4, 1, 1), (0, 0, 0, 0), "t"),
    "M3 ARIMA(0,2,1) / IMA(1,1) inflation (log)": (True, (0, 2, 1), (0, 0, 0, 0), "n"),
    "M4 SARIMA(0,1,1)(0,1,1)12 (log)":           (True,  (0, 1, 1), (0, 1, 1, 12), "n"),
    "M5 SARIMA(0,1,1)(1,0,1)12 + drift (log)":   (True,  (0, 1, 1), (1, 0, 1, 12), "t"),
}


def quiet():
    # statsmodels forces ConvergenceWarning to "always" on import, so filter after importing
    warnings.simplefilter("ignore")


def load_cpi(path):
    raw = pd.read_excel(path).melt(id_vars="Year", var_name="Month", value_name="CPI")
    month_map = {"Jan.": 1, "Feb.": 2, "Mar.": 3, "Apr.": 4, "May": 5, "Jun.": 6,
                 "Jul.": 7, "Aug.": 8, "Sep.": 9, "Oct.": 10, "Nov.": 11, "Dec.": 12}
    raw["Time"] = pd.to_datetime(dict(year=raw["Year"], month=raw["Month"].map(month_map), day=1))
    raw["CPI"] = pd.to_numeric(raw["CPI"], errors="coerce")
    s = raw.set_index("Time")["CPI"].sort_index()
    last = s.last_valid_index()
    # Keep a true monthly index: Oct 2025 (no BLS release) stays NaN and the
    # Kalman filter skips it, instead of silently gluing Sep and Nov together.
    return s[START:last].asfreq("MS")


def fit(y, spec):
    use_log, order, sorder, trend = spec
    endog = np.log(y) if use_log else y
    return ARIMA(endog, order=order, seasonal_order=sorder, trend=trend).fit()


def forecast(res, spec, steps, alpha=0.05):
    use_log = spec[0]
    f = res.get_forecast(steps)
    mean, ci = f.predicted_mean.values, f.conf_int(alpha=alpha).values
    if use_log:
        return np.exp(mean), np.exp(ci[:, 0]), np.exp(ci[:, 1])
    return mean, ci[:, 0], ci[:, 1]


def one_origin(args):
    path, origin = args
    quiet()
    s = load_cpi(path)
    y = s[:origin]
    actual = s[pd.Timestamp(origin) + pd.offsets.MonthBegin():].iloc[:H].values
    base = y.dropna().iloc[-1]
    rows = []
    for name, spec in MODELS.items():
        res = fit(y, spec)
        mean, lo, hi = forecast(res, spec, H)
        for h in range(H):
            rows.append(dict(model=name, origin=origin, h=h + 1, actual=actual[h],
                             fcst=mean[h], lo=lo[h], hi=hi[h], base=base))
    return rows


def dm_test(e1, e2, h):
    """Diebold-Mariano with Harvey-Leybourne-Newbold correction, squared-error loss.
    Negative stat => model 1 more accurate."""
    d = e1 ** 2 - e2 ** 2
    n = len(d)
    dbar = d.mean()
    gamma = [np.sum((d[k:] - dbar) * (d[:n - k] - dbar)) / n for k in range(h)]
    var = (gamma[0] + 2 * sum(gamma[1:])) / n
    dm = dbar / np.sqrt(var)
    hln = dm * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    p = 2 * stats.t.sf(abs(hln), df=n - 1)
    return hln, p


def main():
    quiet()
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join("Data", "cpi_clean.xlsx")
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "model_comparison_output"
    os.makedirs(out_dir, exist_ok=True)
    s = load_cpi(path)
    report = {}

    # 1. Original 80/20 single hold-out (what main.py does), for continuity
    split = int(len(s) * 0.8)
    train, test = s.iloc[:split], s.iloc[split:]
    holdout = {}
    for name, spec in MODELS.items():
        mean, lo, hi = forecast(fit(train, spec), spec, len(test))
        ok = ~np.isnan(test.values)
        holdout[name] = dict(rmse=float(np.sqrt(np.mean((mean[ok] - test.values[ok]) ** 2))),
                             final_fcst=float(mean[-1]), final_actual=float(test.values[-1]))
    report["holdout_80_20"] = dict(train_end=str(train.index[-1].date()), steps=len(test), models=holdout)

    # 2. Rolling-origin evaluation
    origins = [str(d.date()) for d in pd.date_range(FIRST_ORIGIN, LAST_ORIGIN, freq="MS")]
    with ProcessPoolExecutor() as ex:
        rows = [r for chunk in ex.map(one_origin, [(path, o) for o in origins]) for r in chunk]
    fc = pd.DataFrame(rows).dropna(subset=["actual"])
    fc["err"] = fc["fcst"] - fc["actual"]
    fc["err_pp"] = 100 * fc["err"] / fc["base"]  # error in cumulative inflation, % points
    fc["covered"] = (fc["actual"] >= fc["lo"]) & (fc["actual"] <= fc["hi"])
    fc["period"] = np.where(pd.to_datetime(fc["origin"]) < "2020-01-01", "2010-2019", "2020-2025")
    fc.to_csv(os.path.join(out_dir, "rolling_forecasts.csv"), index=False)

    def summarise(g):
        return pd.Series({
            "RMSE_h1": np.sqrt((g.loc[g.h == 1, "err"] ** 2).mean()),
            "RMSE_h3": np.sqrt((g.loc[g.h == 3, "err"] ** 2).mean()),
            "RMSE_h6": np.sqrt((g.loc[g.h == 6, "err"] ** 2).mean()),
            "RMSE_h12": np.sqrt((g.loc[g.h == 12, "err"] ** 2).mean()),
            "RMSE_all": np.sqrt((g["err"] ** 2).mean()),
            "infl_err_pp_h12": np.sqrt((g.loc[g.h == 12, "err_pp"] ** 2).mean()),
            "bias_h12": g.loc[g.h == 12, "err"].mean(),
            "cover95_h1": g.loc[g.h == 1, "covered"].mean(),
            "cover95_h12": g.loc[g.h == 12, "covered"].mean(),
        })

    overall = fc.groupby("model", sort=False).apply(summarise)
    by_period = fc.groupby(["period", "model"], sort=False).apply(summarise)[["RMSE_h1", "RMSE_h12", "RMSE_all"]]
    overall.to_csv(os.path.join(out_dir, "rolling_summary.csv"))
    by_period.to_csv(os.path.join(out_dir, "rolling_by_period.csv"))
    report["rolling_overall"] = overall.round(4).to_dict(orient="index")
    report["rolling_by_period"] = {f"{p} | {m}": v for (p, m), v in by_period.round(4).to_dict(orient="index").items()}
    report["n_origins"] = len(origins)

    # 3. Diebold-Mariano: every challenger vs baseline, plus head-to-heads, at h=1 and h=12
    names = list(MODELS)
    pairs = [(n, names[0]) for n in names[1:]] + [(names[5], names[4]), (names[5], names[2]), (names[4], names[2])]
    dm = {}
    for m1, m2 in pairs:
        for h in (1, 12):
            a = fc[(fc.model == m1) & (fc.h == h)].set_index("origin")["err"]
            b = fc[(fc.model == m2) & (fc.h == h)].set_index("origin")["err"]
            a, b = a.align(b, join="inner")
            stat, p = dm_test(a.values, b.values, h)
            dm[f"{m1.split()[0]} vs {m2.split()[0]}, h={h}"] = dict(stat=round(float(stat), 3), p=round(float(p), 4))
    report["diebold_mariano"] = dm

    # 4. Full-sample fits: residual diagnostics + forecast to May 2027
    future = pd.date_range(s.index[-1] + pd.offsets.MonthBegin(), "2027-05-01", freq="MS")
    final = {}
    fcast_frames = []
    for name, spec in MODELS.items():
        res = fit(s, spec)
        resid = res.resid.iloc[sum(spec[1][1:2]) + spec[2][1] * 12 + 1:]
        lb = acorr_ljungbox(resid.dropna(), lags=[12, 24])["lb_pvalue"].values
        mean, lo, hi = forecast(res, spec, len(future))
        params = {k: round(float(v), 4) for k, v in res.params.items()}
        final[name] = dict(aic=round(float(res.aic), 1), ljung_box_p12=round(float(lb[0]), 4),
                           ljung_box_p24=round(float(lb[1]), 4), params=params,
                           may2027=round(float(mean[-1]), 2), may2027_lo=round(float(lo[-1]), 2),
                           may2027_hi=round(float(hi[-1]), 2),
                           implied_12m_infl_pct=round(float(100 * (mean[11] / s.dropna().iloc[-1] - 1)), 2))
        fcast_frames.append(pd.DataFrame({"model": name, "Time": future, "Forecast": mean, "Lower": lo, "Upper": hi}))
    pd.concat(fcast_frames).to_csv(os.path.join(out_dir, "final_forecasts.csv"), index=False)
    report["final_fit"] = final
    report["last_obs"] = dict(date=str(s.dropna().index[-1].date()), cpi=float(s.dropna().iloc[-1]))

    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)

    pd.set_option("display.width", 200)
    print(f"Rolling-origin evaluation: {len(origins)} origins, h=1..{H}")
    print(overall.round(3).to_string())
    print("\nBy period:\n" + by_period.round(3).to_string())
    print("\nDiebold-Mariano (negative = first model more accurate):")
    for k, v in dm.items():
        print(f"  {k}: {v}")
    print("\n80/20 hold-out RMSE:", {k: round(v['rmse'], 2) for k, v in holdout.items()})
    print("\nFull-sample fits:")
    for k, v in final.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
