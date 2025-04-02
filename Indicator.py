import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from fredapi import Fred
import datetime
import warnings
import argparse
import logging
from functools import lru_cache
from math import erf, sqrt
from scipy.stats import norm
from typing import Tuple, List, Any

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

@lru_cache(maxsize=1)
def get_nasdaq_tickers() -> List[str]:
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/NASDAQ-100")
        for table in tables:
            if "Ticker" in table.columns:
                tickers = table["Ticker"].tolist()
                return [ticker.strip() for ticker in tickers]
        return tables[0]["Ticker"].tolist()
    except Exception as e:
        logging.error("Error fetching Nasdaq-100 tickers: %s", e)
        return []

class DataManager:
    def __init__(self, start_date: str, end_date: str, fred_api_key: str) -> None:
        self.start_date = start_date
        self.end_date = end_date
        self.fred = Fred(api_key=fred_api_key)

    @lru_cache(maxsize=32)
    def get_history(self, ticker: str) -> pd.DataFrame:
        try:
            return yf.Ticker(ticker).history(start=self.start_date, end=self.end_date)
        except Exception as e:
            logging.error("Error fetching history for %s: %s", ticker, e)
            return pd.DataFrame()

    @lru_cache(maxsize=32)
    def get_fred_series(self, series_code: str) -> pd.DataFrame:
        try:
            data = self.fred.get_series(series_code)
            df = pd.DataFrame(data, columns=[series_code]).dropna()
            df.index = pd.to_datetime(df.index)
            mask = (df.index >= pd.to_datetime(self.start_date)) & (df.index <= pd.to_datetime(self.end_date))
            return df.loc[mask]
        except Exception as e:
            logging.error("Error fetching FRED series %s: %s", series_code, e)
            return pd.DataFrame()

class Indicator:
    def __init__(self, data_manager: DataManager) -> None:
        self.dm = data_manager

    def calculate(self) -> float:
        raise NotImplementedError("Must implement calculate method")

    @staticmethod
    def scale_with_history(values: np.ndarray, current: float) -> float:
        values = np.array(values)
        if len(values) < 10:
            return Indicator.robust_percentile(values, current)
        m = np.mean(values)
        s = np.std(values)
        if s < 1e-8:
            return 50.0
        z = (current - m) / s
        sc = 0.5 * (1 + erf(z / sqrt(2))) * 100
        return max(0, min(100, sc))

    @staticmethod
    def robust_percentile(series: np.ndarray, val: float) -> float:
        if len(series) < 5:
            return 50.0
        return (np.sum(series < val) / len(series)) * 100

class MomentumIndicator(Indicator):
    def calculate(self) -> float:
        spy = self.dm.get_history("SPY")
        if spy.empty or len(spy) < 125:
            return 50.0
        spy["SMA125"] = spy["Close"].rolling(window=125).mean()
        spy = spy.dropna()
        if spy.empty:
            return 50.0
        current_price = spy["Close"].iloc[-1]
        sma = spy["SMA125"].iloc[-1]
        momentum_current = (current_price - sma) / sma
        momentum_series = ((spy["Close"] - spy["SMA125"]) / spy["SMA125"]).dropna().values
        return Indicator.scale_with_history(momentum_series, momentum_current)

class NewHighsLowsIndicator(Indicator):
    def calculate(self) -> float:
        tickers = get_nasdaq_tickers()
        ratios = []
        for ticker in tickers:
            df = self.dm.get_history(ticker)
            if df.empty:
                continue
            period_end = df.index[-1]
            half_year_date = period_end - pd.Timedelta(days=182)
            recent_df = df[df.index >= half_year_date]
            cond_half = 1 if (not recent_df.empty and (recent_df["Close"].iloc[-1] >= recent_df["Close"].max())) else 0
            cond_full = 1 if df["Close"].iloc[-1] >= df["Close"].max() else 0
            ratios.append((cond_half + cond_full) / 2)
        raw = np.mean(ratios) if ratios else 0.5
        pseudo_history = np.random.beta(5, 5, 52)
        return Indicator.scale_with_history(pseudo_history, raw)

class MarketBreadthIndicator(Indicator):
    def calculate(self) -> float:
        tickers = get_nasdaq_tickers()
        scores = []
        for ticker in tickers:
            df = self.dm.get_history(ticker)
            if df.empty or len(df) < 50:
                continue
            df["SMA50"] = df["Close"].rolling(window=50).mean()
            df = df.dropna()
            if df.empty:
                continue
            current_price = df["Close"].iloc[-1]
            sma50 = df["SMA50"].iloc[-1]
            std50 = df["Close"].rolling(window=50).std().iloc[-1]
            if std50 < 1e-8:
                score = 0.5
            else:
                score = norm.cdf((current_price - sma50) / std50)
            scores.append(score * 100)
        raw = np.mean(scores) if scores else 50.0
        pseudo_history = np.random.beta(5, 5, 52) * 100
        return Indicator.scale_with_history(pseudo_history, raw)

class PutCallIndicator(Indicator):
    def calculate(self) -> float:
        spy = yf.Ticker("SPY")
        try:
            if not spy.options:
                return 50.0
            expiration = spy.options[0]
            chain = spy.option_chain(expiration)
            put_vol = chain.puts["volume"].sum()
            call_vol = chain.calls["volume"].sum()
            if call_vol <= 0:
                return 50.0
            ratio = put_vol / call_vol
        except Exception as e:
            logging.error("Error computing Put/Call indicator: %s", e)
            return 50.0
        score = 100 / (1 + np.exp((ratio - 0.7) * 10))
        return score

class JunkBondIndicator(Indicator):
    def calculate(self) -> float:
        baa_df = self.dm.get_fred_series("BAA")
        treas_df = self.dm.get_fred_series("DGS10")
        if baa_df.empty or treas_df.empty:
            return 50.0
        common_dates = baa_df.index.intersection(treas_df.index)
        if len(common_dates) == 0:
            return 50.0
        last_date = common_dates[-1]
        baa_val = baa_df.loc[last_date, "BAA"]
        treas_val = treas_df.loc[last_date, "DGS10"]
        spread_current = baa_val - treas_val
        joined = baa_df.join(treas_df, how="inner")
        joined["spread"] = joined["BAA"] - joined["DGS10"]
        spread_history = joined["spread"].dropna().values
        scaled = Indicator.scale_with_history(spread_history, spread_current)
        return max(0, min(100, 100 - scaled))

class VIXIndicator(Indicator):
    def calculate(self) -> float:
        vix_df = self.dm.get_history("^VIX")
        if vix_df.empty:
            return 50.0
        current_vix = vix_df["Close"].iloc[-1]
        vix_history = vix_df["Close"].dropna().values
        scaled = Indicator.scale_with_history(vix_history, current_vix)
        return max(0, min(100, 100 - scaled))

class SafeHavenIndicator(Indicator):
    def calculate(self) -> float:
        tlt_df = self.dm.get_history("TLT")
        spy_df = self.dm.get_history("SPY")
        if tlt_df.empty or spy_df.empty:
            return 50.0
        common = tlt_df.index.intersection(spy_df.index)
        if len(common) < 30:
            return 50.0
        tlt_close = tlt_df.loc[common, "Close"]
        spy_close = spy_df.loc[common, "Close"]
        current_diff = ((tlt_close.iloc[-1] - tlt_close.iloc[-30]) / tlt_close.iloc[-30] -
                        (spy_close.iloc[-1] - spy_close.iloc[-30]) / spy_close.iloc[-30])
        diffs = []
        for i in range(30, len(common)):
            r_tlt = (tlt_close.iloc[i] - tlt_close.iloc[i-30]) / tlt_close.iloc[i-30]
            r_spy = (spy_close.iloc[i] - spy_close.iloc[i-30]) / spy_close.iloc[i-30]
            diffs.append(r_tlt - r_spy)
        diffs = np.array(diffs)
        scaled = Indicator.scale_with_history(diffs, current_diff)
        return max(0, min(100, 100 - scaled))

class CompositeSentiment:
    def __init__(self, data_manager: DataManager, weights: List[float] = None) -> None:
        self.dm = data_manager
        self.indicators = [
            MomentumIndicator(self.dm),
            NewHighsLowsIndicator(self.dm),
            MarketBreadthIndicator(self.dm),
            PutCallIndicator(self.dm),
            JunkBondIndicator(self.dm),
            VIXIndicator(self.dm),
            SafeHavenIndicator(self.dm)
        ]
        self.weights = weights if weights is not None else [1.0] * len(self.indicators)

    def compute(self) -> Tuple[float, List[float]]:
        values = []
        for ind in self.indicators:
            try:
                values.append(ind.calculate())
            except Exception as e:
                logging.error("Error computing indicator %s: %s", type(ind).__name__, e)
                values.append(50.0)
        composite = np.average(values, weights=self.weights)
        composite = max(0, min(100, composite))
        return composite, values

def gauge_plot(value: float) -> None:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": "Composite Sentiment"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "blue"},
            "steps": [
                {"range": [0, 30], "color": "red"},
                {"range": [30, 70], "color": "yellow"},
                {"range": [70, 100], "color": "green"}
            ]
        }
    ))
    fig.show()

def main() -> None:
    parser = argparse.ArgumentParser(description="Compute composite market sentiment.")
    parser.add_argument("--fred_key", type=str, default="054d79a6dffb592fd462713e98e04d85", help="FRED API key")
    parser.add_argument("--start_date", type=str, default=None, help="Start date in YYYY-MM-DD format (default: 3 years ago)")
    parser.add_argument("--end_date", type=str, default=None, help="End date in YYYY-MM-DD format (default: yesterday)")
    args = parser.parse_args()
    today = datetime.date.today()
    default_end = today - datetime.timedelta(days=1)
    default_start = default_end - datetime.timedelta(days=3 * 365)
    start_date = args.start_date if args.start_date else str(default_start)
    end_date = args.end_date if args.end_date else str(default_end)
    logging.info("Using date range from %s to %s", start_date, end_date)
    dm = DataManager(start_date, end_date, args.fred_key)
    composite = CompositeSentiment(dm)
    comp_value, individual_values = composite.compute()
    indicator_names = ["Momentum", "NewHighsLows", "MarketBreadth", "PutCall", "JunkBond", "VIX", "SafeHaven"]
    for name, val in zip(indicator_names, individual_values):
        print(f"{name:15s}: {round(val, 2)}")
    print("Composite Sentiment:", round(comp_value, 2))
    gauge_plot(comp_value)

if __name__ == "__main__":
    main()
