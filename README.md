# Composite Market Sentiment Indicator

This project computes a composite market sentiment indicator by aggregating several technical and market-based indicators.


## Overview

The tool fetches historical market data for key instruments (e.g., SPY, VIX, TLT) and a subset of S&P 500 companies. It calculates individual indicators such as:

- **Momentum Indicator:** Compares SPY’s current price to its 125-day Simple Moving Average.
- **New Highs/Lows Indicator:** Assesses the frequency of new highs and lows among S&P 500 stocks.
- **Market Breadth Indicator:** Uses a 50-day moving average and standard deviation to determine market trends.
- **Put/Call Ratio Indicator:** Evaluates the options market sentiment using SPY option chains.
- **Junk Bond Indicator:** Computes the yield spread between BAA-rated bonds and 10-year Treasuries.
- **VIX Indicator:** Measures volatility based on the VIX index.
- **Safe Haven Indicator:** Compares performance differences between safe-haven assets (TLT) and the broader market (SPY).

A weighted average of these indicators is computed to produce a composite sentiment score ranging from 0 to 100.


Happy Investing!
