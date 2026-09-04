import pandas as pd
import numpy as np
import yfinance as yf
from pypfopt import expected_returns, risk_models
from pypfopt.efficient_frontier import EfficientFrontier


predictions = pd.read_csv("predictions.csv", index_col=0, parse_dates=True)

tickers = ["SPY", "TLT", "GLD"]

print("downloading data..")

# Downloading the daily prices of our portfolio
portfolio_data = yf.download(tickers, start="2005-01-01", end="2026-01-01")["Close"]
portfolio_data = portfolio_data.dropna()

# Using the VIX value to separate the time periods into different trading regimes
vix_data = yf.download("^VIX", start="2005-01-01", end="2026-01-01")["Close"]

# We split the data into regimes using the VIX values
vix_33 = vix_data.quantile(0.33).iloc[0]
vix_66 = vix_data.quantile(0.66).iloc[0]

# For bull days
bull_days = portfolio_data[vix_data.squeeze() <= vix_33]
bull_cov = risk_models.sample_cov(bull_days)
print("Bull Covariance Matrix")
print(bull_cov)

# For bear days
bear_days = portfolio_data[(vix_data.squeeze() > vix_33) & (vix_data.squeeze() <= vix_66)]
bear_cov = risk_models.sample_cov(bear_days)
print("Bear Covariance Matrix")
print(bear_cov)

# For crisis days
crisis_days = portfolio_data[vix_data.squeeze() > vix_66]
crisis_cov = risk_models.sample_cov(crisis_days)
print("Cris Covariance Matrix")
print(crisis_cov)

# The average expected returns for the 3 regimes to jumpstart the optimizer
bull_mu = expected_returns.mean_historical_return(bull_days)
bear_mu = expected_returns.mean_historical_return(bear_days)
crisis_mu = expected_returns.mean_historical_return(crisis_days)

weights = []

for date, probs in predictions.iterrows():
    prob_bull = probs["Bull"]
    prob_bear = probs["Bear"]
    prob_crisis = probs["Crisis"]

    # Custom Covariance matrix for each month
    blended_cov = (prob_bull*bull_cov) + (prob_bear*bear_cov) + (prob_crisis*crisis_cov)
    # Custom mu for each month
    blended_mu = (prob_bull*bull_mu) + (prob_bear*bear_mu) + (prob_crisis*crisis_mu)

    # The optimizer
    ef = EfficientFrontier(blended_mu, blended_cov)

    # Maximize the Sharpe ratio
    try:
        raw_weights = ef.max_sharpe()

        clean_weights = ef.clean_weights()
    except:
        # If there is an error, default to equal weights
        clean_weights = {"SPY": 0.33, "TLT": 0.33, "GLD": 0.33}
    weights.append(clean_weights)

weights_df = pd.DataFrame(weights, index=predictions.index)

print(weights_df.head(10))

weights_df.to_csv("portfolio_weights.csv")