import pandas as pd
import numpy as np
import yfinance as yf
import vectorbt as vbt
import quantstats as qs

# Load the portfolio allocation data
weights = pd.read_csv("portfolio_weights.csv", index_col=0, parse_dates=True)

tickers = ["SPY", "GLD", "TLT"]
print("Downloading prices..")
# Downloading all the prices
prices = yf.download(tickers, start=weights.index[0], end=weights.index[-1])["Close"]

# Defaulting to equal weights
current_weights = np.array([0.33, 0.33, 0.33])
filtered_weights = []

# Only update the portfolio allocation if the new weight is more than 5% more 
# than what it is currently
for date, new_weights in weights.iterrows():
    if np.max(np.abs(new_weights.values - current_weights)) > 0.05:
        current_weights = new_weights.values

    filtered_weights.append(current_weights)

filtered_weights_df = pd.DataFrame(filtered_weights, index=weights.index, columns=weights.columns)

# Forward filling the monthly weights to it accounts for every trading day, 
# not just the month as a whole
target_weights = filtered_weights_df.reindex(prices.index, method="ffill")

print("Running VectorBT Simulation..")

# Setting up fake portfolio
portfolio = vbt.Portfolio.from_orders(
    prices,
    size=target_weights,
    size_type='targetpercent',
    fees=0.001,
    slippage=0.001,
    freq="1D",
    init_cash=100000,
    cash_sharing=True,
    group_by=True
)

returns = portfolio.returns()

print(f"\n=========================================")
print(f"Initial Investment: $100000")
print(f"Final Portfolio Value: ${portfolio.final_value():,.2f}")
print(f"=========================================\n")

print("Tear sheet generating")

qs.reports.html(returns, benchmark="SPY", output = "AI_Tear_Sheet.html", 
title="HMM Regime Portfolio vs SPY")

