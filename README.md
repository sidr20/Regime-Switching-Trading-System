# Regime-Switching Quantitative Trading System

An institutional-grade algorithmic trading engine that uses **Bayesian Hidden Markov Models (HMMs)** and **Mean-Variance Optimization** to dynamically allocate a multi-asset portfolio based on real-time market regimes.

By shifting allocations between Equities, Gold, and Treasury Bonds depending on the predicted market state (Bull, Bear, or Crisis), this system significantly reduces drawdown during crashes and maximizes risk-adjusted returns (Sharpe Ratio).

Tested over a 15-year out-of-sample period with strict transaction costs and slippage, the strategy achieved a **398% Net Return**, vastly outperforming a standard Buy-and-Hold SPY benchmark.

---

## 🧠 System Architecture

The pipeline is split into three core, modular components:

### 1. The Brain (`main.py`)
- **Bayesian HMM:** Built with **JAX** and **NumPyro**, it ingests macroeconomic data and the VIX (Volatility Index) to probabilistically forecast the current market regime. 
- **Walk-Forward Validation:** Uses MCMC (Markov Chain Monte Carlo) sampling over a rolling 5-year window to generate purely out-of-sample monthly probabilities, eliminating look-ahead bias and preventing overfitting.

### 2. The Decision Maker (`portfolio_optimizer.py`)
- **Regime-Conditioned Blending:** Takes the AI's monthly probabilities (e.g., 90% Bull, 10% Bear) and blends historical regime covariance matrices into a custom, forward-looking expected behavior matrix.
- **Mean-Variance Optimization:** Uses **PyPortfolioOpt** to maximize the Sharpe Ratio against the custom matrix, outputting the exact optimal asset weights for SPY, GLD, and TLT for the upcoming month.

### 3. The Reality Check (`backtester.py`)
- **Event-Driven Simulation:** Uses **VectorBT** to simulate the daily drift of portfolio weights and executes rebalancing trades at the end of every month.
- **Friction & Fee-Bleed Prevention:** Enforces a strict **5% Rebalancing Threshold** (only trading when target weights deviate by >5% from drifted weights) and applies a realistic 0.1% transaction fee + 0.1% slippage penalty per trade.
- **Tear Sheet Generation:** Uses **QuantStats** to generate an interactive HTML report detailing the strategy's Sharpe, Sortino, Calmar, and Underwater plots.

---

## 🛠️ Key Technologies
- **JAX & NumPyro:** Probabilistic programming and fast MCMC sampling.
- **VectorBT:** High-performance, Pandas-based backtesting engine.
- **PyPortfolioOpt:** Modern portfolio theory and convex optimization.
- **QuantStats:** Institutional portfolio analytics and reporting.
- **yfinance & pandas-datareader:** Market and macroeconomic data ingestion.

---

## 🚀 Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com/sidr20/Regime-Switching-Trading-System.git
   cd Regime-Switching-Trading-System
   ```

2. **Install dependencies:**
   ```bash
   pip install jax numpyro vectorbt quantstats yfinance pandas_datareader PyPortfolioOpt
   ```

3. **Run the Pipeline:**
   Execute the scripts in order to download data, train the model, and backtest the strategy:
   
   ```bash
   # 1. Train the HMM and predict regime probabilities (Takes ~1 hour due to MCMC sampling)
   python main.py
   
   # 2. Optimize the portfolio weights for each month
   python portfolio_optimizer.py
   
   # 3. Run the VectorBT backtest and generate the Tear Sheet
   python backtester.py
   ```

4. **View Results:**
   Open `AI_Tear_Sheet.html` in your web browser to view the final interactive dashboard.

---

## 📈 Performance Snapshot (15-Year Out-Of-Sample)
- **Initial Investment:** `$100,000.00`
- **Final Net Value:** `$398,244.76`
- **Transaction Costs:** 0.1% Fees + 0.1% Slippage (Included in Net Value)

*Disclaimer: This repository is for educational purposes only and does not constitute financial advice.*
