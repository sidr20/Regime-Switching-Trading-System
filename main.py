from jax import random
from numpyro.infer import MCMC, NUTS
from enum import EnumMeta
import yfinance as yf
import pandas as pd
import datetime
import pandas_datareader.data as web
import numpy as np
import jax
from jax.scipy.special import logsumexp
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist


# Start and end dates for the market history
end_date = datetime.date.today()
start_date = end_date - datetime.timedelta(days=20*365)

# VIX tickers

market_tickers = ['SPY','TLT', 'GLD', '^VIX']

market_data = yf.download(market_tickers, start=start_date, end=end_date)['Close']

market_data.rename(columns={'^VIX': 'VIX'}, inplace=True)

macro_tickers = ['T10Y2Y', 'BAA10Y']

# Federal Reserve Data

macro_data = web.DataReader(macro_tickers, 'fred', start_date, end_date)
macro_data.rename(columns={'T10Y2Y': 'Yield-Spread',
'BAA10Y': 'Credit_Spread'}, inplace=True)

# Merging the two datasets

full_dataset = pd.merge(market_data, macro_data, left_index=True, right_index=True, how='outer')

cleaned_data = full_dataset.ffill().dropna()


# Log returns for the tradeable assets
assets = ['SPY', 'TLT', 'GLD', 'VIX']
for asset in assets:
    cleaned_data[f'{asset}_log'] = np.log(cleaned_data[asset] / cleaned_data[asset].shift(1))

# Standardizing the macro indicators
macros = ['Yield-Spread', 'Credit_Spread']
for macro in macros:
    cleaned_data[f'{macro}_Std'] = (cleaned_data[macro] - cleaned_data[macro].mean()) / cleaned_data[macro].std()


cleaned_data = cleaned_data.dropna()

# --- STEP 1: Data Preparation for NumPyro ---

features = ['SPY_log', 'TLT_log', 'GLD_log', 'VIX_log', 'Yield-Spread_Std', 'Credit_Spread_Std']

data_matrix = jnp.array(cleaned_data[features].values)

def hmm_model(data_matrix):
    states = 3
    features = data_matrix.shape[1]

    # 0 bias priors for the intial state and the transition probabilites
    initial_probs = numpyro.sample('initial_probs', dist.Dirichlet(jnp.ones(states)))

    with numpyro.plate('states', states):
        trans_probs = numpyro.sample('trans_probs', dist.Dirichlet(jnp.ones(states)))

    # Setting the prior distribution(with parameters) for the all of the states for each feature.
    with numpyro.plate('states_plate', states):
        with numpyro.plate('features_plate', features):

            mu = numpyro.sample('mu', dist.Normal(0.0,1.0))
            sigma = numpyro.sample('sigma', dist.HalfNormal(1.0))
            nu = numpyro.sample('nu', dist.Exponential(1.0))


    expanded_data = data_matrix[:, :, None]

    obs_dist = dist.StudentT(df=nu, loc=mu, scale=sigma)

    # Calculates the probability of the data happening accross all 6 features for 3 states
    emission_log_probs = obs_dist.log_prob(expanded_data).sum(axis=1)

    # Defining the forward step 
    def forward_step(prev_log_prob, curr_emission_log_prob):
        # Taking the probability of yesterdays states and adding the probability of transitioning
        # to todays states
        next_log_prob = logsumexp(prev_log_prob[:, None] + jnp.log(trans_probs), axis=0)

        curr_log_prob = next_log_prob + curr_emission_log_prob

        return curr_log_prob, None

    # Adding our initial guess of the probabilities to the initial data of day 0
    init_log_prob = jnp.log(initial_probs) + emission_log_probs[0]

    # Looping until the end of the 5 years
    final_log_prob, _ = jax.lax.scan(forward_step, init_log_prob, emission_log_probs[1:])

    # Summing the probabilities at the end of the 5 years
    total_log_prob = logsumexp(final_log_prob)

    # Putting the total score in the numpyro model
    numpyro.factor('hmm_likelihood', total_log_prob)


def label_switching(mcmc, features):
    # Getting all the posterior parameters
    posterior_samples = mcmc.get_samples()
    # To prevent 'within-chain' label switching mush, we just take the final converged sample
    mu_estimate = posterior_samples['mu'][-1]

    # Getting the index for just VIX
    vix_idx = features.index('VIX_log')
    # Getting the average VIX across all the samples for each state
    vix_means = mu_estimate[vix_idx, :]
    # Sorting the VIX lowest to highest
    sorted_states = jnp.argsort(vix_means)

    # Mapping each VIX value to a fixed state using indices
    states_map = {int(random_state): fixed_state for fixed_state, random_state in 
    enumerate(sorted_states)}
    
    return states_map, posterior_samples

def predict_oos_states(data_window, posterior_samples, states_map):
    # Getting the final converged sample instead of averaging 
    mu = posterior_samples['mu'][-1]
    sigma = posterior_samples['sigma'][-1]
    nu = posterior_samples['nu'][-1]

    trans_probs = posterior_samples['trans_probs'][-1]
    initial_probs = posterior_samples['initial_probs'][-1]

    # Going throught the 5 years again to find the state on the last day.
    data_expanded = data_window[:,:, None]
    obs_dist = dist.StudentT(df=nu, loc=mu, scale=sigma)
    emission_log_probs = obs_dist.log_prob(data_expanded).sum(axis=1)

    def forward_step(prev_log_prob, curr_emission_log_prob):

        next_log_prob = logsumexp(prev_log_prob[:, None] + jnp.log(trans_probs), axis=0)

        return next_log_prob + curr_emission_log_prob, None

    init_log_prob = jnp.log(initial_probs) + emission_log_probs[0]

    final_log_prob, _ = jax.lax.scan(forward_step, init_log_prob, emission_log_probs[1:])

    # Converting the log probabilities into a regular percentage
    final_probs = jnp.exp(final_log_prob - logsumexp(final_log_prob))

    # Multipying todays state with the transition probabilities for forecast tmw
    future_probs = jnp.dot(final_probs, trans_probs)

    # Mapping the probabilities to their fixed states
    fixed_future_probs = jnp.zeros(3)
    for random_state, fixed_state in states_map.items():
        fixed_future_probs = fixed_future_probs.at[fixed_state].set(future_probs[random_state])

    return fixed_future_probs

# 265(weekdays)*5(years)
window_size = 1260

# Every month(21 weekdays aprox)
step_size = 21

all_predictions = []

rand_key = random.PRNGKey(42)
kernel = NUTS(hmm_model)
mcmc = MCMC(kernel, num_warmup=1000, num_samples=2000)

print("Starting the forward walk backtest...")

# Looping through the 5 years
for start_idx in range(0, len(data_matrix) - window_size, step_size):
    end_idx = start_idx + window_size
    # 5-year chunk of data
    data_window = data_matrix[start_idx:end_idx]

    mcmc.run(rand_key, data_matrix=data_window)

    # Assign the label using our function
    states_map, posterior_samples = label_switching(mcmc, features)

    oos_probs = predict_oos_states(data_window, posterior_samples, states_map)
    print(f'Next months predicted probabilities:{oos_probs}')

    all_predictions.append(oos_probs)

# Getting all the predicted dates
dates = cleaned_data.index[window_size::step_size]

# Making a dataframe to store the predictions
results_df = pd.DataFrame(all_predictions, index=dates, columns=['Bull', 'Bear', 'Crisis'])

results_df.to_csv('predictions.csv')
