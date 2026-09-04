import numpyro
import numpyro.distributions as dist
import jax.numpy as jnp
from jax import random

def model():
    states = 3
    features = 6
    with numpyro.plate('states_plate', states):
        with numpyro.plate('features_plate', features):
            mu = numpyro.sample('mu', dist.Normal(0, 1))
            print("mu shape:", mu.shape)

from numpyro.infer import Predictive
predictive = Predictive(model, num_samples=1)
predictive(random.PRNGKey(0))
