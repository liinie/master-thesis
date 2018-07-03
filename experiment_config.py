"""
Config file for experiments.
super_experiment.py copies this file into the output directory
"""
import itertools

import numpy as np

from experiment_commons import ExperimentParams, AgentParams

RANDOM_SEED = 1234
OBS_HORIZONS = [1, 2, 2, 2, 2]
N_EPISODES = 5*10 ** 4

rng = np.random.RandomState(RANDOM_SEED)


def experiment_param_generator():
    obs_horizon_cycle = itertools.cycle(OBS_HORIZONS)
    while True:
        yield ExperimentParams(obs_horizon=next(obs_horizon_cycle),
                               n_episodes=N_EPISODES,
                               n_noise_channels=8)


def agent_param_generator():
    use_regmask_cycle = itertools.cycle([False, False, True, True, True])
    regmask_start_val_cycle = itertools.cycle([0.01, 0.1, 1.0])

    while True:
        if rng.uniform(0, 1) < 0.75:
            entropy_boost = rng.uniform(0.0, 0.5)
            entropy_boost_half_life = N_EPISODES * 10 ** np.random.uniform(-1, 1)
            entropy_boost_decay = 1 / (np.log(2) * entropy_boost_half_life)
        else:
            entropy_boost = 0.0
            entropy_boost_decay = 0.0

        if next(use_regmask_cycle):
            regmask_start_val = next(regmask_start_val_cycle)
            regmask_anneal_episodes = int(10 ** 3.4)
        else:
            regmask_start_val = None
            regmask_anneal_episodes = None

        yield AgentParams(
            gamma=rng.uniform(0.2, 0.8),
            learning_rate=10 ** rng.uniform(-3.0, -0.5),
            momentum=0.0,
            init_stdev=0.0,
            entropy_boost=entropy_boost,
            entropy_boost_decay=entropy_boost_decay,
            optimizer='sgd',
            regmask_start_val=regmask_start_val,
            regmask_anneal_episodes=regmask_anneal_episodes
        )
