import gym.spaces
import numpy as np
from typing import Callable

from collections import namedtuple

EpisodeResult = namedtuple('EpisodeResult', 'ep_return n_steps')


def episode_rollout(env: gym.Env, policy: Callable[[np.ndarray], np.ndarray]):
    assert isinstance(env.action_space, gym.spaces.Discrete)

    obs = env.reset()
    episode_return = 0.0
    n_steps = 0
    while True:
        action_probas = policy(obs)
        action = np.random.choice(np.arange(env.action_space.n), p=action_probas)
        obs, reward, done, _ = env.step(action)
        episode_return += reward
        n_steps += 1
        if done:
            break
    return EpisodeResult(episode_return, n_steps)


def determine_policy_returns(env, policy, n_rollouts):
    returns = []
    for _ in range(n_rollouts):
        result = episode_rollout(env, policy)
        returns.append(result.ep_return)
    return np.array(returns)


