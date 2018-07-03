import json
import os
import time

import keras
import numpy as np
import pandas as pd
import scipy.stats

import food_search_env
import utils
from experiment_commons import ExperimentParams, AgentParams
from food_search_env import FoodSearch
from policy_gradient import LinearPGAgent
from utils import get_git_commit_hash


def run_and_write(output_dir_path, experiment_params, agent_params, random_seed):
    output_dir_path += (f'_obhso{experiment_params.obs_horizon}'
                        f'_e{experiment_params.n_episodes}')

    if os.path.exists(output_dir_path):
        raise ValueError(f'path {output_dir_path} already exists!')

    os.makedirs(output_dir_path)
    run_result = run(experiment_params, agent_params, random_seed)

    for entry in ['agent_params', 'final_model_weights', 'experiment_info']:
        path = os.path.join(output_dir_path, f'{entry}.json')
        content = utils.jsonify_lists_dicts_nparrays(run_result[entry])
        with open(path, 'w') as f:
            json.dump(content, f)

    pd.DataFrame(run_result['results']['episode_metrics'], ).to_csv(
        os.path.join(output_dir_path, 'episode_metrics.csv')
    )

    for result_key in run_result['results']:
        if result_key == 'episode_metrics':
            continue
        path = os.path.join(output_dir_path, f'{result_key}.npy')
        data = np.asarray(run_result['results'][result_key])
        if data.dtype == np.float64:
            data = data.astype(np.float32)
        np.save(path, data)


def _get_model_weight_metrics(model: keras.models.Model,
                              obs_horizon,
                              n_channels,
                              n_noise_channels):
    weight_matrix = model.get_weights()[0]
    W_variance = np.var(weight_matrix)
    weight_tensor = food_search_env._unflatten_weight_matrix(weight_matrix,
                                                             obs_horizon,
                                                             n_channels,
                                                             n_actions=4)
    noise_part = weight_tensor[:, :, -n_noise_channels:, :]
    W_noise_part_variance = np.var(noise_part)
    outside_part = np.r_[
        weight_tensor[:, 0, :].ravel(),
        weight_tensor[:, -1, :].ravel(),
        weight_tensor[0, 1:-1, :].ravel(),
        weight_tensor[-1, 1:-1, :].ravel()
    ]
    W_outside_part_variance = np.var(outside_part)

    inside_part = weight_tensor[1:-1, 1:-1, :]
    W_inside_part_variance = np.var(inside_part)

    # n_inside = inside_part.size
    # n_outside = weight_matrix.size
    # assert n_outside == weight_tensor.size
    # n_total = n_inside + n_outside
    # W_outside_part_variance = (n_total / n_inside)*W_variance - (n_outside/n_inside) * W_inside_part_variance

    return {
        'W_variance': W_variance,
        'W_noise_part_variance': W_noise_part_variance,
        'W_outside_part_variance': W_outside_part_variance,
        'W_inside_part_variance': W_inside_part_variance
    }


def run(experiment_params: ExperimentParams,
        agent_params: AgentParams,
        random_seed=None):
    git_commit_hash = get_git_commit_hash()
    start_timestamp = utils._get_timestamp()
    if random_seed is None:
        random_seed = int(1e5*time.time())%(2 ** 32)
    rng = np.random.RandomState(random_seed)

    env = FoodSearch(obs_horizon=experiment_params.obs_horizon,
                     n_noise_channels=experiment_params.n_noise_channels,
                     rng=rng)
    agent = LinearPGAgent(env.action_space,
                          env.observation_space,
                          **agent_params._asdict())

    episode_metrics = []
    all_action_entropies = []
    episode_tic = time.time()
    for i_episode in range(experiment_params.n_episodes):
        pre_state = env.reset()
        episode_return = 0.0
        episode_step = 0
        while True:
            action_probas = agent.get_action_probas(np.array([pre_state]))[0]
            action = rng.choice(np.arange(env.action_space.n),
                                p=action_probas)
            post_state, reward, done, info = env.step(action)
            episode_return += reward
            episode_step += 1
            agent.remember(pre_state, action, reward)
            all_action_entropies.append(scipy.stats.entropy(action_probas))
            pre_state = post_state
            if done:
                train_metrics = agent.train()
                model_weight_metrics = _get_model_weight_metrics(agent.policy_model,
                                                                 experiment_params.obs_horizon,
                                                                 env.n_channels,
                                                                 env.n_noise_channels)
                episode_toc = time.time()
                episode_metrics.append(
                    {'episode_return': episode_return,
                     'episode_length': episode_step,
                     'episode_real_time': episode_toc - episode_tic,
                     **model_weight_metrics,
                     **train_metrics})
                episode_tic = episode_toc
                break
    return {
        'experiment_info': {
            'git_commit_hash': git_commit_hash,
            'start_time': start_timestamp,
            'random_seed': random_seed,
            'obs_horizon': experiment_params.obs_horizon,
            'n_episodes': experiment_params.n_episodes,
            **experiment_params._asdict()  # Prevent forgetting to include info
        },
        'agent_params': agent_params._asdict(),
        'results': {
            'episode_metrics': episode_metrics,
            'all_action_entropies': all_action_entropies
        },
        'final_model_weights': agent.policy_model.get_weights(),
    }
