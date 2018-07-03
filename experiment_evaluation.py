import logging
import os
from glob import glob
import itertools
from typing import List

import matplotlib.style
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from collections import defaultdict
from pandas.api.types import is_numeric_dtype
import json
from tqdm import tqdm

import experiment_commons as ec
import food_search_env
import policy_gradient

logger = logging.getLogger(__name__)

matplotlib.style.use('seaborn')

# EXPERIMENT_NAME = '2017-09-30_23-48-08__simple_variable_noise_independent_channels'
EXPERIMENT_NAME = '2017-10-03_22-16-26__three_regmask_values'


def evaluate(super_directory):
    ep_metrics_paths = list(glob(os.path.join(super_directory, '**', 'episode_metrics.csv')))
    print(f'Found {len(ep_metrics_paths)} runs for experiment {super_directory}')

    analysis_directory = os.path.join(super_directory, '_analysis')
    try:
        os.makedirs(analysis_directory)
    except FileExistsError:
        print('Analysis has already been performed. Overwriting results.')

    all_episode_metrics = []
    all_agent_params = []
    all_experiment_infos = []
    all_weights = []
    for ep_metrics_path in tqdm(ep_metrics_paths,
                                desc='Reading episode_metrics.csv files.'):
        run_path = os.path.split(ep_metrics_path)[0]
        episode_metrics = pd.read_csv(ep_metrics_path)
        all_episode_metrics.append(episode_metrics)
        agent_params_path = os.path.join(run_path, 'agent_params.json')
        with open(agent_params_path, 'r') as f:
            agent_params = json.load(f)
        all_agent_params.append(agent_params)

        experiment_info_path = os.path.join(run_path, 'experiment_info.json')
        with open(experiment_info_path, 'r') as f:
            experiment_info = json.load(f)
            experiment_info['use_regmask'] = agent_params.get('regmask_start_val') is not None
        all_experiment_infos.append(experiment_info)

        weights_path = os.path.join(run_path, 'final_model_weights.json')
        with open(weights_path, 'r') as f:
            weights = [np.array(w) for w in json.load(f)]
        all_weights.append(weights)

    for info_keys in [
        # ('n_noise_channels',),
        # ('obs_horizon',)
        # ('obs_horizon', 'use_regmask'),
        ('obs_horizon', 'regmask_start_val')
    ]:

        episode_metrics_by_info = defaultdict(list)
        agent_params_by_info = defaultdict(list)
        weights_by_info = defaultdict(list)
        experiment_params_by_info = defaultdict(list)
        for (experiment_info, agent_params, episode_metrics, weights) \
                in zip(all_experiment_infos,
                       all_agent_params,
                       all_episode_metrics,
                       all_weights):

            all_params = {**experiment_info, **agent_params}

            if any(info_key not in all_params for info_key in info_keys):
                continue

            key = tuple(all_params[info_key] for info_key in info_keys)
            episode_metrics_by_info[key].append(episode_metrics)
            weights_by_info[key].append(weights)
            agent_params_by_info[key].append(agent_params)

            experiment_params = ec.ExperimentParams(experiment_info['obs_horizon'],
                                                    experiment_info['n_episodes'],
                                                    experiment_info.get('n_noise_channels',
                                                                        # Default used to be 2
                                                                        2))
            experiment_params_by_info[key].append(
                experiment_params)

        if not episode_metrics_by_info:
            logger.warning(f'No data found for info keys {info_keys}! Skipping...')
            continue

        try:
            info_values = sorted(episode_metrics_by_info.keys())
        except Exception:
            info_values = episode_metrics_by_info.keys()

        for info_value in info_values:
            print(f'plot analysis for {_keys_to_str(info_keys)} {info_value}...')

            prefix = f'{_keys_to_str(info_keys)}_{info_value}'
            plot_correlations(episode_metrics_by_info[info_value],
                              agent_params_by_info[info_value],
                              analysis_directory,
                              prefix)
            plot_mean_return_progress(episode_metrics_by_info[info_value],
                                      analysis_directory,
                                      prefix,
                                      best_fractions=[0.05, 0.1, 0.3])
            plot_columns(episode_metrics_by_info[info_value],
                         ['undiscounted_entropy_loss'],
                         analysis_directory,
                         prefix)

            if 'W_inside_part_variance' in all_episode_metrics[0].columns:
                plot_columns(episode_metrics_by_info[info_value],
                             ['W_inside_part_variance', 'W_outside_part_variance'],
                             analysis_directory,
                             prefix,
                             xscale='log')

            make_videos(episode_metrics_by_info[info_value],
                        weights_by_info[info_value],
                        experiment_params_by_info[info_value],
                        analysis_directory,
                        prefix)

        for best_fraction, best_of_k, xscale, plot_eventual in itertools.product(
                [0.05, 0.1, 0.3],
                [True, False],
                ['linear', 'log'],
                [False]):
            plot_mean_return_progress_by_info(episode_metrics_by_info,
                                              analysis_directory,
                                              best_fraction,
                                              best_of_k,
                                              xscale,
                                              file_prefix=f'{_keys_to_str(info_keys)}_',
                                              info_key=_keys_to_str(info_keys),
                                              plot_eventual=plot_eventual)


def _keys_to_str(keys):
    if not (isinstance(keys, tuple) or isinstance(keys, list)):
        return keys
    return '-'.join(str(k) for k in keys)


def plot_correlations(all_episode_metrics, all_agent_params, analysis_directory, file_prefix):
    scores = [episode_metrics['episode_return'].sum()
              for episode_metrics in all_episode_metrics]
    param_df: pd.DataFrame = pd.concat([pd.Series(scores, name='score'),
                                        pd.DataFrame(all_agent_params)], axis=1)
    for column in param_df.columns:
        if column == 'score' or not is_numeric_dtype(param_df[column]):
            continue
        fig, ax = correlate_scores(param_df, column, 'score')
        path = os.path.join(analysis_directory, f'{file_prefix}_correlation_{column}_score.png')
        fig.savefig(path)
        print(f'saved {path}')
        plt.close(fig)


def correlate_scores(df, score_x, score_y):
    fig, ax = plt.subplots()
    sns.regplot(x=score_x, y=score_y, data=df, ax=ax)
    return fig, ax


def plot_mean_return_progress(all_episode_metrics, analysis_directory, file_prefix,
                              best_fractions):
    # -- plot mean returns ---
    fig, ax = plt.subplots(figsize=(8, 7))
    plot_summary(ax,
                 all_episode_metrics,
                 'episode_return',
                 rolling_window_size=500,
                 label='mean of all')
    for best_fraction in best_fractions:
        best_episode_metrics = select_best_fraction(
            all_episode_metrics,
            'episode_return',
            fraction=best_fraction,
            score=lambda series: np.sum(series),
            best_of_k_mode=False
        )
        plot_summary(ax,
                     best_episode_metrics,
                     'episode_return',
                     rolling_window_size=500,
                     label=f'best overall {100*best_fraction:.3}%')
        # -- plot mean returns of 30 runs with best eventual performance ---
        best_episode_metrics = select_best_fraction(
            all_episode_metrics,
            'episode_return',
            fraction=best_fraction,
            score=lambda series: np.sum(series[-2000:]),
            best_of_k_mode=False
        )
        plot_summary(ax,
                     best_episode_metrics,
                     'episode_return',
                     rolling_window_size=500,
                     label=f'best eventual {100*best_fraction:.3}%'
                     )
    ax.legend(loc=0)
    path = os.path.join(analysis_directory, f'{file_prefix}_mean_episode_return.png')
    fig.savefig(path,
                dpi=300)
    print(f'saved {path}')
    plt.close(fig)


def plot_mean_return_progress_by_info(all_episode_metrics_by_info, analysis_directory,
                                      best_fraction, best_of_k, xscale, file_prefix, info_key, plot_eventual=True
                                      ):
    fig, ax = plt.subplots(figsize=(8, 7))
    prop_cycler = ax._get_lines.prop_cycler

    try:
        all_info_values = sorted(all_episode_metrics_by_info.keys())
    except Exception as e:
        all_info_values = all_episode_metrics_by_info.keys()

    for info_values in all_info_values:
        color = next(prop_cycler)['color']
        best_episode_metrics = select_best_fraction(
            all_episode_metrics_by_info[info_values],
            'episode_return',
            fraction=best_fraction,
            score=lambda series: np.sum(series),
            best_of_k_mode=best_of_k)
        plot_summary(ax,
                     best_episode_metrics,
                     'episode_return',
                     rolling_window_size=500,
                     label=f'{info_key} {_keys_to_str(info_values)}, best overall {100*best_fraction:.3}%',
                     color=color,
                     ls='-')
        # -- plot mean returns of best_k runs with best eventual performance ---

        if plot_eventual:
            best_episode_metrics = select_best_fraction(
                all_episode_metrics_by_info[info_values],
                'episode_return',
                fraction=best_fraction,
                score=lambda series: np.sum(series[-2000:]),
                best_of_k_mode=best_of_k
            )

            plot_summary(ax,
                         best_episode_metrics,
                         'episode_return',
                         rolling_window_size=500,
                         label=f'{info_key} {_keys_to_str(info_values)}, best eventual {100*best_fraction:.3}%',
                         color=color,
                         ls='--'
                         )
    ax.legend(loc=0)
    ax.set_xscale(xscale)
    path = os.path.join(analysis_directory,
                        f'{file_prefix}_comparison_mean_episode_return_best_'
                        f'{best_fraction}_'
                        f'{"best_of_k" if best_of_k else "best_of_all"}_'
                        f'{xscale}_'
                        f'{"with_eventual" if plot_eventual else ""}.png')
    fig.savefig(path,
                dpi=300)
    print(f'saved {path}')
    plt.close(fig)


def plot_columns(all_episode_metrics, columns, analysis_directory, file_prefix, xscale='linear'):
    fig, ax = plt.subplots(figsize=(8, 7))
    for column in columns:
        plot_summary(ax,
                     all_episode_metrics,
                     column,
                     rolling_window_size=500,
                     label=column,
                     xscale=xscale)
    ax.legend(loc=0)
    path = os.path.join(analysis_directory, f'{file_prefix}_{"-".join(columns)}.png')
    fig.savefig(path,
                dpi=300)
    print(f'saved {path}')
    plt.close(fig)


def select_best_fraction(dfs, column_key, fraction, score, best_of_k_mode):
    """

    :param dfs:
    :param column_key:
    :param fraction: If float, interpret as fraction
    :param score: scoring function for dataframes
    :param best_of_k_mode: get "best out of k mode"
    :return:
    """
    assert isinstance(fraction, float)
    n = int(round(fraction*len(dfs)))

    if best_of_k_mode:
        scores = np.array([score(df[column_key]) for df in dfs])
        k = int(round(1/fraction))
        print(f'using best_k_mode. fraction={fraction}, k={k}, len(scores)={len(scores)}')
        kept_dfs = []
        for i in range(len(scores)//k):
            kept_dfs.append(max(dfs[i*k:(i + 1)*k],
                                key=lambda df: score(df[column_key])))
        return kept_dfs
    else:
        return sorted(dfs, key=lambda df: -score(df[column_key]))[:n]


def plot_summary(ax, dfs, column_key, rolling_window_size, error_region=True, xscale='linear', **plot_kwargs):
    # print('len(dfs)', len(dfs))
    # print('column_key', column_key)
    # print('dfs[0][column_key]', dfs[0][column_key])
    aligned_rows = pd.concat([df[column_key] for df in dfs], axis=1)
    mean_series = get_aggregate(np.mean, aligned_rows, rolling_window_size)
    # mean_series.plot(ax=ax, **plot_kwargs)
    _line, = ax.plot(mean_series, **plot_kwargs)
    ax.set_xscale(xscale)

    if error_region:
        color = _line.get_color()
        stdev_series = get_aggregate(np.std, aligned_rows, rolling_window_size)
        error_series = stdev_series/np.sqrt(len(dfs))
        ax.fill_between(stdev_series.index,
                        mean_series - 1*error_series,
                        mean_series + 1*error_series,
                        alpha=0.2,
                        color=color)

    ax.set_xlim(rolling_window_size/2, len(mean_series))
    # ax.set_xlim()


def get_aggregate(aggregate_fn, aligned_rows, rolling_window_size):
    agg_series = aggregate_fn(aligned_rows, axis=1)
    rolling_mean_series = agg_series.rolling(rolling_window_size, center=True).mean()
    # Make episode index start at 1 (better for log plots)
    rolling_mean_series.index = np.arange(len(rolling_mean_series)) + 1
    return rolling_mean_series


def make_videos(all_episode_metrics: List[pd.DataFrame],
                all_weights: List[List[np.ndarray]],
                all_experiment_params,
                analysis_directory: str,
                prefix: str):
    # select best run
    best_run_idx = np.argmax([m['episode_return'].sum() for m in all_episode_metrics])
    best_weights = all_weights[best_run_idx]
    best_experiment_params = all_experiment_params[best_run_idx]

    make_video(best_experiment_params,
               best_weights,
               n_episodes=5,
               filepath=os.path.join(analysis_directory, f'{prefix}_best.mp4'))


def make_video(experiment_params, weights_and_biases, n_episodes, filepath):
    env = food_search_env.FoodSearch(experiment_params.obs_horizon,
                                     experiment_params.n_noise_channels)
    pg_agent = policy_gradient.LinearPGAgent(env.action_space,
                                             env.observation_space,
                                             # IRRELEVANT:
                                             gamma=0.5, learning_rate=0.0, momentum=0.0,
                                             init_stdev=0.0, entropy_boost=0.0,
                                             entropy_boost_decay=0.0, optimizer='sgd',
                                             regmask_start_val=0.0,
                                             regmask_anneal_episodes=1)
    pg_agent.policy_model.set_weights(weights_and_biases)
    policy = pg_agent.get_policy()
    assert len(weights_and_biases) == 2, 'Assuming the model is linear, with weight matrix and bias vector'
    # clip = food_search_env.render_video(env, policy, n_episodes=n_episodes,
    #                                     weight_tensor=food_search_env._unflatten_weight_matrix(
    #                                         weights_and_biases[0],
    #                                         obs_horizon=experiment_params.obs_horizon,
    #                                         n_channels=experiment_params.n_noise_channels + 3,
    #                                         n_actions=4),
    #                                     bias_tensor=weights_and_biases[1])

    clip = food_search_env.render_video(env, policy, n_episodes=n_episodes,
                                        weight_tensor=None,
                                        bias_tensor=None)
    clip.write_videofile(filename=filepath)


if __name__ == '__main__':
    evaluate(os.path.join('experiments', EXPERIMENT_NAME))
