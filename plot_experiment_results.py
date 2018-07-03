import os
from glob import glob

import numpy as np
import matplotlib.style
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.style.use('seaborn')
N_ACTIONS = 4

def plot_experiment_results(base_dir_path, smoothing_window):
    for p in glob(os.path.join(base_dir_path, '**', 'agent_params.json'), recursive=True):
        experiment_dir = os.path.split(p)[0]
        result_path = os.path.join(experiment_dir, 'results.png')
        # if os.path.exists(result_path):
        #     print(f'Skipping {result_path}')
        #     continue

        print(f'Plotting {result_path}')
        episode_metrics = pd.read_csv(os.path.join(experiment_dir,
                                                   'episode_metrics.csv'))
        ep_returns = episode_metrics['episode_return']
        ep_lengths = episode_metrics['episode_length']

        action_entropies = np.load(os.path.join(experiment_dir,
                                                'all_action_entropies.npy'))

        print('ep_returns.shape', ep_returns.shape)
        fig = plot_results(ep_returns, ep_lengths, action_entropies, smoothing_window)
        fig.savefig(result_path, dpi=300)


def _get_rolling_mean(raw_values, smoothing_window):
    return pd.Series(raw_values).rolling(smoothing_window, center=True).mean()


def plot_results(ep_returns,
                 ep_lengths,
                 action_entropies,
                 smoothing_window):
    nrows = 4
    fig, axs = plt.subplots(nrows=nrows, ncols=1, figsize=(5*1, 3*nrows))
    smoothed_ep_returns = _get_rolling_mean(ep_returns, smoothing_window)
    axs[0].plot(smoothed_ep_returns)
    axs[0].set_xlabel('Episode')
    axs[0].set_ylabel('Return')

    ep_starts = np.cumsum(ep_lengths)
    axs[1].plot(ep_starts, smoothed_ep_returns.values)
    axs[1].set_xlabel('Step')
    axs[1].set_ylabel('Return')

    smoothed_ep_lengths = _get_rolling_mean(ep_lengths, smoothing_window)
    axs[2].plot(smoothed_ep_lengths)
    axs[2].set_xlabel('Episode')
    axs[2].set_ylabel('Episode length')

    smoothed_action_entropies = _get_rolling_mean(action_entropies, smoothing_window)
    rolling_action_entropies = pd.Series(action_entropies).rolling(smoothing_window, center=True)

    axs[3].plot(smoothed_action_entropies, color='k')
    axs[3].fill_between(smoothed_action_entropies.index,
                        rolling_action_entropies.quantile(0.05),
                        rolling_action_entropies.quantile(0.95),
                        alpha=0.2, color='k')
    axs[3].fill_between(smoothed_action_entropies.index,
                        rolling_action_entropies.quantile(0.25),
                        rolling_action_entropies.quantile(0.75),
                        alpha=0.5, color='k')

    axs[3].set_xlabel('Step')
    axs[3].set_ylabel('Action entropy')
    axs[3].set_ylim(0, 1.05 * np.log(N_ACTIONS))

    fig.tight_layout()
    return fig


if __name__ == '__main__':
    plot_experiment_results('experiments', smoothing_window=100)
