"""
Starts multiple processes which execute single runs
"""
import logging
import os
import utils

import argparse
import json
import multiprocessing
import shutil

import time

logging.basicConfig(filename=os.path.join(
    os.path.dirname(__file__),
    'logs',
    f'{utils._get_timestamp()}_super_experiment.log'),
    level=logging.DEBUG)
logger = logging.getLogger(__name__)

import experiment_config
import single_run
import traceback
import pympler.tracker


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_processes', type=int,
                        help='Number of parallel processes.',
                        required=True)
    parser.add_argument('--n_runs_per_process',
                        type=int,
                        help='Number of runs per process. '
                             'Set to -1 for indefinite running.',
                        default=-1)
    parser.add_argument('--description',
                        type=str,
                        required=True,
                        help='Short description of experiment')
    parser.add_argument('--random_seed',
                        type=int,
                        required=False,
                        default=None,
                        help='Random seed, leave empty to use system time')
    parser.add_argument('--allow_unclean',
                        type=bool,
                        help='Allow execution of experiment '
                             'if working directory is unclean',
                        default=False)

    args = parser.parse_args()
    return args


def runner(index,
           n_runs,
           parent_directory_path,
           init_random_seed,
           experiment_name,
           experiment_param_generator,
           agent_param_generator):
    """
    :param index:
    :param n_runs:
    :param parent_directory_path:
    :param init_random_seed:
    :param experiment_name:
    :param experiment_param_generator:
    :param agent_param_generator:
    :return:
    """
    logger = logging.getLogger(__name__)

    assert os.path.exists(parent_directory_path)
    timestamp = utils._get_timestamp()
    output_dir_path_prefix = os.path.join(parent_directory_path,
                                          f'{timestamp}_{experiment_name}_{index}')

    if n_runs is None:
        n_runs = float('inf')
    run_counter = 0
    experiment_params_iterable = experiment_param_generator()
    agent_params_iterable = agent_param_generator()
    memory_tracker = pympler.tracker.SummaryTracker()

    logger.info(f'Experiment: {experiment_name}, Starting runner {index}')
    while run_counter < n_runs:
        logger.info(f'Runner {index}: Starting run {run_counter} (n_runs={n_runs})')
        memory_tracker.print_diff()
        output_dir = f'{output_dir_path_prefix}_{run_counter}'
        experiment_params = next(experiment_params_iterable)
        agent_params = next(agent_params_iterable)
        random_seed = utils.reproducible_hash(
            f'{init_random_seed}_{index}_{run_counter}')%(2**32)
        logger.info(f'random seed: {random_seed}')
        try:
            single_run.run_and_write(output_dir,
                                     experiment_params,
                                     agent_params,
                                     random_seed)
        except Exception as e:
            logger.error(f'Runner {index}: Exception {e} was raised. '
                         f'Details written to exceptions.txt',
                         exc_info=True)
            exception_with_tb = traceback.format_exc()
            exceptions_path = os.path.join(parent_directory_path,
                                           'exceptions.txt')
            with open(exceptions_path, 'a') as f:
                f.write(
                    f'{utils._get_timestamp()}:\n'
                    f'index={index}\n'
                    f'run_counter={run_counter}\n'
                    f'{exception_with_tb}\n'
                    f'-------------------------\n'
                )
        run_counter += 1


def check_unclean_working_directory():
    if not utils.git_working_directory_clean():
        raise ValueError('There unstaged changes. '
                         'This is problematic because the experiment may '
                         'not be reproducible if it is not tracked in git. '
                         'Please commit your changes or pass the option '
                         '--allow_unclean True for quick tests.')


def write_super_experiment_info(parent_directory_path, init_random_seed,
                                args):
    experiment_info = os.path.join(parent_directory_path,
                                   'super_experiment_info.json')
    with open(experiment_info, 'w') as f:
        json.dump({'git_commit_hash': utils.get_git_commit_hash(),
                   'init_random_seed': init_random_seed,
                   **vars(args)}, f)


def copy_experiment_config(parent_directory_path):
    dest_filename = os.path.split(experiment_config.__file__)[1] + '_'
    dest_path = os.path.join(parent_directory_path, dest_filename)
    shutil.copy(experiment_config.__file__, dest_path)


def main():
    args = get_args()
    timestamp = utils._get_timestamp()
    description = args.description.replace(' ', '_')

    if args.random_seed is None:
        init_random_seed = int(time.time()*1e5)%2**32
    else:
        init_random_seed = args.random_seed

    experiment_name = f'{timestamp}__{description}'
    if not args.allow_unclean:
        check_unclean_working_directory()
    else:
        logging.warning('Using --allow_unclean True is strongly discouraged.')
    parent_directory_path = os.path.join('experiments', experiment_name)
    if os.path.exists(parent_directory_path):
        raise ValueError(f'Directory {parent_directory_path} already exists!')
    os.makedirs(parent_directory_path)

    n_runs_per_process = args.n_runs_per_process
    if n_runs_per_process == -1:
        n_runs_per_process = float('inf')

    write_super_experiment_info(parent_directory_path,
                                init_random_seed,
                                args)
    copy_experiment_config(parent_directory_path)

    processes = []
    for i_process in range(args.n_processes):
        process = multiprocessing.Process(target=runner,
                                          name=f'awa_runner_{i_process}',
                                          args=(i_process,
                                                n_runs_per_process,
                                                parent_directory_path,
                                                init_random_seed,
                                                experiment_name,
                                                experiment_config.experiment_param_generator,
                                                experiment_config.agent_param_generator))
        process.start()
        processes.append(process)

    for process in processes:
        process.join()


if __name__ == '__main__':
    main()
