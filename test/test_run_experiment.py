import unittest

import numpy as np

import single_run
from experiment_commons import ExperimentParams, AgentParams


class TestRunExperiment(unittest.TestCase):
    def test_sanity_one_episode(self):
        experiment_params = ExperimentParams(obs_horizon=1,
                                             n_episodes=1,
                                             n_noise_channels=2)
        agent_params = AgentParams(gamma=1.0,
                                   learning_rate=1.0,
                                   momentum=0.0,
                                   init_stdev=0.0,
                                   entropy_boost=0.0,
                                   entropy_boost_decay=0.0,
                                   optimizer='sgd',
                                   regmask_start_val=0.0,
                                   regmask_anneal_episodes=1)
        results = single_run.run(experiment_params,
                                 agent_params,
                                 random_seed=1234)
        self.assertIn('experiment_info', results)
        self.assertIn('agent_params', results)
        self.assertIn('results', results)
        self.assertIn('final_model_weights', results)

        # noinspection PyTypeChecker
        self.assertGreater(np.sum(results['final_model_weights'][0]**2), 0)

        # only one episode, weights starting at 0, so the entropy should be constant and maximal
        entropies_ = results['results']['all_action_entropies']
        self.assertTrue(np.allclose(entropies_,
                                    np.log(4)))

        # one episode
        self.assertEqual(len(results['results']['episode_metrics']),
                         experiment_params.n_episodes)

    def test_seed_determinism(self):
        """
        Check that the same seed leads to the same outcomes
        """
        experiment_params = ExperimentParams(obs_horizon=2,
                                             n_episodes=10,
                                             n_noise_channels=2)
        agent_params = AgentParams(gamma=1.0,
                                   learning_rate=1.0,
                                   momentum=0.0,
                                   init_stdev=0.0,
                                   entropy_boost=0.0,
                                   entropy_boost_decay=0.0,
                                   optimizer='sgd',
                                   regmask_start_val=0.0,
                                   regmask_anneal_episodes=1)
        results_a = single_run.run(experiment_params,
                                   agent_params,
                                   random_seed=12345)
        results_b = single_run.run(experiment_params,
                                   agent_params,
                                   random_seed=results_a['experiment_info']['random_seed'])
        self.maxDiff = None
        for results in [results_a, results_b]:
            for m in results['results']['episode_metrics']:
                m['episode_real_time'] = 1.0

        self.assertEqual(results_a['results']['episode_metrics'],
                         results_b['results']['episode_metrics'])

        self.assertTrue(np.allclose(results_a['final_model_weights'][0],
                                    results_b['final_model_weights'][0]))

        self.assertTrue(np.allclose(results_a['final_model_weights'][1],
                                    results_b['final_model_weights'][1]))

    def test_seed_nondeterminism(self):
        """
        Check that the no specification of a seed leads to the different outcomes
        """
        experiment_params = ExperimentParams(obs_horizon=2,
                                             n_episodes=10,
                                             n_noise_channels=2)
        agent_params = AgentParams(gamma=1.0,
                                   learning_rate=1.0,
                                   momentum=0.0,
                                   init_stdev=0.0,
                                   entropy_boost=0.0,
                                   entropy_boost_decay=0.0,
                                   optimizer='sgd',
                                   regmask_start_val=0.0,
                                   regmask_anneal_episodes=1)
        results_a = single_run.run(experiment_params,
                                   agent_params)
        results_b = single_run.run(experiment_params,
                                   agent_params)

        self.assertNotEqual(results_a['results']['episode_metrics'],
                            results_b['results']['episode_metrics'])

        self.assertFalse(np.allclose(results_a['final_model_weights'][0],
                                     results_b['final_model_weights'][0]))

        self.assertFalse(np.allclose(results_a['final_model_weights'][1],
                                     results_b['final_model_weights'][1]))
