import unittest

import gym.spaces
import numpy as np

import policy_gradient as pg
from food_search_env import _unflatten_weight_matrix


def policy_setup():
    learning_rate = 1.0
    state_space = gym.spaces.Box(-1, 1, shape=(2,))
    action_space = gym.spaces.Discrete(2)
    agent = pg.LinearPGAgent(action_space=action_space,
                             state_space=state_space,
                             gamma=1.0,
                             learning_rate=learning_rate,
                             momentum=0.0,
                             init_stdev=0.0,
                             entropy_boost=0.0,
                             entropy_boost_decay=0.0,
                             optimizer='sgd',
                             regmask_start_val=None,
                             regmask_anneal_episodes=None)

    initial_weights = agent.policy_model.get_weights()
    return agent, initial_weights


class TestPolicyGradient(unittest.TestCase):
    def test_accumulate_rewards(self):
        cuml_rws = pg.LinearPGAgent.accumulate_rewards([1, 2, 3, 4], gamma=0.5)
        self.assertListEqual(cuml_rws.tolist(),
                             [3.25, 4.5, 5.0, 4.0])

        cuml_rws = pg.LinearPGAgent.accumulate_rewards([1, 0, 2, 0], gamma=0.5)
        self.assertListEqual(cuml_rws.tolist(),
                             [1.5, 1.0, 2.0, 0.0])

        cuml_rws = pg.LinearPGAgent.accumulate_rewards([1, 0, 1, 0, 2], gamma=1.0)
        self.assertListEqual(cuml_rws.tolist(),
                             [4.0, 3.0, 3.0, 2.0, 2.0])

    def test_sanity(self):
        agent, initial_weights = policy_setup()
        # noinspection PyTypeChecker
        self.assertTrue(np.all(agent.policy_model.get_weights()[0] == 0))
        # noinspection PyTypeChecker
        self.assertTrue(np.all(agent.policy_model.get_weights()[1] == 0))

        agent.remember(state=[1, 0], action=1, reward=1.0)
        agent.train()
        agent.remember(state=[0, 1], action=0, reward=1.0)
        agent.train()

        action_probas = agent.get_action_probas(states=np.array([[1, 0]]))
        self.assertEqual(np.argmax(action_probas), 1)

        action_probas = agent.get_action_probas(states=np.array([[0, 1]]))
        self.assertEqual(np.argmax(action_probas), 0)

    def test_temporal_gap(self):
        agent, initial_weights = policy_setup()
        agent.remember(state=[1, 0], action=1, reward=0.0)
        agent.remember(state=[0, 1], action=0, reward=0.0)

        agent.remember(state=[0, 0], action=1, reward=0.0)
        agent.remember(state=[0, 0], action=0, reward=0.0)
        agent.remember(state=[0, 0], action=1, reward=0.0)
        agent.remember(state=[0, 0], action=0, reward=1.0)
        agent.train()

        action_probas = agent.get_action_probas(states=np.array([[1, 0]]))
        self.assertEqual(np.argmax(action_probas), 1)

        action_probas = agent.get_action_probas(states=np.array([[0, 1]]))
        self.assertEqual(np.argmax(action_probas), 0)

    def test_regmask(self):
        obs_horizon = 2
        n_actions = 4
        state_shape = (5, 5, 3)
        agent = pg.LinearPGAgent(action_space=gym.spaces.Discrete(n_actions),
                                 state_space=gym.spaces.Box(-1, 1, shape=state_shape),
                                 gamma=1.0,
                                 learning_rate=0.0,
                                 momentum=0.0,
                                 init_stdev=0.0,
                                 entropy_boost=0.0,
                                 entropy_boost_decay=0,
                                 optimizer='sgd',
                                 regmask_start_val=10.0,
                                 regmask_anneal_episodes=100)
        agent.remember(state=np.ones(state_shape),
                       action=0,
                       reward=0.0)
        train_metrics = agent.train()
        self.assertEqual(train_metrics['regmask_loss'], 0.0)

        # set "inner weights" to nonzero values
        current_weight_matrix = agent.policy_model.get_weights()[0]
        current_weight_tensor = _unflatten_weight_matrix(current_weight_matrix,
                                                         obs_horizon=obs_horizon,
                                                         n_channels=state_shape[-1],
                                                         n_actions=n_actions)
        current_weight_tensor[1:-1, 1:-1, :, :] = 1.0
        agent.policy_model.set_weights([
            current_weight_tensor.reshape(current_weight_matrix.shape),
            np.zeros(n_actions)
        ])

        # make sure that the outer weights are really outer weights
        test_state = np.ones(state_shape)
        test_state[1:-1, 1:-1, :] = 0
        test_state[:, :, :1] = 0
        action_probas = agent.get_action_probas(np.array([test_state]))[0]
        self.assertListEqual(action_probas.tolist(),
                             [0.25, 0.25, 0.25, 0.25])

        agent.remember(state=np.ones(state_shape),
                       action=0,
                       reward=0.0)
        train_metrics = agent.train()
        self.assertEqual(train_metrics['regmask_loss'], 0.0)


        # Now, set the outer weights to nonzero values and expect a loss
        agent.policy_model.set_weights([
            np.ones_like(current_weight_matrix),
            np.zeros(n_actions)
        ])
        agent.train_steps = 0
        agent.remember(state=np.ones(state_shape),
                       action=0,
                       reward=0.0)
        train_metrics = agent.train()
        self.assertEqual(train_metrics['regmask_loss'], 1920)

