import unittest

import numpy as np

import food_search_env as fse
import handcrafted_policies as hcp


def objects_action_test(test: unittest.TestCase,
                        policy,
                        objects,
                        expected_action):
    objects = 3 - np.array(objects)
    observation = fse.FoodSearch.board_from_objects(objects,
                                                    n_informative_channels=3,
                                                    n_noise_channels=2,
                                                    rng=np.random.RandomState(1234))

    action_probas = policy(observation)
    msg = (f'objects:\n{repr(3 - objects)}\n'
           f'expected_action: {expected_action}\n'
           f'taken_action: {np.argmax(action_probas)}\n'
           f'action_probas: {action_probas}')
    test.assertTrue(np.allclose(
        action_probas, np.eye(hcp.N_ACTIONS)[expected_action]),
        msg
    )


def test_symmetric(test: unittest.TestCase,
                   policy,
                   objects,
                   expected_action):
    for i in range(4):
        objects = np.rot90(objects, k=-1)
        expected_action = (expected_action + 1)%4
        objects_action_test(test, policy, objects, expected_action)


class FoodSearchEnvTest(unittest.TestCase):
    def test_policy_obsho_1(self):
        obs_horizon = 1
        policy = hcp.make_deterministic_policy(weight_tensor=hcp.hc_weight_tensor_by_obsho[obs_horizon],
                                               bias_tensor=hcp.hc_bias_tensor_by_obsho[obs_horizon])

        observation = np.zeros((3, 3, hcp.N_CHANNELS))
        self.assertTrue(np.allclose(
            policy(observation), np.eye(hcp.N_ACTIONS)[hcp.NORTH]))

        observation = np.zeros((3, 3, hcp.N_CHANNELS))
        observation[1, 2, 0] = 1
        self.assertTrue(np.allclose(
            policy(observation), np.eye(hcp.N_ACTIONS)[hcp.EAST]))

        observation = np.zeros((3, 3, hcp.N_CHANNELS))
        observation[2, 1, 1] = 1
        observation[1, 0, 1] = 1
        self.assertTrue(np.allclose(
            policy(observation), np.eye(hcp.N_ACTIONS)[hcp.SOUTH]))

        observation = np.zeros((3, 3, hcp.N_CHANNELS))
        observation[2, 2, 0] = 1
        self.assertTrue(np.allclose(
            policy(observation), np.eye(hcp.N_ACTIONS)[hcp.EAST]))

        observation = np.zeros((3, 3, hcp.N_CHANNELS))
        observation[2, 2, 0] = 1
        observation[1, 2, 2] = 1
        self.assertTrue(np.allclose(
            policy(observation), np.eye(hcp.N_ACTIONS)[hcp.SOUTH]))

        observation = np.zeros((3, 3, hcp.N_CHANNELS))
        observation[0, 0, 1] = 1
        self.assertTrue(np.allclose(
            policy(observation), np.eye(hcp.N_ACTIONS)[hcp.WEST]))

        objects = [
            [0, 1, 0],
            [0, 0, 0],
            [0, 0, 0]
        ]
        expected_action = hcp.EAST
        test_symmetric(self, policy, objects, expected_action)

        objects = [
            [1, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ]
        expected_action = hcp.EAST
        objects_action_test(self, policy, objects, expected_action)

        objects = [
            [0, 0, 1],
            [0, 0, 0],
            [0, 0, 0]
        ]
        expected_action = hcp.EAST
        objects_action_test(self, policy, objects, expected_action)

        objects = [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 1]
        ]
        expected_action = hcp.WEST
        objects_action_test(self, policy, objects, expected_action)

        objects = [
            [0, 0, 0],
            [0, 0, 0],
            [1, 0, 0]
        ]
        expected_action = hcp.WEST
        objects_action_test(self, policy, objects, expected_action)


    def test_policy_obsho_2(self):
        obs_horizon = 2
        policy = hcp.make_deterministic_policy(weight_tensor=hcp.hc_weight_tensor_by_obsho[obs_horizon],
                                               bias_tensor=hcp.hc_bias_tensor_by_obsho[obs_horizon])

        # Will be "inverted", so that we can use better readable format
        # 0 = "nothing"
        # 1 = "BAD"
        # 2 = "GOOD"
        # 3 = "GREAT"
        objects = [
            [0, 0, 3, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2],
        ]
        expected_action = hcp.NORTH
        test_symmetric(self, policy, objects, expected_action)

        objects = [
            [0, 0, 3, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
            [2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2],
        ]
        expected_action = hcp.SOUTH
        test_symmetric(self, policy, objects, expected_action)

        objects = [
            [0, 0, 3, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
        expected_action = hcp.EAST
        test_symmetric(self, policy, objects, expected_action)

        objects = [
            [0, 3, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
        expected_action = hcp.NORTH
        test_symmetric(self, policy, objects, expected_action)

        objects = [
            [0, 3, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
        expected_action = hcp.WEST
        test_symmetric(self, policy, objects, expected_action)

        objects = [
            [0, 2, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
        expected_action = hcp.WEST
        test_symmetric(self, policy, objects, expected_action)

        objects = [
            [0, 2, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
        expected_action = hcp.NORTH
        test_symmetric(self, policy, objects, expected_action)

        objects = [
            [0, 2, 0, 0, 0],
            [0, 0, 2, 0, 0],
            [0, 1, 0, 2, 0],
            [0, 0, 0, 0, 0],
            [3, 0, 0, 0, 0],
        ]
        expected_action = hcp.SOUTH
        test_symmetric(self, policy, objects, expected_action)
