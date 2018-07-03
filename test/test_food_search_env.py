import unittest

import numpy as np

import food_search_env as fse


class FoodSearchEnvTest(unittest.TestCase):
    def test_actions(self):
        env = fse.FoodSearch(obs_horizon=2, n_noise_channels=2)
        observation = env.reset()
        env.board = np.zeros_like(env.board)
        env.agent_position = np.array([8, 8])
        observation, reward, done, _ = env.step(0)
        self.assertListEqual(env.agent_position.tolist(),
                             [7, 8])
        self.assertFalse(done)

        observation, reward, done, _ = env.step(1)
        self.assertListEqual(env.agent_position.tolist(),
                             [7, 9])
        self.assertFalse(done)

        observation, reward, done, _ = env.step(2)
        self.assertListEqual(env.agent_position.tolist(),
                             [8, 9])
        self.assertFalse(done)

        observation, reward, done, _ = env.step(3)
        self.assertListEqual(env.agent_position.tolist(),
                             [8, 8])
        self.assertFalse(done)

    def test_rewards(self):
        env = fse.FoodSearch(obs_horizon=2, n_noise_channels=2)
        observation = env.reset()
        env.board = np.zeros_like(env.board)
        env.agent_position = np.array([8, 8])
        env.board[7, 8, 0] = 1
        observation, reward, done, _ = env.step(0)  # North
        self.assertTrue(done)
        self.assertEqual(reward, env.apple_rewards[0])

    def test_reset(self):
        env = fse.FoodSearch(obs_horizon=2, n_noise_channels=2,
                             rng=np.random.RandomState(1234567))
        observation = env.reset()
        self.assertEqual(observation.shape, env.observation_space.shape)
        self.assertTrue(
            all(x in {0, 1} for x in observation.flatten())
        )

        # Object encoding part of board has to have unique object at each location
        self.assertTrue(
            np.all(np.sum(env.board[:, :, :len(env.apple_rewards)],
                          axis=-1).flatten() <= 1)
        )

        self.assertTrue(
            np.any(np.sum(env.board[:, :, :],
                          axis=-1).flatten() > 1)
        )
        self.assertListEqual(
            env.board[3, 3].tolist(),
            [1, 0, 0, 0, 1]
        )

    def test_free_agent_neighborhood(self):
        env = fse.FoodSearch(obs_horizon=1, n_noise_channels=2)
        for i in range(1000):
            observation = env.reset()
            # No great apples in observation
            self.assertTrue(
                np.all(observation[:, :, 0] == 0),
                msg=f'{observation[:, :, 0]}'
            )


    def test_wall(self):
        env = fse.FoodSearch(obs_horizon=2, n_noise_channels=2)
        self.assertListEqual(
            env.board[env.wall_distance_from_boundary,
            env.wall_distance_from_boundary:-env.wall_distance_from_boundary,
            env.wall_channel].tolist(),
            [1]*(env.board_size - 2*env.wall_distance_from_boundary)
        )
        self.assertListEqual(
            env.board[-env.wall_distance_from_boundary - 1,
            env.wall_distance_from_boundary:-env.wall_distance_from_boundary,
            env.wall_channel].tolist(),
            [1]*(env.board_size - 2*env.wall_distance_from_boundary)
        )
        self.assertListEqual(
            env.board[env.wall_distance_from_boundary:-env.wall_distance_from_boundary,
            env.wall_distance_from_boundary,
            env.wall_channel].tolist(),
            [1]*(env.board_size - 2*env.wall_distance_from_boundary)
        )
        self.assertListEqual(
            env.board[env.wall_distance_from_boundary:-env.wall_distance_from_boundary,
            -env.wall_distance_from_boundary - 1,
            env.wall_channel].tolist(),
            [1]*(env.board_size - 2*env.wall_distance_from_boundary)
        )

    def test_render_board(self):
        board_channel_a = np.array(
            [[0, 1],
             [0, 0],
             [0, 1],
             [1, 0],
             [0, 0]])

        board_channel_b = np.array(
            [[0, 1],
             [1, 0],
             [0, 0],
             [0, 0],
             [1, 0]])

        board_channel_c = np.array(
            [[1, 1],
             [1, 0],
             [0, 0],
             [0, 0],
             [0, 1]])

        board = np.concatenate([board_channel_a[:, :, np.newaxis],
                                board_channel_b[:, :, np.newaxis],
                                board_channel_c[:, :, np.newaxis]],
                               axis=-1
                               )

        img = fse.render_board(board,
                               object_colors=np.array([[1, 0, 0],
                                                       [0, 1, 0],
                                                       [0, 0, 1]]))
        self.assertEqual(img.dtype, np.uint8)
        self.assertEqual(img.shape, (board.shape[0],
                                     board.shape[1],
                                     3))

        expected_img = [[[0, 0, 1], [1, 1, 1]],
                        [[0, 1, 1], [0, 0, 0]],
                        [[0, 0, 0], [1, 0, 0]],
                        [[1, 0, 0], [0, 0, 0]],
                        [[0, 1, 0], [0, 0, 1]]]
        self.assertListEqual(
            expected_img,
            img.tolist()
        )
