from datetime import datetime

import gym
import gym.spaces
import numpy as np
import matplotlib.pyplot as plt
import moviepy.editor as mpy

import handcrafted_policies
import policy_analysis

OBJECT_COLORS = [
                    [120, 255, 120],  # GREAT
                    [0, 200, 255],  # GOOD
                    [255, 120, 30],  # BAD
                    [255, 255, 255],
                    [255, 255, 255],
                ] + [[255, 255, 255]]*100
OBJECT_COLORS = np.asarray(OBJECT_COLORS)


AGENT_COLOR = [100, 100, 100]


class FoodSearch(gym.Env):
    action_direction_map = {
        0: np.array([-1, 0]),  # NORTH
        1: np.array([0, 1]),  # EAST
        2: np.array([1, 0]),  # SOUTH
        3: np.array([0, -1]),  # WEST
    }

    apple_rewards = [+3, +1, -1]
    step_penalty = -0.05
    action_names = ['↑', '→', '↓', '←']

    episode_max_steps = 16

    def __init__(self, obs_horizon, n_noise_channels, rng=None):
        if rng is None:
            self.rng = np.random.RandomState()
        else:
            self.rng = rng

        self.n_noise_channels = n_noise_channels
        self.channel_names = ([f'{"" if r < 0 else "+"}{r}'
                               for r in self.apple_rewards] +
                              [f'N{i}' for i in range(n_noise_channels)])

        self.obs_horizon = obs_horizon

        # Length of these lists control number of objects/channels (excluding noise channels)
        self.apple_probabilities = [0.03, 0.03, 0.14]  # Rest will be "no apple"
        assert len(self.apple_probabilities) == len(self.apple_rewards)
        assert np.sum(self.apple_probabilities) < 1.0
        self.object_probabilities = np.append(self.apple_probabilities,
                                              [1 - np.sum(self.apple_probabilities)])
        # noinspection PyTypeChecker
        assert np.isclose(np.sum(self.object_probabilities), 1.0)

        self.wall_channel = np.argmin(self.apple_rewards)

        self.n_channels = self.n_noise_channels + len(self.apple_rewards)

        self.board_size = 20

        # Should be at least the maximum observation horizon
        self.wall_distance_from_boundary = 2

        self.board_shape = (self.board_size, self.board_size, self.n_channels)

        # Probability of a pixel in a channel being 1
        self.object_density = 0.2

        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(-1, 1, shape=(2*self.obs_horizon + 1,
                                                              2*self.obs_horizon + 1,
                                                              self.n_channels))
        self.board = None
        self._reset()

    def _reset(self):
        self.episode_step = 0
        self.agent_position = np.array([self.board_size//2, self.board_size//2])
        self.board = self._sample_board()
        return self._get_observation()

    def _step(self, action):
        self.episode_step += 1
        assert 0 <= action < 4
        done = False
        reward = self.step_penalty
        direction = self.action_direction_map[action]
        self.agent_position += direction

        if any(self.agent_position < 0) or any(self.agent_position > self.board_size - 1):
            done = True
            reward = -1.0
            self.agent_position = np.clip(self.agent_position, 0, self.board_size - 1)
        else:
            active_channels, = np.where(
                self.board[self.agent_position[0], self.agent_position[1], :len(self.apple_rewards)] > 0)
            assert len(active_channels) <= 1
            if len(active_channels) == 1 and active_channels[0] < len(self.apple_rewards):
                # Apple was collected
                reward = self.apple_rewards[active_channels[0]]
                done = True

        if self.episode_step >= self.episode_max_steps:
            done = True
        return self._get_observation(), reward, done, None

    def _sample_board(self):
        # print(
        objects = self.rng.choice(np.arange(len(self.apple_rewards) + 1),
                                  self.board_shape[:2],
                                  p=self.object_probabilities)
        # Remove apple from agent position
        no_apple_idx = len(self.apple_rewards)
        objects[self.agent_position[0], self.agent_position[1]] = no_apple_idx

        # Free neighborhood from valuable apples
        # @formatter:off
        agent_neighborhood = objects[self.agent_position[0] - 1:self.agent_position[0] + 2:,
                             self.agent_position[1] - 1:self.agent_position[0] + 2]

        # @formatter:on
        agent_neighborhood[:, :] = (no_apple_idx*(agent_neighborhood == 0) +
                                    agent_neighborhood*(agent_neighborhood != 0))

        board = self.board_from_objects(objects, len(self.apple_rewards),
                                        self.n_noise_channels, self.rng)
        self.make_boundary(board,
                           self.wall_distance_from_boundary,
                           self.wall_channel,
                           self.n_channels)
        return board

    @staticmethod
    def board_from_objects(objects, n_informative_channels, n_noise_channels, rng):
        n_channels = n_informative_channels + n_noise_channels

        if np.max(objects) >= n_informative_channels + 1:
            print(sorted(set(objects.flatten())))
            print(n_informative_channels)

        assert np.max(objects) < n_informative_channels + 1
        board = np.zeros(objects.shape + (n_channels,), dtype=np.int8)
        for channel in range(n_informative_channels):
            xs, ys = np.where(objects == channel)
            board[xs, ys, channel] = +1
        board[:, :, n_informative_channels:] = rng.choice([0, 1],
                                                          (board.shape[0],
                                                           board.shape[1],
                                                           n_noise_channels))
        return board

    @staticmethod
    def make_boundary(board, distance_from_boundary, wall_channel, n_channels):
        """
        mutates board
        """
        wall = np.eye(n_channels)[wall_channel]
        board[distance_from_boundary, distance_from_boundary:-distance_from_boundary, :] = wall
        board[-distance_from_boundary - 1, distance_from_boundary:-distance_from_boundary, :] = wall
        board[distance_from_boundary:-distance_from_boundary, distance_from_boundary, :] = wall
        board[distance_from_boundary:-distance_from_boundary, -distance_from_boundary - 1, :] = wall

    def _get_observation(self):
        board_with_margin = np.pad(self.board,
                                   pad_width=((self.obs_horizon, self.obs_horizon),
                                              (self.obs_horizon, self.obs_horizon),
                                              (0, 0)),
                                   mode='constant',
                                   constant_values=0)
        # np pad makes a copy already, but let's copy again, just to be sure
        return board_with_margin[
               self.agent_position[0]:self.agent_position[0] + 2*self.obs_horizon + 1,
               self.agent_position[1]:self.agent_position[1] + 2*self.obs_horizon + 1,
               :].copy()


def render_board_with_agent(board, agent_position, obs_horizon):
    assert obs_horizon > 0

    img = render_board(board)

    greyed_board = (img.astype(np.int64) + 2*220)//3
    # greyed_board = img
    #
    from_y = max(0, agent_position[0] - obs_horizon)
    to_y = min(img.shape[0], agent_position[0] + obs_horizon + 1)
    from_x = max(0, agent_position[1] - obs_horizon)
    to_x = min(img.shape[1], agent_position[1] + obs_horizon + 1)
    greyed_board[from_y:to_y, from_x:to_x] = img[from_y:to_y, from_x:to_x]

    greyed_board[agent_position[0], agent_position[1]] = AGENT_COLOR

    return greyed_board.astype(np.uint8)


def render_board(board, object_colors=OBJECT_COLORS):
    assert all(x in {0, 1} for x in board.flatten())
    assert board.ndim == 3
    assert board.shape[2] >= 3  # 3 fruits to display
    assert np.shape(object_colors)[0] >= board.shape[2], f'{np.shape(object_colors)[0]} vs {board.shape[2]}'
    # board[:, :, np.newaxis, :] * object_colors.T
    # img = np.minimum(255, np.dot(board, object_colors[:board.shape[2]]))
    img = np.clip(255 - np.dot(board, 255-object_colors[:board.shape[2]]),
                  0, 255)
    return img.astype(np.uint8)


def _scale_up(img, scale):
    return np.repeat(
        np.repeat(img, scale, axis=0),
        scale,
        axis=1)


def _hstack_imgs(imgs, gap):
    assert isinstance(imgs, list)
    assert len(imgs) > 1
    max_height = max(img.shape[0] for img in imgs)
    total_width = sum(img.shape[1] for img in imgs) + gap*(len(imgs) - 1)
    img = np.concatenate(
        [np.pad(img, pad_width=((0, max_height - img.shape[0]), (0, gap), (0, 0)), mode='constant', constant_values=255)
         for img in imgs], axis=1)
    if gap > 0:
        return img[:, :-gap]
    else:
        return img


def _render_action_probas(probas, height, width, pad, bar_width=0.9,
                          color=(50, 140, 200), bg_color=(255, 255, 255)):
    assert np.isclose(np.sum(probas), 1.0)
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:, :] = bg_color
    for i in range(len(probas)):
        top = int(round((1 - probas[i])*height))
        left = i*(width//len(probas))
        right = int((i + bar_width)*(width//len(probas)))
        img[top:, left:right, :] = color
        img[:top, left:right, :] = (np.array(color) + 2*255)//3
    return np.pad(img, ((pad, pad), (pad, pad), (0, 0)),
                  mode='constant',
                  constant_values=255)


def _render_network_weights(weight_tensor, bias_tensor, channel_names, action_names):
    """
    :param weight_tensor: (y, x, channel, action)
    """
    height, width, n_channels, n_actions = weight_tensor.shape
    channel_names.append('bias')
    weight_min = -10
    weight_max = +10
    fig, axes = plt.subplots(nrows=n_actions, ncols=n_channels + 1)
    for action, axs_row in enumerate(axes):
        for channel, ax in enumerate(axs_row):
            if channel < n_channels:
                ax.imshow(weight_tensor[:, :, channel, action],
                          vmin=weight_min, vmax=weight_max)
            else:
                ax.imshow(bias_tensor[action].reshape(1, 1),
                          vmin=weight_min, vmax=weight_max)
            ax.set_xticks([])
            ax.set_yticks([])

    for ax, channel_name in zip(axes[0], channel_names):
        ax.set_title(channel_name)

    for ax, action_name in zip(axes[:, 0], action_names):
        ax.set_ylabel(action_name, rotation=0, size=20, labelpad=20,
                      fontdict={'va': 'center'})

    fig.tight_layout()
    return _mpl_figure_to_rgb_img(fig, 400, 500)


def _mpl_figure_to_rgb_img(fig: plt.Figure, height, width):
    fig.set_dpi(100)
    fig.set_size_inches(width/100, height/100)

    canvas = fig.canvas
    canvas.draw()
    width, height = np.round(fig.get_size_inches()*fig.get_dpi()).astype(int)
    # image = np.fromstring(fig.canvas.tostring_rgb(), dtype='uint8')

    img = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(height, width, 3)
    plt.close(fig)
    return img


def render_and_save_video(env, policy, name_prefix='test_video', n_episodes=5, wait_after_episode=2,
                          weight_tensor=None, bias_tensor=None):
    clip = render_video(env, policy, n_episodes, weight_tensor, bias_tensor, wait_after_episode)

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    clip.write_videofile(f'test_videos/{name_prefix}_{timestamp}.mp4')


def render_video(env, policy, n_episodes=5, weight_tensor=None, bias_tensor=None, wait_after_episode=2):
    if weight_tensor is not None:
        assert bias_tensor is not None
        weights_biases_img = _render_network_weights(weight_tensor, bias_tensor, env.channel_names,
                                                     env.action_names)
    else:
        assert bias_tensor is None
        weights_biases_img = None

    def _make_frame(env, action_probas):
        full_img = render_board_with_agent(env.board, env.agent_position, env.obs_horizon)
        full_img = _scale_up(full_img, 20)

        observation_img = render_board(env._get_observation())
        observation_img = _scale_up(observation_img, 20)

        action_probas_img = _render_action_probas(action_probas, 80, 40, pad=2)

        stacked_imgs = [full_img, action_probas_img, observation_img]
        if weights_biases_img is not None:
            stacked_imgs.append(weights_biases_img)

        img = _hstack_imgs(stacked_imgs, gap=8)
        return img

    board_frames = []
    for i_episode in range(n_episodes):
        observation = env.reset()
        while True:
            action_probas = policy(observation)
            assert np.shape(action_probas) == (4,)
            action = np.random.choice(np.arange(4), p=action_probas)

            board_frames.append(_make_frame(env, action_probas))
            observation, reward, done, _ = env.step(action)
            if done:
                board_render = _make_frame(env, action_probas)
                board_frames.extend([board_render]*wait_after_episode)
                break
    clip = mpy.ImageSequenceClip(board_frames, fps=4)
    return clip


def _unflatten_weight_matrix(weights, obs_horizon, n_channels, n_actions):
    return weights.reshape(2*obs_horizon + 1, 2*obs_horizon + 1, n_channels, n_actions)


if __name__ == '__main__':

    for obs_horizon in [1, 2]:
        weight_tensor = handcrafted_policies.hc_weight_tensor_by_obsho[obs_horizon]
        bias_tensor = handcrafted_policies.hc_bias_tensor_by_obsho[obs_horizon]

        policy = handcrafted_policies.make_deterministic_policy(weight_tensor, bias_tensor)

        env = FoodSearch(obs_horizon, n_noise_channels=2)
        returns = policy_analysis.determine_policy_returns(env, policy, 100)
        print(f'obs_horizon: {obs_horizon}, '
              f'mean return: {returns.mean():.4}, '
              f'stddev: {returns.std() / np.sqrt(len(returns)):.4}')

        render_and_save_video(FoodSearch(obs_horizon, n_noise_channels=2),
                              policy,
                              name_prefix=f'obsho_{obs_horizon}',
                              weight_tensor=None, #weight_tensor,
                              bias_tensor=None) #bias_tensor)
