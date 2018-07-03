from typing import Tuple, Optional

import gym
import gym.spaces
import numpy as np
from keras import Input
from keras.models import Sequential
from keras.optimizers import SGD, Adam
from tqdm import tqdm

import keras
from keras.layers import Dense, Activation, Flatten, Reshape
from keras.initializers import RandomNormal

import keras.backend as K

import tensorflow as tf

rng = np.random.RandomState(1234)


def _get_regmask_loss(regmask_start_val,
                      regmask_anneal_episodes,
                      state_shape,
                      n_actions,
                      weight_matrix,
                      episode_input):
    """
    Penalizes the weights connected to the outermost ring of inputs
    """
    if regmask_start_val is None:
        assert regmask_anneal_episodes is None
        return tf.constant(0.0)

    unflattened_weights = K.reshape(weight_matrix, state_shape + (n_actions,))
    abs_weights = K.abs(unflattened_weights)
    casted_regmask_start_val = tf.cast(regmask_start_val, tf.float32)
    current_lambda = tf.maximum(tf.constant(0, dtype=tf.float32),
                                casted_regmask_start_val*(1 -
                                                          (tf.cast(episode_input, tf.float32)/
                                                           tf.cast(regmask_anneal_episodes, tf.float32))))

    regmask = tf.pad(
        tf.constant(0,
                    dtype=tf.float32,
                    shape=(state_shape[0] - 2, state_shape[1] - 2, 1, 1),
                    name='regmask_core'),
        paddings=[[1, 1], [1, 1], [0, 0], [0, 0]],
        mode='CONSTANT',
        name='regmask',
        constant_values=current_lambda
    )
    return K.sum(regmask*abs_weights)


class LinearPGAgent:
    EPSILON = 1e-6

    def __init__(self,
                 action_space: gym.spaces.Discrete,
                 state_space: gym.spaces.Box,
                 gamma: float,
                 learning_rate: float,
                 momentum: float,
                 init_stdev: float,
                 entropy_boost: float,
                 entropy_boost_decay: float,
                 optimizer: str,
                 regmask_start_val: Optional[float],
                 regmask_anneal_episodes: Optional[int]
                 ):
        self.gamma = gamma
        self.optimizer = optimizer
        self.entropy_boost = entropy_boost
        self.entropy_boost_decay = entropy_boost_decay
        self.state_shape = state_space.shape
        self.action_space = action_space
        model = self.make_model(action_space, self.state_shape, init_stdev)
        self.policy_model = model
        self.train_fn = self.make_train_fn(model, learning_rate, momentum, optimizer,
                                           regmask_start_val, regmask_anneal_episodes)
        self.states = []
        self.rewards = []
        self.actions = []
        self.train_steps = 0

    def make_model(self, action_space, input_shape, init_stdev):
        input_dim = np.product(input_shape)
        model = Sequential()
        model.add(
            Dense(units=action_space.n,
                  kernel_initializer=RandomNormal(mean=0.0,
                                                  stddev=init_stdev,
                                                  seed=None),
                  input_dim=input_dim))
        model.add(Activation('softmax'))
        return model

    def make_train_fn(self, model, learning_rate, momentum, optimizer,
                      regmask_start_val, regmask_anneal_episodes):
        assert (regmask_start_val is None) == (regmask_anneal_episodes is None)

        if optimizer == 'sgd':
            opt = SGD(lr=learning_rate, momentum=momentum)
        elif optimizer == 'adam':
            opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
        else:
            raise ValueError('Unrecognized optimizer: {}'.format(optimizer))
        state_input = model.input
        action_input = Input(batch_shape=(None,),
                             dtype=np.int32,
                             name='action_input')
        reward_input = Input(batch_shape=(None,),
                             dtype=np.float32,
                             name='reward_input')
        episode = tf.placeholder(dtype=tf.int32, shape=(), name='episode')

        action_probas = model(state_input)
        indexer = tf.stack([tf.range(0, tf.shape(action_input)[0]),
                            action_input],
                           axis=1)
        selected_action_probas = tf.gather_nd(params=action_probas,
                                              indices=indexer)

        mean_entropy = K.mean(-K.sum(action_probas*K.log(action_probas + 1e-6), axis=1))
        objectives = reward_input*K.log(selected_action_probas)
        entropy_loss_decay_coefficient = K.exp(-self.entropy_boost_decay*tf.cast(episode, dtype=tf.float32))
        undiscounted_entropy_loss = -K.log(mean_entropy + self.EPSILON)
        entropy_loss = entropy_loss_decay_coefficient*self.entropy_boost*undiscounted_entropy_loss

        regmask_loss = _get_regmask_loss(regmask_start_val,
                                         regmask_anneal_episodes,
                                         self.state_shape,
                                         self.action_space.n,
                                         model.layers[0].kernel,
                                         episode)

        loss = (-K.sum(objectives) +
                entropy_loss +
                regmask_loss)
        updates = opt.get_updates(
            loss=loss,
            params=model.trainable_weights)

        # --- Statistics for monitoring purposes only
        grads_overall = opt.get_gradients(loss=loss, params=model.trainable_weights)
        grads_entropy = opt.get_gradients(loss=entropy_loss, params=model.trainable_weights)

        # We assume a linear model with weight-matrix and bias-vector
        assert len(grads_overall) == len(grads_entropy) == 2
        W_grad_length_overall = K.sqrt(K.sum(K.square(grads_overall[0])))
        W_grad_length_entropy = K.sqrt(K.sum(K.square(grads_entropy[0])))

        # weight_variance = K.var(model.trainable_weights[0])
        # -------------------------------

        train_fn = K.function(inputs=[state_input, action_input, reward_input, episode],
                              outputs=[entropy_loss,
                                       undiscounted_entropy_loss,
                                       W_grad_length_overall,
                                       W_grad_length_entropy,
                                       regmask_loss
                                       ],
                              updates=updates)
        return train_fn

    def act(self, state: np.ndarray):
        assert state.shape == self.state_shape
        action_probas = self.policy_model.predict([state.reshape(1, -1)])[0]
        return rng.choice(np.arange(self.action_space.n),
                          p=action_probas)

    def get_action_probas(self, states: np.ndarray):
        assert states.shape[1:] == self.state_shape, (f'states.shape: {states.shape}, '
                                                      f'self.state_shape: {self.state_shape}')
        return self.policy_model.predict_on_batch(states.reshape(states.shape[0], -1))
        # return self.policy_model.predict_on_batch([states])

    def get_policy(self):
        def policy(state):
            return self.get_action_probas(np.array([state]))[0]

        return policy

    def remember(self, state, action, reward):
        self.states.append(np.asarray(state).reshape((-1,)))
        self.actions.append(action)
        self.rewards.append(reward)

    def train(self):
        cuml_rewards = self.accumulate_rewards(self.rewards, self.gamma)
        train_metrics = self.train_fn([self.states,
                                       self.actions,
                                       cuml_rewards,
                                       self.train_steps])
        self.train_steps += 1
        self.states = []
        self.rewards = []
        self.actions = []
        metric_keys = ['entropy_loss',
                       'undiscounted_entropy_loss',
                       'W_grad_length_overall',
                       'W_grad_length_entropy',
                       'regmask_loss']
        if len(metric_keys) != len(train_metrics):
            raise ValueError('Expecting same lengths of metric '
                             'keys and train_metrics!')
        return dict(zip(
            metric_keys,
            train_metrics
        ))

    @staticmethod
    def accumulate_rewards(rewards, gamma):
        """
        Takes in single rewards, returns accumulated rewards for each step onwards
        """
        n_steps = len(rewards)
        mask = np.triu(gamma**(np.arange(n_steps)[None, :] - np.arange(n_steps)[:, None]))
        cuml_rewards = np.sum(mask*rewards, axis=1)
        return cuml_rewards
