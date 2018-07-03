from collections import namedtuple

ExperimentParams = namedtuple('ExperimentParams',
                              ['obs_horizon',
                               'n_episodes',
                               'n_noise_channels'])

AgentParams = namedtuple('AgentParams',
                         ['gamma',
                          'learning_rate',
                          'momentum',
                          'init_stdev',
                          'entropy_boost',
                          'entropy_boost_decay',
                          'optimizer',
                          'regmask_start_val',
                          'regmask_anneal_episodes'])
