# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 10:27:40 2020

@author: Mieszko Ferens
"""

from pytest import mark
from sympy import arg
import chainer
import chainerrl
import chainer.links as L
import chainer.functions as F
import numpy as np
from chainerrl.links.mlp import MLP

# from chainerrl.misc.prioritized import MyPrioritizedBuffer as PrioritizedBuffer
# from chainerrl.replay_buffers.prioritized import PrioritizedReplayBuffer, PrioritizedBuffer
import chainer.backends.cuda
# import cupy as cp
# if chainer.cuda.available:
#     print("GPU is available")
#     device_id = 1  
#     chainer.cuda.get_device_from_id(device_id).use()



def exp_return_of_episode(episode):
    return np.exp(sum(x['reward'] for x in episode))
class A3CFCSoftmax(chainer.ChainList, chainerrl.agents.a3c.A3CModel, chainerrl.recurrent.RecurrentChainMixin):
    """FR: Just a simple modification of the LSTM layer that runs with discrete action spaces"""

    def __init__(self, obs_size, action_size, hidden_size=90, lstm_size=40,hidden_sizes=(90,60)):
        # self.pi_head = L.Linear(obs_size, obs_size)
        # self.v_head = L.Linear(obs_size, obs_size)
        # self.pi_lstm = L.LSTM(hidden_size, lstm_size)
        # self.v_lstm = L.LSTM(hidden_size, lstm_size)
        # self.pi = chainerrl.policies.MellowmaxPolicy(
        #     model=chainerrl.links.MLP(obs_size, action_size, hidden_sizes))
        # self.v = chainerrl.links.MLP(obs_size, 1, hidden_sizes=hidden_sizes)
        self.pi = chainerrl.policies.FCSoftmaxPolicy(obs_size, action_size,n_hidden_channels=90,
        n_hidden_layers=2,last_wscale=0.01,nonlinearity=F.tanh)
        self.v = chainerrl.v_function.FCVFunction(obs_size,n_hidden_channels=60,
        n_hidden_layers=1,last_wscale=0.01,nonlinearity=F.tanh)
        super().__init__(self.pi, self.v)

    def pi_and_v(self, state):
        # def forward(head,  tail):
        #     h = F.tanh(head(state))
        #     return tail(h)

        # pout = forward(self.pi_head, self.pi)
        # vout = forward(self.v_head, self.v)
        return self.pi(state), self.v(state)

        # return pout, vout

class A3CFFSoftmax(chainer.ChainList, chainerrl.agents.a3c.A3CModel):
    """An example of A3C feedforward softmax policy."""

    def __init__(self, ndim_obs, n_actions, hidden_sizes=(90, 60)):
        self.pi = chainerrl.policies.SoftmaxPolicy(
            model=chainerrl.links.MLP(ndim_obs, n_actions, hidden_sizes))
        self.v = chainerrl.links.MLP(ndim_obs, 1, hidden_sizes=hidden_sizes)
        super().__init__(self.pi, self.v)

    def pi_and_v(self, state):
        return self.pi(state), self.v(state)


class DuelingDQN(chainer.Chain, chainerrl.q_function.StateQFunction):
    """Dueling Q-Network

    See: http://arxiv.org/abs/1511.06581
    """

    def __init__(self, n_actions, n_input_channels=4,
                 activation=F.relu, bias=0.1):
        self.n_actions = n_actions
        self.n_input_channels = n_input_channels
        self.activation = activation

        super().__init__()
        with self.init_scope():
            self.conv_layers = chainer.ChainList(
                L.Convolution2D(n_input_channels, 32, 8, stride=4,
                                initial_bias=bias),
                L.Convolution2D(32, 64, 4, stride=2, initial_bias=bias),
                L.Convolution2D(64, 64, 3, stride=1, initial_bias=bias))

            self.a_stream = MLP(3136, n_actions, [512])
            self.v_stream = MLP(3136, 1, [512])

    def __call__(self, x):
        h = x
        for link in self.conv_layers:
            h = self.activation(link(h))

        # Advantage
        batch_size = x.shape[0]
        ya = self.a_stream(h)
        mean = F.reshape(
            F.sum(ya, axis=1) / self.n_actions, (batch_size, 1))
        ya, mean = F.broadcast(ya, mean)
        ya -= mean

        # State value
        ys = self.v_stream(h)

        ya, ys = F.broadcast(ya, ys)
        q = ya + ys
        return chainerrl.action_value.DiscreteActionValue(q)

class DistributionalDuelingDQN(
        chainer.Chain, chainerrl.q_function.StateQFunction, chainerrl.recurrent.RecurrentChainMixin):
    """Distributional dueling fully-connected Q-function with discrete actions.

    """

    def __init__(self, n_actions, n_atoms, v_min, v_max,
                 n_input_channels=4, activation=F.relu, bias=0.1):
        assert n_atoms >= 2
        assert v_min < v_max

        self.n_actions = n_actions
        self.n_input_channels = n_input_channels
        self.activation = activation
        self.n_atoms = n_atoms

        super().__init__()
        z_values = self.xp.linspace(v_min, v_max,
                                    num=n_atoms,
                                    dtype=np.float32)
        self.add_persistent('z_values', z_values)

        with self.init_scope():
            self.conv_layers = chainer.ChainList(
                L.Convolution2D(n_input_channels, 32, 8, stride=4,
                                initial_bias=bias),
                L.Convolution2D(32, 64, 4, stride=2, initial_bias=bias),
                L.Convolution2D(64, 64, 3, stride=1, initial_bias=bias))

            self.main_stream = L.Linear(3136, 1024)
            self.a_stream = L.Linear(512, n_actions * n_atoms)
            self.v_stream = L.Linear(512, n_atoms)
    def __call__(self, x):
        h = x
        for link in self.conv_layers:
            h = self.activation(link(h))

        # Advantage
        batch_size = x.shape[0]

        h = self.activation(self.main_stream(h))
        h_a, h_v = F.split_axis(h, 2, axis=-1)
        ya = F.reshape(self.a_stream(h_a),
                       (batch_size, self.n_actions, self.n_atoms))

        mean = F.sum(ya, axis=1, keepdims=True) / self.n_actions

        ya, mean = F.broadcast(ya, mean)
        ya -= mean

        # State value
        ys = F.reshape(self.v_stream(h_v), (batch_size, 1, self.n_atoms))
        ya, ys = F.broadcast(ya, ys)
        q = F.softmax(ya + ys, axis=2)

        return chainerrl.action_value.DistributionalDiscreteActionValue(q, self.z_values)

class A2CFFSoftmax(chainer.ChainList, chainerrl.agents.a2c.A2CModel):
    """An example of A2C feedforward softmax policy."""

    def __init__(self, ndim_obs, n_actions, hidden_sizes=(64, 64)):
        self.pi = chainerrl.policies.SoftmaxPolicy(
            model=chainerrl.links.MLP(ndim_obs, n_actions, hidden_sizes))
        self.v = chainerrl.links.MLP(ndim_obs, 1, hidden_sizes=hidden_sizes)
        super().__init__(self.pi, self.v)

    def pi_and_v(self, state):
        return self.pi(state), self.v(state)

class A2CGaussian(chainer.ChainList, chainerrl.agents.a2c.A2CModel):
    """An example of A2C recurrent Gaussian policy."""

    def __init__(self, obs_size, action_size):
        self.pi = chainerrl.policies.FCGaussianPolicyWithFixedCovariance(
            obs_size,
            action_size,
            np.log(np.e - 1),
            n_hidden_layers=2,
            n_hidden_channels=64,
            nonlinearity=F.tanh)
        self.v = chainerrl.v_function.FCVFunction(obs_size, n_hidden_layers=2,
                                        n_hidden_channels=64,
                                        nonlinearity=F.tanh)
        super().__init__(self.pi, self.v)

    def pi_and_v(self, state):
        return self.pi(state), self.v(state)

# Q-function definition
class QFunction(chainer.Chain):
    
    def __init__(self, obs_size, n_actions, n_hidden_channels=90):
        super().__init__()
        with self.init_scope():
            self.l0 = L.Linear(obs_size, n_hidden_channels)
            self.l1 = L.Linear(n_hidden_channels, n_hidden_channels)
            self.l2 = L.Linear(n_hidden_channels, n_actions)
    
    def __call__(self, obs, test=False):
        h0 = F.tanh(self.l0(obs))
        h1 = F.tanh(self.l1(h0))
        return chainerrl.action_value.DiscreteActionValue(self.l2(h1))

# Funtion that instances an agent with certain parameters
def create_agent(
        gamma, obs_size, n_actions, exploration_func, alg, exp_type='constant',
        epsilon=0.1):
    # Error handling
    if(type(gamma) != float and type(gamma) != int):
        raise KeyboardInterrupt(
            'Error while creating agent: Gamma type invalid')
    if(type(exp_type) != str):
        raise KeyboardInterrupt(
            'Error while creating agent: Exploration type invalid')
    if(type(epsilon) != float and type(epsilon) != int
       and type(epsilon) != list):
        raise KeyboardInterrupt(
            'Error while creating agent: Epsilon type invalid')
    # print(alg)
    if(alg in 'D_D_Q_N'):
        # Q function instanciation
        q_func = QFunction(obs_size, n_actions)
        
        # Optimizer
        opt = chainer.optimizers.Adam(eps=1e-2)
        opt.setup(q_func)
        
        # q_func.to_gpu(1)
        
        # Exploration & agent info
        if(exp_type in 'constant'):
            agent_info = ('DDQN - pure ' + 'Constant ' + chr(949) + '=' +
                          str(epsilon) + ' (' + chr(947) + '=' + str(gamma) +
                          ')')
            explorer = chainerrl.explorers.ConstantEpsilonGreedy(
                    epsilon=epsilon, random_action_func=exploration_func)
        elif(exp_type in 'linear decay'):
            if(type(epsilon) != list):
                raise KeyboardInterrupt(
                    'Error while creating agent: Linear decay exploration does'
                    'not work with a constant epsilon')
            agent_info = ('DDQN - ' + 'Linear decay ' + chr(949) + '=' +
                          str(epsilon[0]) + '->' + str(epsilon[1]) + ' in ' +
                          str(epsilon[2]) + ' time steps' + ' (' + chr(947) +
                          '=' + str(gamma) + ')')
            explorer = chainerrl.explorers.LinearDecayEpsilonGreedy(
            start_epsilon=epsilon[0], end_epsilon=epsilon[1],
            decay_steps=epsilon[2], random_action_func=exploration_func)
        else: # If type doesn't match with any known one raise error
            raise KeyboardInterrupt(
                'Error while creating agent: Unknown exploration type')
        
        # Experience replay
        replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=1000000)
        
        # Agent (DDQN)
        agent = chainerrl.agents.DoubleDQN(
                q_func, opt, replay_buffer, gamma, explorer,
                replay_start_size=80000, target_update_interval=30000)
        
        return agent, agent_info, q_func, opt, explorer, replay_buffer
    elif(alg in 'DQN'):
        # Q function instanciation
        q_func = QFunction(obs_size, n_actions)
        # Optimizer
        opt = chainer.optimizers.Adam(eps=1e-2)
        opt.setup(q_func)
        # Exploration & agent info
        if(exp_type in 'constant'):
            agent_info = ('DQN - ' + 'Constant ' + chr(949) + '=' +
                          str(epsilon) + ' (' + chr(947) + '=' + str(gamma) +
                          ')')
            explorer = chainerrl.explorers.ConstantEpsilonGreedy(
                    epsilon=epsilon, random_action_func=exploration_func)
        elif(exp_type in 'linear decay'):
            if(type(epsilon) != list):
                raise KeyboardInterrupt(
                    'Error while creating agent: Linear decay exploration does'
                    'not work with a constant epsilon')
            agent_info = ('DQN - ' + 'Linear decay ' + chr(949) + '=' +
                          str(epsilon[0]) + '->' + str(epsilon[1]) + ' in ' +
                          str(epsilon[2]) + ' time steps' + ' (' + chr(947) +
                          '=' + str(gamma) + ')')
            explorer = chainerrl.explorers.LinearDecayEpsilonGreedy(
            start_epsilon=epsilon[0], end_epsilon=epsilon[1],
            decay_steps=epsilon[2], random_action_func=exploration_func)
        else: # If type doesn't match with any known one raise error
            raise KeyboardInterrupt(
                'Error while creating agent: Unknown exploration type')
        
        # Experience replay
        replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=1000000)
        
        # Agent (DQN)
        agent = chainerrl.agents.DQN(q_function=q_func, optimizer=opt, replay_buffer=replay_buffer, gamma=gamma, explorer=explorer, 
                gpu=None, replay_start_size=80000, 
                update_interval=1, target_update_interval=30000)
        
        return agent, agent_info, q_func, opt, explorer, replay_buffer
    elif(alg in 'PS-DDQN'):
        # Q function instanciation
        q_func = QFunction(obs_size, n_actions)
        
        # Optimizer
        opt = chainer.optimizers.Adam(eps=1e-2)
        opt.setup(q_func)
        
        # Exploration & agent info
        if(exp_type in 'constant'):
            agent_info = ('DDQN with PER ' + 'Constant ' + chr(949) + '=' +
                          str(epsilon) + ' (' + chr(947) + '=' + str(gamma) +
                          ')')
            explorer = chainerrl.explorers.ConstantEpsilonGreedy(
                    epsilon=epsilon, random_action_func=exploration_func)
        elif(exp_type in 'linear decay'):
            if(type(epsilon) != list):
                raise KeyboardInterrupt(
                    'Error while creating agent: Linear decay exploration does'
                    'not work with a constant epsilon')
            agent_info = ('DDQN - ' + 'Linear decay ' + chr(949) + '=' +
                          str(epsilon[0]) + '->' + str(epsilon[1]) + ' in ' +
                          str(epsilon[2]) + ' time steps' + ' (' + chr(947) +
                          '=' + str(gamma) + ')')
            explorer = chainerrl.explorers.LinearDecayEpsilonGreedy(
            start_epsilon=epsilon[0], end_epsilon=epsilon[1],
            decay_steps=epsilon[2], random_action_func=exploration_func)
        else: # If type doesn't match with any known one raise error
            raise KeyboardInterrupt(
                'Error while creating agent: Unknown exploration type')
        
        # Experience replay
        # replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=1000000)
        betasteps = 5 * 10 ** 7 
        # Experience replay
        # replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=1000000)
        replay_buffer = chainerrl.replay_buffer.PrioritizedReplayBuffer(capacity=1000000,
        alpha=0.5, beta0=0.4, betasteps=betasteps,
        num_steps=3,normalize_by_max='memory')
        # Agent (DDQN)
        agent = chainerrl.agents.DoubleDQN(
                q_func, opt, replay_buffer, gamma, explorer,
                replay_start_size=100000, target_update_interval=50000)
        
        return agent, agent_info, q_func, opt, explorer, replay_buffer
    elif(alg in 'PE-DDQN'):
        # Q function instanciation
        q_func = QFunction(obs_size, n_actions)
        
        # Optimizer
        opt = chainer.optimizers.Adam(eps=1e-2)
        opt.setup(q_func)
        
        # Exploration & agent info
        if(exp_type in 'constant'):
            agent_info = ('DDQN with PrioritizedEpisodicReplayBuffer ' + 'Constant ' + chr(949) + '=' +
                          str(epsilon) + ' (' + chr(947) + '=' + str(gamma) +
                          ')')
            explorer = chainerrl.explorers.ConstantEpsilonGreedy(
                    epsilon=epsilon, random_action_func=exploration_func)
        elif(exp_type in 'linear decay'):
            if(type(epsilon) != list):
                raise KeyboardInterrupt(
                    'Error while creating agent: Linear decay exploration does'
                    'not work with a constant epsilon')
            agent_info = ('DDQN - ' + 'Linear decay ' + chr(949) + '=' +
                          str(epsilon[0]) + '->' + str(epsilon[1]) + ' in ' +
                          str(epsilon[2]) + ' time steps' + ' (' + chr(947) +
                          '=' + str(gamma) + ')')
            explorer = chainerrl.explorers.LinearDecayEpsilonGreedy(
            start_epsilon=epsilon[0], end_epsilon=epsilon[1],
            decay_steps=epsilon[2], random_action_func=exploration_func)
        else: # If type doesn't match with any known one raise error
            raise KeyboardInterrupt(
                'Error while creating agent: Unknown exploration type')
        
        # Experience replay
        # replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=1000000)
        betasteps = 5 * 10 ** 7 
        # Experience replay
        # replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=1000000)
        # replay_buffer = 
        # chainerrl.replay_buffer.PrioritizedEpisodicReplayBuffer(capacity=1000000,
        # alpha=0.5, beta0=0.4, betasteps=betasteps)

        replay_buffer = \
            chainerrl.replay_buffer.PrioritizedEpisodicReplayBuffer(
                capacity=10000,
                uniform_ratio=0.1,
                default_priority_func=exp_return_of_episode,
                wait_priority_after_sampling=False,
                return_sample_weights=False)
        # replay_buffer = chainerrl.replay_buffer.PrioritizedEpisodicReplayBuffer(
        #     capacity=1000000,
        # alpha=0.5, beta0=0.4)
        # replay_buffer =   chainerrl.replay_buffer.PrioritizedEpisodicReplayBuffer(
        # capacity=5 * 10 ** 3,
        # uniform_ratio=0.1,
        # default_priority_func=exp_return_of_episode,
        # wait_priority_after_sampling=False,
        # return_sample_weights=False)
    
        # Agent (DDQN)
        agent = chainerrl.agents.DoubleDQN(
                q_func, opt, replay_buffer, gamma, explorer,
                replay_start_size=100000, target_update_interval=50000)
        
        return agent, agent_info, q_func, opt, explorer, replay_buffer
    elif(alg in 'PS-DQN'):
        # Q function instanciation
        q_func = QFunction(obs_size, n_actions)
        
        # Optimizer
        opt = chainer.optimizers.Adam(eps=1e-2)
        opt.setup(q_func)
        
        # Exploration & agent info
        if(exp_type in 'constant'):
            agent_info = ('DQN with PER - ' + 'Constant ' + chr(949) + '=' +
                          str(epsilon) + ' (' + chr(947) + '=' + str(gamma) +
                          ')')
            explorer = chainerrl.explorers.ConstantEpsilonGreedy(
                    epsilon=epsilon, random_action_func=exploration_func)
        elif(exp_type in 'linear decay'):
            if(type(epsilon) != list):
                raise KeyboardInterrupt(
                    'Error while creating agent: Linear decay exploration does'
                    'not work with a constant epsilon')
            agent_info = ('DQN PER - ' + 'Linear decay ' + chr(949) + '=' +
                          str(epsilon[0]) + '->' + str(epsilon[1]) + ' in ' +
                          str(epsilon[2]) + ' time steps' + ' (' + chr(947) +
                          '=' + str(gamma) + ')')
            explorer = chainerrl.explorers.LinearDecayEpsilonGreedy(
            start_epsilon=epsilon[0], end_epsilon=epsilon[1],
            decay_steps=epsilon[2], random_action_func=exploration_func)
        else: # If type doesn't match with any known one raise error
            raise KeyboardInterrupt(
                'Error while creating agent: Unknown exploration type')
        # update_interval = 1
        betasteps = 5 * 10 ** 7 
        # Experience replay
        # replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=1000000)
        replay_buffer = chainerrl.replay_buffer.PrioritizedReplayBuffer(capacity=1000000,
        alpha=0.5, beta0=0.4, betasteps=betasteps,
        num_steps=3,normalize_by_max='memory')
        # Agent (DQN)
        agent = chainerrl.agents.DQN(q_function=q_func, optimizer=opt, replay_buffer=replay_buffer, gamma=gamma, explorer=explorer, 
                gpu=None, replay_start_size=100000, minibatch_size=32, 
                update_interval=1, target_update_interval=50000)
        
        return agent, agent_info, q_func, opt, explorer, replay_buffer
    elif(alg in 'DQN UI4'):
        # Q function instanciation
        q_func = QFunction(obs_size, n_actions)
        
        # Optimizer
        opt = chainer.optimizers.Adam(eps=1e-2)
        opt.setup(q_func)
        
        # Exploration & agent info
        if(exp_type in 'constant'):
            agent_info = ('DQN PER UI4 - ' + 'Constant ' + chr(949) + '=' +
                          str(epsilon) + ' (' + chr(947) + '=' + str(gamma) +
                          ')')
            explorer = chainerrl.explorers.ConstantEpsilonGreedy(
                    epsilon=epsilon, random_action_func=exploration_func)
        elif(exp_type in 'linear decay'):
            if(type(epsilon) != list):
                raise KeyboardInterrupt(
                    'Error while creating agent: Linear decay exploration does'
                    'not work with a constant epsilon')
            agent_info = ('DQN PER - ' + 'Linear decay ' + chr(949) + '=' +
                          str(epsilon[0]) + '->' + str(epsilon[1]) + ' in ' +
                          str(epsilon[2]) + ' time steps' + ' (' + chr(947) +
                          '=' + str(gamma) + ')')
            explorer = chainerrl.explorers.LinearDecayEpsilonGreedy(
            start_epsilon=epsilon[0], end_epsilon=epsilon[1],
            decay_steps=epsilon[2], random_action_func=exploration_func)
        else: # If type doesn't match with any known one raise error
            raise KeyboardInterrupt(
                'Error while creating agent: Unknown exploration type')
        # update_interval = 1
        betasteps = 5 * 10 ** 7/4
        # Experience replay
        # replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=1000000)
        replay_buffer = chainerrl.replay_buffer.PrioritizedReplayBuffer(capacity=1000000,
        alpha=0.5, beta0=0.4, betasteps=betasteps,
        num_steps=3,normalize_by_max='memory')
        # Agent (DQN)
        agent = chainerrl.agents.DQN(q_function=q_func, optimizer=opt, replay_buffer=replay_buffer, gamma=gamma, explorer=explorer, 
                gpu=None, replay_start_size=100000, minibatch_size=32, 
                update_interval=4, target_update_interval=50000)
        
        return agent, agent_info, q_func, opt, explorer, replay_buffer
    elif(alg in 'RAINBOW'):
        #带distribution 没有效果
        # Q function instanciation
        # q_func = DistributionalDuelingDQN(n_actions,n_atoms=51, v_min=-10, v_max = 10)
        # q_func = chainerrl.q_functions.DistributionalFCStateQFunctionWithDiscreteAction(
        # obs_size, n_actions, n_atoms=51, v_min=-10, v_max=10,
        # n_hidden_channels=60,
        # n_hidden_layers=2)     
        # 
        q_func = QFunction(obs_size, n_actions)         
        # Optimizer
        # opt = chainer.optimizers.Adam(eps=1e-2)
        opt = chainer.optimizers.Adam(6.25e-5, eps=1.5 * 10 ** -4)
        opt.setup(q_func)
        # Noisy nets
        chainerrl.links.to_factorized_noisy(q_func, sigma_scale=0.5)
        # Turn off explorer
        
        # Exploration & agent info
        if(exp_type in 'constant'):
            # agent_info = ('RAINBOW - ' + 'Constant ' + chr(949) + '=' +
            #               str(epsilon) + ' (' + chr(947) + '=' + str(gamma) +
            #               ')')
            agent_info = ('DDQN+PER+Noisy+Distribution - ' + 'Constant ' + chr(949) + '=' +
                           ' (' + chr(947) + '=' + str(gamma) +
                          ')')
            explorer = chainerrl.explorers.Greedy()
            # explorer = chainerrl.explorers.ConstantEpsilonGreedy(
            #         epsilon=epsilon, random_action_func=exploration_func)
        elif(exp_type in 'linear decay'):
            if(type(epsilon) != list):
                raise KeyboardInterrupt(
                    'Error while creating agent: Linear decay exploration does'
                    'not work with a constant epsilon')
            agent_info = ('RAINBOW - ' + 'Linear decay ' + chr(949) + '=' +
                          str(epsilon[0]) + '->' + str(epsilon[1]) + ' in ' +
                          str(epsilon[2]) + ' time steps' + ' (' + chr(947) +
                          '=' + str(gamma) + ')')
            explorer = chainerrl.explorers.LinearDecayEpsilonGreedy(
            start_epsilon=epsilon[0], end_epsilon=epsilon[1],
            decay_steps=epsilon[2], random_action_func=exploration_func)
        else: # If type doesn't match with any known one raise error
            raise KeyboardInterrupt(
                'Error while creating agent: Unknown exploration type')
        update_interval = 4
        betasteps = 5 * 10 ** 7 / update_interval
        replay_buffer = chainerrl.replay_buffer.PrioritizedReplayBuffer(capacity=1000000,
        alpha=0.5, beta0=0.4, betasteps=betasteps,
        num_steps=3,normalize_by_max='memory')
        # Experience replay
        # replay_buffer = chainerrl.replay_buffer.PrioritizedReplayBuffer(capacity=1000000,
        # alpha=0.5, beta0=0.4, betasteps=betasteps,
        # num_steps=3,normalize_by_max='memory')
        # replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=1000000)
        def phi(x):
        # Feature extractor
            return np.asarray(x, dtype=np.float32) / 255
        # agent = chainerrl.agents.DoubleDQN(
        #         q_func, opt, replay_buffer, gamma, explorer,
        #         replay_start_size=100000, target_update_interval=50000)
        agent = chainerrl.agents.CategoricalDoubleDQN(q_function=q_func, optimizer=opt, replay_buffer=replay_buffer, gamma=gamma, explorer=explorer, 
                gpu=None, replay_start_size=100000, minibatch_size=32, 
                update_interval=4, target_update_interval=50000,
                batch_accumulator='mean',phi=phi)
        
        return agent, agent_info, q_func, opt, explorer, replay_buffer
    elif(alg in 'IQN'):
            # Q function instanciation
            # q_func = DistributionalDuelingDQN(n_actions,n_atoms=51, v_min=-10, v_max = 10)
        hidden_size = 64
        q_func = chainerrl.agents.iqn.ImplicitQuantileQFunction(
            psi=chainerrl.links.Sequence(
                L.Linear(obs_size, hidden_size),
                F.relu,
            ),
            phi=chainerrl.links.Sequence(
                chainerrl.agents.iqn.CosineBasisLinear(64, hidden_size),
                F.relu,
            ),
            f=L.Linear(hidden_size, n_actions),
        )
        # Use epsilon-greedy for exploration
        # Exploration & agent info
        if(exp_type in 'constant'):
            # agent_info = ('RAINBOW - ' + 'Constant ' + chr(949) + '=' +
            #               str(epsilon) + ' (' + chr(947) + '=' + str(gamma) +
            #               ')')
            agent_info = ('IQN - ' + 'Constant ' + chr(949) + '=' +
                           ' (' + chr(947) + '=' + str(gamma) +
                          ')')
            explorer = chainerrl.explorers.Greedy()
            # explorer = chainerrl.explorers.ConstantEpsilonGreedy(
            #         epsilon=epsilon, random_action_func=exploration_func)
        elif(exp_type in 'linear decay'):
            if(type(epsilon) != list):
                raise KeyboardInterrupt(
                    'Error while creating agent: Linear decay exploration does'
                    'not work with a constant epsilon')
            agent_info = ('RAINBOW - ' + 'Linear decay ' + chr(949) + '=' +
                          str(epsilon[0]) + '->' + str(epsilon[1]) + ' in ' +
                          str(epsilon[2]) + ' time steps' + ' (' + chr(947) +
                          '=' + str(gamma) + ')')
            explorer = chainerrl.explorers.LinearDecayEpsilonGreedy(
            start_epsilon=epsilon[0], end_epsilon=epsilon[1],
            decay_steps=epsilon[2], random_action_func=exploration_func)
        else: # If type doesn't match with any known one raise error
            raise KeyboardInterrupt(
                'Error while creating agent: Unknown exploration type')        

        opt = chainer.optimizers.Adam(1e-3)
        opt.setup(q_func)

        rbuf_capacity = 1000000  # 5 * 10 ** 5
        rbuf =  chainerrl.replay_buffer.ReplayBuffer(rbuf_capacity)
        agent = chainerrl.agents.IQN(
            q_func, opt, rbuf, gpu=None, gamma=gamma,
            explorer=explorer, replay_start_size=100000,
            target_update_interval=50000,
            update_interval=1,
            minibatch_size=32
        )
        
        return agent, agent_info, q_func, opt, explorer, rbuf
    elif(alg in 'DOUBLEIQN'):
            # Q function instanciation
            # q_func = DistributionalDuelingDQN(n_actions,n_atoms=51, v_min=-10, v_max = 10)
        hidden_size = 64
        q_func = chainerrl.agents.iqn.ImplicitQuantileQFunction(
            psi=chainerrl.links.Sequence(
                L.Linear(obs_size, hidden_size),
                F.relu,
            ),
            phi=chainerrl.links.Sequence(
                chainerrl.agents.iqn.CosineBasisLinear(64, hidden_size),
                F.relu,
            ),
            f=L.Linear(hidden_size, n_actions),
        )
        # Use epsilon-greedy for exploration
        # Exploration & agent info
        if(exp_type in 'constant'):
            # agent_info = ('RAINBOW - ' + 'Constant ' + chr(949) + '=' +
            #               str(epsilon) + ' (' + chr(947) + '=' + str(gamma) +
            #               ')')
            agent_info = ('DOUBLEIQN - ' + 'Constant ' + chr(949) + '=' +
                           ' (' + chr(947) + '=' + str(gamma) +
                          ')')
            explorer = chainerrl.explorers.Greedy()
            # explorer = chainerrl.explorers.ConstantEpsilonGreedy(
            #         epsilon=epsilon, random_action_func=exploration_func)
        elif(exp_type in 'linear decay'):
            if(type(epsilon) != list):
                raise KeyboardInterrupt(
                    'Error while creating agent: Linear decay exploration does'
                    'not work with a constant epsilon')
            agent_info = ('RAINBOW - ' + 'Linear decay ' + chr(949) + '=' +
                          str(epsilon[0]) + '->' + str(epsilon[1]) + ' in ' +
                          str(epsilon[2]) + ' time steps' + ' (' + chr(947) +
                          '=' + str(gamma) + ')')
            explorer = chainerrl.explorers.LinearDecayEpsilonGreedy(
            start_epsilon=epsilon[0], end_epsilon=epsilon[1],
            decay_steps=epsilon[2], random_action_func=exploration_func)
        else: # If type doesn't match with any known one raise error
            raise KeyboardInterrupt(
                'Error while creating agent: Unknown exploration type')        

        opt = chainer.optimizers.Adam(1e-3)
        opt.setup(q_func)

        rbuf_capacity = 1000000  # 5 * 10 ** 5
        
        rbuf =  chainerrl.replay_buffer.ReplayBuffer(rbuf_capacity)

        agent = chainerrl.agents.DoubleIQN(
            q_func, opt, rbuf, gpu=None, gamma=gamma,
            explorer=explorer, replay_start_size=100000,
            target_update_interval=50000,
            update_interval=1,
            minibatch_size=32
        )
        
        return agent, agent_info, q_func, opt, explorer, rbuf
    elif(alg in 'PS-DOUBLEIQNNOISY'):
        hidden_size = 128
        q_func = chainerrl.agents.iqn.ImplicitQuantileQFunction(
            psi=chainerrl.links.Sequence(
                L.Linear(obs_size, hidden_size),
                F.relu,
            ),
            phi=chainerrl.links.Sequence(
                chainerrl.agents.iqn.CosineBasisLinear(64, hidden_size),
                F.relu,
            ),
            f=L.Linear(hidden_size, n_actions),
        )
        # q_func.to_gpu(1)
        # Noisy nets
        chainerrl.links.to_factorized_noisy(q_func, sigma_scale=0.5)
        opt = chainer.optimizers.Adam(6.25e-5, eps=1.5 * 10 ** -4)
        # Use epsilon-greedy for exploration
        # Exploration & agent info
        if(exp_type in 'constant'):
            # agent_info = ('RAINBOW - ' + 'Constant ' + chr(949) + '=' +
            #               str(epsilon) + ' (' + chr(947) + '=' + str(gamma) +
            #               ')')
            agent_info = ('DOUBLEIQN - NOISY NET -PER - Stratified+Prioritized + ' + 'Constant ' + chr(949) + '=' +
                           ' (' + chr(947) + '=' + str(gamma) +
                          ')')
            explorer = chainerrl.explorers.Greedy()
            # explorer = chainerrl.explorers.ConstantEpsilonGreedy(epsilon=0.1, random_action_func=exploration_func)
            # explorer = chainerrl.explorers.ConstantEpsilonGreedy(
            #         epsilon=epsilon, random_action_func=exploration_func)
        elif(exp_type in 'linear decay'):
            if(type(epsilon) != list):
                raise KeyboardInterrupt(
                    'Error while creating agent: Linear decay exploration does'
                    'not work with a constant epsilon')
            agent_info = ('RAINBOW - ' + 'Linear decay ' + chr(949) + '=' +
                          str(epsilon[0]) + '->' + str(epsilon[1]) + ' in ' +
                          str(epsilon[2]) + ' time steps' + ' (' + chr(947) +
                          '=' + str(gamma) + ')')
            explorer = chainerrl.explorers.LinearDecayEpsilonGreedy(
            start_epsilon=epsilon[0], end_epsilon=epsilon[1],
            decay_steps=epsilon[2], random_action_func=exploration_func)
        else: # If type doesn't match with any known one raise error
            raise KeyboardInterrupt(
                'Error while creating agent: Unknown exploration type')        

        opt = chainer.optimizers.Adam(1e-3)
        opt.setup(q_func)

        betasteps = 10**6
        # Experience replay
        # replay_buffer = CustomPrioritizedBuffer(capacity=1000000, alpha=0.6, beta0=0.4, betasteps=betasteps,
        # num_steps=3,normalize_by_max='memory')
        replay_buffer = chainerrl.replay_buffer.PrioritizedReplayBuffer(capacity=1000000,
        alpha=0.5, beta0=0.4, betasteps=betasteps,
        num_steps=3,normalize_by_max='memory')
        # replay_buffer = CustomPrioritizedReplayBuffer(capacity=1000000, alpha=0.6, beta0=0.4, betasteps=betasteps, num_steps=3, normalize_by_max='memory')

        agent = chainerrl.agents.DoubleIQN(
            q_func, opt, replay_buffer, gpu=None, gamma=gamma,
            explorer=explorer, replay_start_size=100000,
            target_update_interval=10000,
            update_interval=1,
            minibatch_size=32
        )
        
        return agent, agent_info, q_func, opt, explorer, replay_buffer
    elif(alg in 'PS-UNIFORMDOUBLEIQNNOISY'):
        hidden_size = 128
        q_func = chainerrl.agents.iqn.ImplicitQuantileQFunction(
            psi=chainerrl.links.Sequence(
                L.Linear(obs_size, hidden_size),
                F.relu,
            ),
            phi=chainerrl.links.Sequence(
                chainerrl.agents.iqn.CosineBasisLinear(64, hidden_size),
                F.relu,
            ),
            f=L.Linear(hidden_size, n_actions),
        )
        # q_func.to_gpu(1)
        # Noisy nets
        chainerrl.links.to_factorized_noisy(q_func, sigma_scale=0.5)
        opt = chainer.optimizers.Adam(6.25e-5, eps=1.5 * 10 ** -4)
        # Use epsilon-greedy for exploration
        # Exploration & agent info
        if(exp_type in 'constant'):
            # agent_info = ('RAINBOW - ' + 'Constant ' + chr(949) + '=' +
            #               str(epsilon) + ' (' + chr(947) + '=' + str(gamma) +
            #               ')')
            agent_info = ('DOUBLEIQN - NOISY NET -PER - UNIFORM - Stratified + Uniform + Prioritized + ' + 'Constant ' + chr(949) + '=' +
                           ' (' + chr(947) + '=' + str(gamma) +
                          ')')
            explorer = chainerrl.explorers.Greedy()
            # explorer = chainerrl.explorers.ConstantEpsilonGreedy(epsilon=0.1, random_action_func=exploration_func)
            # explorer = chainerrl.explorers.ConstantEpsilonGreedy(
            #         epsilon=epsilon, random_action_func=exploration_func)
        elif(exp_type in 'linear decay'):
            if(type(epsilon) != list):
                raise KeyboardInterrupt(
                    'Error while creating agent: Linear decay exploration does'
                    'not work with a constant epsilon')
            agent_info = ('RAINBOW - ' + 'Linear decay ' + chr(949) + '=' +
                          str(epsilon[0]) + '->' + str(epsilon[1]) + ' in ' +
                          str(epsilon[2]) + ' time steps' + ' (' + chr(947) +
                          '=' + str(gamma) + ')')
            explorer = chainerrl.explorers.LinearDecayEpsilonGreedy(
            start_epsilon=epsilon[0], end_epsilon=epsilon[1],
            decay_steps=epsilon[2], random_action_func=exploration_func)
        else: # If type doesn't match with any known one raise error
            raise KeyboardInterrupt(
                'Error while creating agent: Unknown exploration type')        

        opt = chainer.optimizers.Adam(1e-3)
        opt.setup(q_func)

        betasteps = 10**6
        # Experience replay
        # replay_buffer = CustomPrioritizedBuffer(capacity=1000000, alpha=0.6, beta0=0.4, betasteps=betasteps,
        # num_steps=3,normalize_by_max='memory')
        replay_buffer = chainerrl.replay_buffers.UniformPrioritizedReplayBuffer(capacity=1000000,
        alpha=0.5, beta0=0.4, betasteps=betasteps,
        num_steps=3,normalize_by_max='memory')
        # replay_buffer = CustomPrioritizedReplayBuffer(capacity=1000000, alpha=0.6, beta0=0.4, betasteps=betasteps, num_steps=3, normalize_by_max='memory')

        agent = chainerrl.agents.DoubleIQN(
            q_func, opt, replay_buffer, gpu=None, gamma=gamma,
            explorer=explorer, replay_start_size=100000,
            target_update_interval=10000,
            update_interval=1,
            minibatch_size=32
        )
        
        return agent, agent_info, q_func, opt, explorer, replay_buffer
    elif(alg in 'PS-TIMEDOUBLEIQNNOISY'):
        hidden_size = 128
        q_func = chainerrl.agents.iqn.ImplicitQuantileQFunction(
            psi=chainerrl.links.Sequence(
                L.Linear(obs_size, hidden_size),
                F.relu,
            ),
            phi=chainerrl.links.Sequence(
                chainerrl.agents.iqn.CosineBasisLinear(64, hidden_size),
                F.relu,
            ),
            f=L.Linear(hidden_size, n_actions),
        )
        # q_func.to_gpu(1)
        # Noisy nets
        chainerrl.links.to_factorized_noisy(q_func, sigma_scale=0.5)
        opt = chainer.optimizers.Adam(6.25e-5, eps=1.5 * 10 ** -4)
        # Use epsilon-greedy for exploration
        # Exploration & agent info
        if(exp_type in 'constant'):
            # agent_info = ('RAINBOW - ' + 'Constant ' + chr(949) + '=' +
            #               str(epsilon) + ' (' + chr(947) + '=' + str(gamma) +
            #               ')')
            agent_info = ('DOUBLEIQN - NOISY NET -PER - Stratified + Time + Prioritized + ' + 'Constant ' + chr(949) + '=' +
                           ' (' + chr(947) + '=' + str(gamma) +
                          ')')
            explorer = chainerrl.explorers.Greedy()
            # explorer = chainerrl.explorers.ConstantEpsilonGreedy(epsilon=0.1, random_action_func=exploration_func)
            # explorer = chainerrl.explorers.ConstantEpsilonGreedy(
            #         epsilon=epsilon, random_action_func=exploration_func)
        elif(exp_type in 'linear decay'):
            if(type(epsilon) != list):
                raise KeyboardInterrupt(
                    'Error while creating agent: Linear decay exploration does'
                    'not work with a constant epsilon')
            agent_info = ('RAINBOW - ' + 'Linear decay ' + chr(949) + '=' +
                          str(epsilon[0]) + '->' + str(epsilon[1]) + ' in ' +
                          str(epsilon[2]) + ' time steps' + ' (' + chr(947) +
                          '=' + str(gamma) + ')')
            explorer = chainerrl.explorers.LinearDecayEpsilonGreedy(
            start_epsilon=epsilon[0], end_epsilon=epsilon[1],
            decay_steps=epsilon[2], random_action_func=exploration_func)
        else: # If type doesn't match with any known one raise error
            raise KeyboardInterrupt(
                'Error while creating agent: Unknown exploration type')        

        opt = chainer.optimizers.Adam(1e-3)
        opt.setup(q_func)

        betasteps = 10**6
        # Experience replay
        # replay_buffer = CustomPrioritizedBuffer(capacity=1000000, alpha=0.6, beta0=0.4, betasteps=betasteps,
        # num_steps=3,normalize_by_max='memory')
        # replay_buffer = chainerrl.replay_buffers.TimePrioritizedReplayBuffer(capacity=1000000,
        replay_buffer = chainerrl.replay_buffers.PrioritizedReplayBuffer(capacity=1000000,
        alpha=0.5, beta0=0.4, betasteps=betasteps,
        num_steps=3,normalize_by_max='memory')
        # replay_buffer = CustomPrioritizedReplayBuffer(capacity=1000000, alpha=0.6, beta0=0.4, betasteps=betasteps, num_steps=3, normalize_by_max='memory')

        agent = chainerrl.agents.DoubleIQN(
            q_func, opt, replay_buffer, gpu=None, gamma=gamma,
            explorer=explorer, replay_start_size=80000,
            target_update_interval=30000,
            update_interval=1,
            minibatch_size=32
        )
        
        return agent, agent_info, q_func, opt, explorer, replay_buffer
    elif(alg in 'PS-DDQNNOISE'):
        # Q function instanciation
        # q_func = DistributionalDuelingDQN(n_actions,n_atoms=51, v_min=-10, v_max = 10)
        # q_func = chainerrl.q_functions.DistributionalFCStateQFunctionWithDiscreteAction(
        # obs_size, n_actions, n_atoms=51, v_min=-10, v_max=10,
        # n_hidden_channels=60,
        # n_hidden_layers=2)
        # Optimizer
        # opt = chainer.optimizers.Adam(eps=1e-
        q_func = QFunction(obs_size, n_actions)
        
        # Noisy nets
        chainerrl.links.to_factorized_noisy(q_func, sigma_scale=0.5)
        opt = chainer.optimizers.Adam(6.25e-5, eps=1.5 * 10 ** -4)
        # Turn off explorer
        opt.setup(q_func)
        # Exploration & agent info
        if(exp_type in 'constant'):
            # agent_info = ('RAINBOW - ' + 'Constant ' + chr(949) + '=' +
            #               str(epsilon) + ' (' + chr(947) + '=' + str(gamma) +
            #               ')')
            agent_info = ('DDQN+PER+NOISY - ' + 'Constant ' + chr(949) + '=' +
                           ' (' + chr(947) + '=' + str(gamma) +
                          ')')
            explorer = chainerrl.explorers.Greedy()
            # explorer = chainerrl.explorers.ConstantEpsilonGreedy(
            #         epsilon=epsilon, random_action_func=exploration_func)
        elif(exp_type in 'linear decay'):
            if(type(epsilon) != list):
                raise KeyboardInterrupt(
                    'Error while creating agent: Linear decay exploration does'
                    'not work with a constant epsilon')
            agent_info = ('RAINBOW - ' + 'Linear decay ' + chr(949) + '=' +
                          str(epsilon[0]) + '->' + str(epsilon[1]) + ' in ' +
                          str(epsilon[2]) + ' time steps' + ' (' + chr(947) +
                          '=' + str(gamma) + ')')
            explorer = chainerrl.explorers.LinearDecayEpsilonGreedy(
            start_epsilon=epsilon[0], end_epsilon=epsilon[1],
            decay_steps=epsilon[2], random_action_func=exploration_func)
        else: # If type doesn't match with any known one raise error
            raise KeyboardInterrupt(
                'Error while creating agent: Unknown exploration type')
        update_interval = 4
        betasteps = 5 * 10 ** 7 / update_interval
        replay_buffer = chainerrl.replay_buffer.PrioritizedReplayBuffer(capacity=1000000,
        alpha=0.5, beta0=0.4, betasteps=betasteps,
        num_steps=3,normalize_by_max='memory')
        # Experience replay
        # replay_buffer = chainerrl.replay_buffer.PrioritizedReplayBuffer(capacity=1000000,
        # alpha=0.5, beta0=0.4, betasteps=betasteps,
        # num_steps=3,normalize_by_max='memory')
        # replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=1000000)
        def phi(x):
        # Feature extractor
            return np.asarray(x, dtype=np.float32) / 255
        
        # agent = chainerrl.agents.CategoricalDoubleDQN(q_function=q_func, optimizer=opt, replay_buffer=replay_buffer, gamma=gamma, explorer=explorer, 
        #         gpu=None, replay_start_size=100000, minibatch_size=32, 
        #         update_interval=4, target_update_interval=64000,
        #         batch_accumulator='mean',phi=phi)
        agent = chainerrl.agents.DoubleDQN(
                q_func, opt, replay_buffer, gamma, explorer,
                replay_start_size=100000, target_update_interval=50000)
        
        return agent, agent_info, q_func, opt, explorer, replay_buffer
        
        return agent, agent_info, q_func, opt, explorer, replay_buffer
    elif(alg in 'A3CFF'):
        # q_func = QFunction(obs_size, n_actions)
        # Optimizer
        opt = chainerrl.optimizers.rmsprop_async.RMSpropAsync(lr = 7e-4, eps=1e-2,alpha=0.99)
        model = A3CFFSoftmax(obs_size, n_actions)
        opt.setup(model)
        opt.add_hook(chainer.optimizer.GradientClipping(40))
        # Exploration & agent info
        if(exp_type in 'constant'):
            agent_info = ('A3C - tmax=5 - FFSoftmax' +  ' (' + chr(947) + '=' + str(gamma) +
                          ')')
            # explorer = chainerrl.explorers.ConstantEpsilonGreedy(
            #         epsilon=epsilon, random_action_func=exploration_func)
        elif(exp_type in 'linear decay'):
            if(type(epsilon) != list):
                raise KeyboardInterrupt(
                    'Error while creating agent: Linear decay exploration does'
                    'not work with a constant epsilon')
            agent_info = ('A3C - ' + 'Linear decay ' + chr(949) + '=' +
                          str(epsilon[0]) + '->' + str(epsilon[1]) + ' in ' +
                          str(epsilon[2]) + ' time steps' + ' (' + chr(947) +
                          '=' + str(gamma) + ')')
            explorer = chainerrl.explorers.LinearDecayEpsilonGreedy(
            start_epsilon=epsilon[0], end_epsilon=epsilon[1],
            decay_steps=epsilon[2], random_action_func=exploration_func)
        else: # If type doesn't match with any known one raise error
            raise KeyboardInterrupt(
                'Error while creating agent: Unknown exploration type')
        
        # Experience replay
        # replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=1000000)
        # Agent (A3C)
        agent = chainerrl.agents.A3C(model = model, optimizer = opt, t_max=5, gamma=gamma, beta=0.01,process_idx=4)
        return agent, agent_info, model ,opt
    elif(alg in 'A3CFC'):
        # q_func = QFunction(obs_size, n_actions)
        # Optimizer
        opt = chainerrl.optimizers.rmsprop_async.RMSpropAsync(lr = 7e-4, eps=1e-1,alpha=0.99)
        model = A3CFCSoftmax(obs_size, n_actions)
        opt.setup(model)
        opt.add_hook(chainer.optimizer.GradientClipping(40))
        # Exploration & agent info
        if(exp_type in 'constant'):
            agent_info = ('A3C - tmax=5 - FCSoftmax' +  ' (' + chr(947) + '=' + str(gamma) +
                          ')')
            # explorer = chainerrl.explorers.ConstantEpsilonGreedy(
            #         epsilon=epsilon, random_action_func=exploration_func)
        elif(exp_type in 'linear decay'):
            if(type(epsilon) != list):
                raise KeyboardInterrupt(
                    'Error while creating agent: Linear decay exploration does'
                    'not work with a constant epsilon')
            agent_info = ('A3C - ' + 'Linear decay ' + chr(949) + '=' +
                          str(epsilon[0]) + '->' + str(epsilon[1]) + ' in ' +
                          str(epsilon[2]) + ' time steps' + ' (' + chr(947) +
                          '=' + str(gamma) + ')')
            explorer = chainerrl.explorers.LinearDecayEpsilonGreedy(
            start_epsilon=epsilon[0], end_epsilon=epsilon[1],
            decay_steps=epsilon[2], random_action_func=exploration_func)
        else: # If type doesn't match with any known one raise error
            raise KeyboardInterrupt(
                'Error while creating agent: Unknown exploration type')
        
        # Experience replay
        # replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=1000000)
        
        # Agent (A3C)
        agent = chainerrl.agents.A3C(model = model, optimizer = opt, t_max=5, gamma=gamma, beta=0.01)
        return agent, agent_info, model ,opt
    elif(alg in '2A3C'):
        # q_func = QFunction(obs_size, n_actions)
        
        # Optimizer
        opt = chainerrl.optimizers.rmsprop_async.RMSpropAsync(lr = 7e-4, eps=1e-2,alpha=0.99)
        
        model = A3CFFSoftmax(obs_size, n_actions)
        opt.setup(model)
        opt.add_hook(chainer.optimizer.GradientClipping(40))
        # Exploration & agent info
        if(exp_type in 'constant'):
            agent_info = ('A3C - tmax=2 -' +  ' (' + chr(947) + '=' + str(gamma) +
                          ')')
            # explorer = chainerrl.explorers.ConstantEpsilonGreedy(
            #         epsilon=epsilon, random_action_func=exploration_func)
        elif(exp_type in 'linear decay'):
            if(type(epsilon) != list):
                raise KeyboardInterrupt(
                    'Error while creating agent: Linear decay exploration does'
                    'not work with a constant epsilon')
            agent_info = ('A3C - ' + 'Linear decay ' + chr(949) + '=' +
                          str(epsilon[0]) + '->' + str(epsilon[1]) + ' in ' +
                          str(epsilon[2]) + ' time steps' + ' (' + chr(947) +
                          '=' + str(gamma) + ')')
            explorer = chainerrl.explorers.LinearDecayEpsilonGreedy(
            start_epsilon=epsilon[0], end_epsilon=epsilon[1],
            decay_steps=epsilon[2], random_action_func=exploration_func)
        else: # If type doesn't match with any known one raise error
            raise KeyboardInterrupt(
                'Error while creating agent: Unknown exploration type')
        
        # Experience replay
        # replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=1000000)
        
        # Agent (A3C)
        agent = chainerrl.agents.A3C(model = model, optimizer = opt, t_max=2, gamma=gamma, beta=0.01)
        
        return agent, agent_info, model ,opt        
    elif(alg in '10A3C'):
        # q_func = QFunction(obs_size, n_actions)
        
        # Optimizer
        opt = chainerrl.optimizers.rmsprop_async.RMSpropAsync(lr = 7e-4, eps=1e-2,alpha=0.99)
        
        model = A3CFFSoftmax(obs_size, n_actions)
        opt.setup(model)
        opt.add_hook(chainer.optimizer.GradientClipping(40))
        # Exploration & agent info
        if(exp_type in 'constant'):
            agent_info = ('A3C - tmax=10 -' +  ' (' + chr(947) + '=' + str(gamma) +
                          ')')
            # explorer = chainerrl.explorers.ConstantEpsilonGreedy(
            #         epsilon=epsilon, random_action_func=exploration_func)
        elif(exp_type in 'linear decay'):
            if(type(epsilon) != list):
                raise KeyboardInterrupt(
                    'Error while creating agent: Linear decay exploration does'
                    'not work with a constant epsilon')
            agent_info = ('A3C - ' + 'Linear decay ' + chr(949) + '=' +
                          str(epsilon[0]) + '->' + str(epsilon[1]) + ' in ' +
                          str(epsilon[2]) + ' time steps' + ' (' + chr(947) +
                          '=' + str(gamma) + ')')
            explorer = chainerrl.explorers.LinearDecayEpsilonGreedy(
            start_epsilon=epsilon[0], end_epsilon=epsilon[1],
            decay_steps=epsilon[2], random_action_func=exploration_func)
        else: # If type doesn't match with any known one raise error
            raise KeyboardInterrupt(
                'Error while creating agent: Unknown exploration type')
        
        # Experience replay
        # replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=1000000)
        
        # Agent (A3C)
        agent = chainerrl.agents.A3C(model = model, optimizer = opt, t_max=10, gamma=gamma, beta=0.01)
        
        return agent, agent_info, model ,opt
    elif(alg in '20A3C'):
        # q_func = QFunction(obs_size, n_actions)
        
        # Optimizer
        opt = chainerrl.optimizers.rmsprop_async.RMSpropAsync(lr = 7e-4, eps=1e-2,alpha=0.99)
        
        model = A3CFFSoftmax(obs_size, n_actions)
        opt.setup(model)
        opt.add_hook(chainer.optimizer.GradientClipping(40))
        # Exploration & agent info
        if(exp_type in 'constant'):
            agent_info = ('A3C - tmax=20 -' +  ' (' + chr(947) + '=' + str(gamma) +
                          ')')
            # explorer = chainerrl.explorers.ConstantEpsilonGreedy(
            #         epsilon=epsilon, random_action_func=exploration_func)
        elif(exp_type in 'linear decay'):
            if(type(epsilon) != list):
                raise KeyboardInterrupt(
                    'Error while creating agent: Linear decay exploration does'
                    'not work with a constant epsilon')
            agent_info = ('A3C - ' + 'Linear decay ' + chr(949) + '=' +
                          str(epsilon[0]) + '->' + str(epsilon[1]) + ' in ' +
                          str(epsilon[2]) + ' time steps' + ' (' + chr(947) +
                          '=' + str(gamma) + ')')
            explorer = chainerrl.explorers.LinearDecayEpsilonGreedy(
            start_epsilon=epsilon[0], end_epsilon=epsilon[1],
            decay_steps=epsilon[2], random_action_func=exploration_func)
        else: # If type doesn't match with any known one raise error
            raise KeyboardInterrupt(
                'Error while creating agent: Unknown exploration type')
        
        # Experience replay
        # replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=1000000)
        
        # Agent (A3C)
        agent = chainerrl.agents.A3C(model = model, optimizer = opt, t_max=20, gamma=gamma, beta=0.01)
        
        return agent, agent_info, model ,opt
    elif(alg in 'A2C'):

        # Q function instanciation
        q_func = QFunction(obs_size, n_actions)
        
        # Optimizer
        opt = chainer.optimizers.RMSprop(lr = 7e-4,eps = 1e-5 ,alpha = 0.99)
        
        model = A2CFFSoftmax(obs_size, n_actions)
        opt.setup(model)
        # Exploration & agent info
        if(exp_type in 'constant'):
            agent_info = ('A2C - ' +  ' (' + chr(947) + '=' + str(gamma) +
                          ')')
            # explorer = chainerrl.explorers.ConstantEpsilonGreedy(
            #         epsilon=epsilon, random_action_func=exploration_func)
        elif(exp_type in 'linear decay'):
            if(type(epsilon) != list):
                raise KeyboardInterrupt(
                    'Error while creating agent: Linear decay exploration does'
                    'not work with a constant epsilon')
            agent_info = ('A2C - ' + 'Linear decay ' + chr(949) + '=' +
                          str(epsilon[0]) + '->' + str(epsilon[1]) + ' in ' +
                          str(epsilon[2]) + ' time steps' + ' (' + chr(947) +
                          '=' + str(gamma) + ')')
            explorer = chainerrl.explorers.LinearDecayEpsilonGreedy(
            start_epsilon=epsilon[0], end_epsilon=epsilon[1],
            decay_steps=epsilon[2], random_action_func=exploration_func)
        else: # If type doesn't match with any known one raise error
            raise KeyboardInterrupt(
                'Error while creating agent: Unknown exploration type')
        
        # Experience replay
        # replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=1000000)
        
        # Agent (A2C)
        def phi(x):
    # Feature extractor
            return np.asarray(x, dtype=np.float32) / 255
        agent = chainerrl.agents.A2C(model=model, optimizer = opt , gamma = gamma,num_processes = 1,phi=phi)
        
        return agent, agent_info, model, opt
    elif(alg in 'NSQ5_'):
        #   
        # Q function instanciation
        q_func = QFunction(obs_size, n_actions)
        opt = chainerrl.optimizers.rmsprop_async.RMSpropAsync(lr=7e-4, eps=1e-1, alpha=0.99)
        opt.setup(q_func)
        def phi(x):
        # Feature extractor
            return np.asarray(x, dtype=np.float32) / 255
        # Exploration & agent info
        if(exp_type in 'constant'):
            agent_info = ('NSQ5_ - ' + 'Constant ' + chr(949) + '=' +
                          str(epsilon) + ' (' + chr(947) + '=' + str(gamma) +
                          ')')
            explorer = chainerrl.explorers.ConstantEpsilonGreedy(
                    epsilon=epsilon, random_action_func=exploration_func)
        elif(exp_type in 'linear decay'):
            if(type(epsilon) != list):
                raise KeyboardInterrupt(
                    'Error while creating agent: Linear decay exploration does'
                    'not work with a constant epsilon')
            agent_info = ('NSQ5_ - ' + 'Linear decay ' + chr(949) + '=' +
                          str(epsilon[0]) + '->' + str(epsilon[1]) + ' in ' +
                          str(epsilon[2]) + ' time steps' + ' (' + chr(947) +
                          '=' + str(gamma) + ')')
            explorer = chainerrl.explorers.LinearDecayEpsilonGreedy(
            start_epsilon=epsilon[0], end_epsilon=epsilon[1],
            decay_steps=epsilon[2], random_action_func=exploration_func)
        else: # If type doesn't match with any known one raise error
            raise KeyboardInterrupt(
                'Error while creating agent: Unknown exploration type')
        
        # Experience replay
        # replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=10 ** 6)
        
        # Agent (NSQ)
        agent = chainerrl.agents.NSQ(q_func, opt, t_max=5, gamma=0.99,
        i_target=50000,explorer=explorer,phi=phi)
        
        return agent, agent_info, q_func, opt, explorer
    elif(alg in 'NSQ5'):
        #   
        # Q function instanciation
        q_func = QFunction(obs_size, n_actions)
        
        # Optimizer
        opt = chainer.optimizers.Adam(eps=1e-2)
        opt.setup(q_func)
        
        # Exploration & agent info
        if(exp_type in 'constant'):
            agent_info = ('NSQ5 - ' + 'Constant ' + chr(949) + '=' +
                          str(epsilon) + ' (' + chr(947) + '=' + str(gamma) +
                          ')')
            explorer = chainerrl.explorers.ConstantEpsilonGreedy(
                    epsilon=epsilon, random_action_func=exploration_func)
        elif(exp_type in 'linear decay'):
            if(type(epsilon) != list):
                raise KeyboardInterrupt(
                    'Error while creating agent: Linear decay exploration does'
                    'not work with a constant epsilon')
            agent_info = ('NSQ5 - ' + 'Linear decay ' + chr(949) + '=' +
                          str(epsilon[0]) + '->' + str(epsilon[1]) + ' in ' +
                          str(epsilon[2]) + ' time steps' + ' (' + chr(947) +
                          '=' + str(gamma) + ')')
            explorer = chainerrl.explorers.LinearDecayEpsilonGreedy(
            start_epsilon=epsilon[0], end_epsilon=epsilon[1],
            decay_steps=epsilon[2], random_action_func=exploration_func)
        else: # If type doesn't match with any known one raise error
            raise KeyboardInterrupt(
                'Error while creating agent: Unknown exploration type')
        
        # Experience replay
        replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=10 ** 6)
        
        # Agent (NSQ)
        agent = chainerrl.agents.NSQ(q_func, opt, t_max=5, gamma=0.99,
        i_target=50000,explorer=explorer)
        
        return agent, agent_info, q_func, opt, explorer, replay_buffer 
    elif(alg in 'NSQ10'):
        #   
        # Q function instanciation
        q_func = QFunction(obs_size, n_actions)
        
        # Optimizer
        opt = chainer.optimizers.Adam(eps=1e-2)
        opt.setup(q_func)
        
        # Exploration & agent info
        if(exp_type in 'constant'):
            agent_info = ('NSQ10 - ' + 'Constant ' + chr(949) + '=' +
                          str(epsilon) + ' (' + chr(947) + '=' + str(gamma) +
                          ')')
            explorer = chainerrl.explorers.ConstantEpsilonGreedy(
                    epsilon=epsilon, random_action_func=exploration_func)
        elif(exp_type in 'linear decay'):
            if(type(epsilon) != list):
                raise KeyboardInterrupt(
                    'Error while creating agent: Linear decay exploration does'
                    'not work with a constant epsilon')
            agent_info = ('NSQ10 - ' + 'Linear decay ' + chr(949) + '=' +
                          str(epsilon[0]) + '->' + str(epsilon[1]) + ' in ' +
                          str(epsilon[2]) + ' time steps' + ' (' + chr(947) +
                          '=' + str(gamma) + ')')
            explorer = chainerrl.explorers.LinearDecayEpsilonGreedy(
            start_epsilon=epsilon[0], end_epsilon=epsilon[1],
            decay_steps=epsilon[2], random_action_func=exploration_func)
        else: # If type doesn't match with any known one raise error
            raise KeyboardInterrupt(
                'Error while creating agent: Unknown exploration type')
        
        # Experience replay
        replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=10 ** 6)
        
        # Agent (NSQ)
        agent = chainerrl.agents.NSQ(q_func, opt, t_max=10, gamma=0.99,
        i_target=50000,explorer=explorer)
        
        return agent, agent_info, q_func, opt, explorer, replay_buffer      
    elif(alg in 'PPO'):
        
        # Optimizer
        opt = chainer.optimizers.Adam(eps=1e-2)
        
        model = A3CFFSoftmax(obs_size, n_actions)
        opt.setup(model)
        obs_normalizer = chainerrl.links.EmpiricalNormalization(
        obs_size, clip_threshold=5)
        
        # model.to_gpu(1)
        # obs_normalizer.to_gpu(1)


        # Exploration & agent info
        if(exp_type in 'constant'):
            agent_info = ('PPO ')
        elif(exp_type in 'linear decay'):
            if(type(epsilon) != list):
                raise KeyboardInterrupt(
                    'Error while creating agent: Linear decay exploration does'
                    'not work with a constant epsilon')
            agent_info = ('PPO - ' + 'Linear decay ' + chr(949) + '=' +
                          str(epsilon[0]) + '->' + str(epsilon[1]) + ' in ' +
                          str(epsilon[2]) + ' time steps' + ' (' + chr(947) +
                          '=' + str(gamma) + ')')
            explorer = chainerrl.explorers.LinearDecayEpsilonGreedy(
            start_epsilon=epsilon[0], end_epsilon=epsilon[1],
            decay_steps=epsilon[2], random_action_func=exploration_func)
        else: # If type doesn't match with any known one raise error
            raise KeyboardInterrupt(
                'Error while creating agent: Unknown exploration type')
        
        
        # Agent (DDQN)
        agent = chainerrl.agents.PPO(model = model, optimizer = opt, obs_normalizer=obs_normalizer, gpu=None, 
        gamma=0.995, lambd=0.95, value_func_coef=1.0, entropy_coef=0.01, update_interval=2048, minibatch_size=64, 
        epochs=10, clip_eps=0.2, clip_eps_vf=None, standardize_advantages=True, recurrent=False, max_recurrent_sequence_len=None, 
        act_deterministically=False, value_stats_window=1000, entropy_stats_window=1000, 
        value_loss_stats_window=100, policy_loss_stats_window=100)
        return agent, agent_info, model ,opt
    elif(alg in 'SARSA'):
        #   
        # Q function instanciation
        q_func = QFunction(obs_size, n_actions)
        
        # Optimizer
        opt = chainer.optimizers.Adam(eps=1e-2)
        opt.setup(q_func)
        
        # Exploration & agent info
        if(exp_type in 'constant'):
            agent_info = ('SARSA - ' + 'Constant ' + chr(949) + '=' +
                          str(epsilon) + ' (' + chr(947) + '=' + str(gamma) +
                          ')')
            explorer = chainerrl.explorers.ConstantEpsilonGreedy(
                    epsilon=epsilon, random_action_func=exploration_func)
        elif(exp_type in 'linear decay'):
            if(type(epsilon) != list):
                raise KeyboardInterrupt(
                    'Error while creating agent: Linear decay exploration does'
                    'not work with a constant epsilon')
            agent_info = ('SARSA - ' + 'Linear decay ' + chr(949) + '=' +
                          str(epsilon[0]) + '->' + str(epsilon[1]) + ' in ' +
                          str(epsilon[2]) + ' time steps' + ' (' + chr(947) +
                          '=' + str(gamma) + ')')
            explorer = chainerrl.explorers.LinearDecayEpsilonGreedy(
            start_epsilon=epsilon[0], end_epsilon=epsilon[1],
            decay_steps=epsilon[2], random_action_func=exploration_func)
        else: # If type doesn't match with any known one raise error
            raise KeyboardInterrupt(
                'Error while creating agent: Unknown exploration type')
        
        # Experience replay
        replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=10 ** 6)
        
        # Agent (SARSA)
        agent = chainerrl.agents.SARSA(
                q_func, opt, replay_buffer, gamma, explorer,
                replay_start_size=100000, target_update_interval=50000)
        
        return agent, agent_info, q_func, opt, explorer, replay_buffer
    elif(alg in 'SAC'):
        # Policy
        policy = chainerrl.policies.FCSoftmaxPolicy(
                obs_size, n_actions, n_hidden_channels=60, n_hidden_layers=1,
                last_wscale=0.01, nonlinearity=F.tanh)   
        # Q function instanciation
        q_func1 = QFunction(obs_size, n_actions)
        q_func2 = QFunction(obs_size, n_actions)
        # Optimizer
        opt1 = chainer.optimizers.Adam(eps=1e-2)
        opt2 = chainer.optimizers.Adam(eps=1e-2)
        opt3 = chainer.optimizers.Adam(eps=1e-2)
        opt1.setup(q_func1)
        opt2.setup(q_func2)
        opt3.setup(policy)
        # Exploration & agent info
        if(exp_type in 'constant'):
            agent_info = ('SAC - ' + 'Constant ' + chr(949) + '=' +
                          str(epsilon) + ' (' + chr(947) + '=' + str(gamma) +
                          ')')
            explorer = chainerrl.explorers.ConstantEpsilonGreedy(
                    epsilon=epsilon, random_action_func=exploration_func)
        elif(exp_type in 'linear decay'):
            if(type(epsilon) != list):
                raise KeyboardInterrupt(
                    'Error while creating agent: Linear decay exploration does'
                    'not work with a constant epsilon')
            agent_info = ('SAC - ' + 'Linear decay ' + chr(949) + '=' +
                          str(epsilon[0]) + '->' + str(epsilon[1]) + ' in ' +
                          str(epsilon[2]) + ' time steps' + ' (' + chr(947) +
                          '=' + str(gamma) + ')')
            explorer = chainerrl.explorers.LinearDecayEpsilonGreedy(
            start_epsilon=epsilon[0], end_epsilon=epsilon[1],
            decay_steps=epsilon[2], random_action_func=exploration_func)
        else: # If type doesn't match with any known one raise error
            raise KeyboardInterrupt(
                'Error while creating agent: Unknown exploration type')
        
        # Experience replay
        replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=10 ** 6)
        
        # Agent (SAC)
        agent = chainerrl.agents.SoftActorCritic(policy=policy, q_func1=q_func1, q_func2=q_func2, policy_optimizer= opt3, 
        q_func1_optimizer=opt1, q_func2_optimizer=opt2, replay_buffer=replay_buffer, gamma=gamma, replay_start_size=1000000, soft_update_tau=0.005,burnin_action_func=None)
        print(agent.policy_optimizer.t)
        return agent, agent_info, q_func1, opt1, explorer, replay_buffer    
    elif(alg in 'PAL'):

        # Q function instanciation
        q_func = QFunction(obs_size, n_actions)
        
        # Optimizer
        opt = chainer.optimizers.Adam(eps=1e-2)
        opt.setup(q_func)
        
        # Exploration & agent info
        if(exp_type in 'constant'):
            agent_info = ('PAL - ' + 'Constant ' + chr(949) + '=' +
                          str(epsilon) + ' (' + chr(947) + '=' + str(gamma) +
                          ')')
            explorer = chainerrl.explorers.ConstantEpsilonGreedy(
                    epsilon=epsilon, random_action_func=exploration_func)
        elif(exp_type in 'linear decay'):
            if(type(epsilon) != list):
                raise KeyboardInterrupt(
                    'Error while creating agent: Linear decay exploration does'
                    'not work with a constant epsilon')
            agent_info = ('PAL - ' + 'Linear decay ' + chr(949) + '=' +
                          str(epsilon[0]) + '->' + str(epsilon[1]) + ' in ' +
                          str(epsilon[2]) + ' time steps' + ' (' + chr(947) +
                          '=' + str(gamma) + ')')
            explorer = chainerrl.explorers.LinearDecayEpsilonGreedy(
            start_epsilon=epsilon[0], end_epsilon=epsilon[1],
            decay_steps=epsilon[2], random_action_func=exploration_func)
        else: # If type doesn't match with any known one raise error
            raise KeyboardInterrupt(
                'Error while creating agent: Unknown exploration type')
        
        # Experience replay
        replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=10 ** 6)
        
        # Agent (PAL)
        agent = chainerrl.agents.PAL(
                q_func, opt, replay_buffer, gamma, explorer,
                replay_start_size=100000, target_update_interval=50000)
        
        return agent, agent_info, q_func, opt, explorer, replay_buffer
    elif(alg in 'TRPO'):
        # Policy
        policy = chainerrl.policies.FCSoftmaxPolicy(
                obs_size, n_actions, n_hidden_channels=90, n_hidden_layers=2,
                last_wscale=0.01, nonlinearity=F.tanh)
        
        # Value function
        vf = chainerrl.v_functions.FCVFunction(
                obs_size, n_hidden_channels=60, n_hidden_layers=1,
                last_wscale=0.01, nonlinearity=F.tanh)
        
        # Optimizer
        opt = chainer.optimizers.Adam()
        opt.setup(vf)
        
        # Agent (TRPO)
        agent = chainerrl.agents.TRPO(
                policy=policy, vf=vf, vf_optimizer=opt, gamma=gamma,
                update_interval=50000)
        
        # Info
        agent_info = 'TRPO' + ' (' + chr(947) + '=' + str(gamma) + ')'
        
        return agent, agent_info, policy, opt

# Funtion that instances one or more agents with certain parameters
# make_training_agents 只会调用 create_agents，再通过create_agents  调用  create_agent
def create_agents(
        exp_type, epsilon, repetitions, gamma, obs_size, n_actions,
        exploration_func, alg):
    
    agents = []
    for i in range(repetitions):
        # Instance agent with its objects and append to list
        agents.append(create_agent(gamma, obs_size, n_actions,
                                   exploration_func, alg, exp_type, epsilon))
    
    # Return list of turples which contain agents with equal parameters
    return agents

# Funtion that creates instances of certain agents based on parameters
def make_training_agents(
        env, gammas, exp_types, epsilons, alg, repetitions=1):
    # 0.995   'const'    0.2    'DDQN','SARSA','TRPO'   1
    """
    This function returns a list of lists. Each of the lists represents a type
    of agent and contains as many as repetitions declares (default is 1). Each
    of the internal lists contains turples that each contain all necesary
    objects for an agent to be trained.
    In the turples index 0 is the agent and index 1 is the info, the other
    indexes don't need to be accesed.
    Parameters:
        env: gym environment
        gammas: Discount factors to be used; you may define only one
        explorators: Exploration types to be used; you may define only one per
                     agent (constant, linear)
        epsilons: Epsilon values to be used; you may define only one per agent
        repetitions: How many replicas of a type of agent are to be created
                     (default is 1)
        alg: The algorithm of the agent; some of the other parameters don't
             matter depending on what algorithm is picked (DDQN, TRPO, SARSA,
             PAL) 
        NOTE 1: If you define multiple values for a parameter, you can only
                define one of the other if it is to be the same for all agent
                types
                Example 1: gammas = 0.7, alg = 'DDQN', explorators = 'const',
                           epsilons = [0.1, 0.2], repetitions = [2, 3]
                Example 2: gammas = [0.1, 0.5], alg = ['DDQN', 'SARSA'],
                           explorators = ['const', 'linear'],
                           epsilons = [0.2, [0.4,0.05,5000]], repetitions = 3
                Multiple value parameters are gammas, explorators, epsilons and
                repetitions
        NOTE 2: For algorithms with policy instead of a Q-function, explorators
                and epsilons parameters are unused (but need a value!)
                Algorithms with policy include: TRPO
    """
    
    # If there are multiple algorithms process is repeated for each one
    # alg = 是单参数   
    if(type(alg) != list and type(alg) == str):
        alg = [alg]
    elif(type(alg) != list and type(alg) != str):
        raise KeyboardInterrupt(
            'Error while creating agents: Incorrect parameter types')
    # alg 是多参数
    agents = [] # List that contains lists of turples
    for a in range(len(alg)):
        # 对每一个参数循环处理
        # Check and save unused parameters for current algorithm
        # 在TRPO这种情况中 explorators  and epsilons parameters  用不着，也就是说只需要一个gammas和repetitions
        # 正常单值的情况下会为TRPO 创建一个agent
        if(alg[a] in 'TRPO'):
            exp_types_save = exp_types
            exp_types = 'const' # Any valid value
            epsilons_save = epsilons
            epsilons = 0.2 # Any valid value
            print('Warning: Not using exp_types and epsilons parameters for ' +
                  alg[a])
        
        # Error handling 
        # 除了env和arg一共有四个参数  如有有任一为none
        if(exp_types == None or epsilons == None or gammas == None):
            raise KeyboardInterrupt(
                'Error while creating agents: Missing required arguments')
        # 如果这四个参数有任一参数不符合数据规则
        if((type(exp_types) != str and type(exp_types) != list) or
           (type(epsilons) != float and type(epsilons) != list and
            type(epsilons) != int) or
           (type(repetitions) != int and type(repetitions) != list) or
           (type(gammas) != float and type(gammas) != list and
            type(gammas) != int)):
               raise KeyboardInterrupt(
                   'Error while creating agents: Incorrect parameter types')
        
        # Define some parameters from the environment
        obs_size = env.observation_space.shape[0]    # 22
        n_actions = env.action_space.n              # 4
        exploration_func = env.action_space.sample  # <bound method Discrete.sample of Discrete(4)>
        # print("mark")
        # print(obs_size)
        # print("n/_actions")
        # print(n_actions)
        # print(exploration_func)

        # Parameter check   only one agent
        # 其他三个参数都是单个值，但repetitions是一个数组，抛出错误
        # 不抛出错误意味着 所有的参数都是单值
        # 那么 agents 追加一个 create_agents结果 参数4+3+1（四个定义的参数，三个环境相关的参数，一个arg参数）
        if(type(exp_types) == str and type(epsilons) == float and
           (type(gammas) == float or type(gammas) == int)):
            # One type of agent
            # Error handling
            if(type(repetitions) == list):
                raise KeyboardInterrupt(
                    'Error while creating agents: Repetitions parameter was '
                    'expected to be an integer')
            # Agent instanciation
            agents.append(create_agents(
                exp_types, epsilons, repetitions, gammas, obs_size, n_actions,
                exploration_func, alg[a]))
        # 如果exp_types/epsilons/gammas 有是数组的，
        # 判断一下该数组的长度和其他三个参数的长度是否一致（仅检查数组的情况），不一致报错
        # 否则（不报错）（其他参数都是数组或者其他参数与这个参数的长度是一样的）：把哪些不是数组的参数扩展到和该参数一样的长度
        # 再给agents追加 追加一个 create_agents结果 参数4+3+1（四个定义的参数数组，三个环境相关的参数，一个arg参数）
        elif(type(exp_types) == list):
            # Multiple types of agent
            agent_num = len(exp_types)
            
            # Error handling
            if(type(epsilons) == list and len(epsilons) != agent_num):
                raise KeyboardInterrupt(
                    'Error while creating agents: Number of exploration types '
                    'doesn´t match number of epsilons\n'
                    'TIP: You may pass just one value of epsilon')
            if(type(repetitions) == list and len(repetitions) != agent_num):
                raise KeyboardInterrupt(
                    'Error while creating agents: Number of exploration types '
                    'doesn´t match number of repetitions\n'
                    'TIP: You may pass just one value of repetitions')
            if(type(gammas) == list and len(gammas) != agent_num):
                raise KeyboardInterrupt(
                    'Error while creating agents: Number of exploration types '
                    'doesn´t match number of gammas\n'
                    'TIP: You may pass just one value of gamma')
            
            # Agent instanciation
            
            # Create a list of just one element in case there is only one, so
            # code in the loop can be simplified (epsilons[i], repetitions[i],
            # gammas[i])
            if(type(epsilons) == float):
                temp = []
                for i in range(agent_num):
                    temp.append(epsilons)
                epsilons = temp
            if(type(repetitions) == int):
                temp = []
                for i in range(agent_num):
                    temp.append(repetitions)
                repetitions = temp
            if(type(gammas) == float or type(gammas) == int):
                temp = []
                for i in range(agent_num):
                    temp.append(gammas)
                gammas = temp
            
            """
            To create various agent and compare them, many of the objects that
            the agent requires have to be instanced multiple times. If not,
            they will share instances of certain objects and training will be
            biased.
            """
            for i in range(agent_num):
                """
                Create one list of turples, each turple contains data on an
                instance of an agent. The list contains agents with equal
                parameters.
                """
                agents.append(create_agents(
                    exp_types[i], epsilons[i], repetitions[i], gammas[i],
                    obs_size, n_actions, exploration_func, alg[a]))
        
        elif(type(epsilons) == list):
            # Multiple types of agent
            agent_num = len(epsilons)
            
            # Error handling
            if(type(exp_types) == list and len(exp_types) != agent_num):
                raise KeyboardInterrupt(
                    'Error while creating agents: Number of exploration types '
                    'doesn´t match number of epsilons\n'
                    'TIP: You may pass just one value of epsilon')
            if(type(repetitions) == list and len(repetitions) != agent_num):
                raise KeyboardInterrupt(
                    'Error while creating agents: Number of exploration types '
                    'doesn´t match number of repetitions\n'
                    'TIP: You may pass just one value of repetitions')
            if(type(gammas) == list and len(gammas) != agent_num):
                raise KeyboardInterrupt(
                    'Error while creating agents: Number of exploration types '
                    'doesn´t match number of gammas\n'
                    'TIP: You may pass just one value of gamma')
            
            # Agent instanciation
            
            # Create a list of just one element in case there is only one, so
            # code in the loop can be simplified (exp_types[i], repetitions[i],
            # gammas[i])
            if(type(exp_types) == str):
                temp = []
                for i in range(agent_num):
                    temp.append(exp_types)
                exp_types = temp
            if(type(repetitions) == int):
                temp = []
                for i in range(agent_num):
                    temp.append(repetitions)
                repetitions = temp
            if(type(gammas) == float or type(gammas) == int):
                temp = []
                for i in range(agent_num):
                    temp.append(gammas)
                gammas = temp
            
            """
            To create various agent and compare them, many of the objects that
            the agent requires have to be instanced multiple times. If not,
            they will share instances of certain objects and training will be
            biased.
            """
            for i in range(agent_num):
                """
                Create one list of turples, each turple contains data on an
                instance of an agent. The list contains agents with equal
                parameters.
                """
                agents.append(create_agents(
                    exp_types[i], epsilons[i], repetitions[i], gammas[i],
                    obs_size, n_actions, exploration_func, alg[a]))
        
        elif(type(gammas) == list):
            # Multiple types of agent
            agent_num = len(gammas)
            
            # Error handling
            if(type(epsilons) == list and len(epsilons) != agent_num):
                raise KeyboardInterrupt(
                    'Error while creating agents: Number of exploration types '
                    'doesn´t match number of epsilons\n'
                    'TIP: You may pass just one value of epsilon')
            if(type(exp_types) == list and len(exp_types) != agent_num):
                raise KeyboardInterrupt(
                    'Error while creating agents: Number of exploration types '
                    'doesn´t match number of epsilons\n'
                    'TIP: You may pass just one value of epsilon')
            if(type(repetitions) == list and len(repetitions) != agent_num):
                raise KeyboardInterrupt(
                    'Error while creating agents: Number of exploration types '
                    'doesn´t match number of repetitions\n'
                    'TIP: You may pass just one value of repetitions')
            
            # Agent instanciation
            
            # Create a list of just one element in case there is only one, so
            # code in the loop can be simplified (epsilons[i], exp_types[i],
            # repetitions[i])
            if(type(epsilons) == float):
                temp = []
                for i in range(agent_num):
                    temp.append(epsilons)
                epsilons = temp
            if(type(exp_types) == str):
                temp = []
                for i in range(agent_num):
                    temp.append(exp_types)
                exp_types = temp
            if(type(repetitions) == int):
                temp = []
                for i in range(agent_num):
                    temp.append(repetitions)
                repetitions = temp
            
            """
            To create various agent and compare them, many of the objects that
            the agent requires have to be instanced multiple times. If not,
            they will share instances of certain objects and training will be
            biased.
            """
            for i in range(agent_num):
                """
                Create one list of turples, each turple contains data on an
                instance of an agent. The list contains agents with equal
                parameters.
                """
                agents.append(create_agents(
                    exp_types[i], epsilons[i], repetitions[i], gammas[i],
                    obs_size, n_actions, exploration_func, alg[a],env))
        
        else: # Error handling for other cases
            raise KeyboardInterrupt(
                'Error: Unexpected parameter types or values')
        
        # Reassing unused parameters for next algorithm
        if(alg[a] in 'TRPO'):
            exp_types = exp_types_save
            epsilons = epsilons_save
    
    return agents