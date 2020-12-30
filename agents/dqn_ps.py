# -*- coding: utf-8 -*-

import random
import numpy as np
from collections import deque
from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

_INTERACTION = namedtuple('Interaction', ('percept', 'action', 'reward'))


########################################################################
#       DQN
# ---------------
#
# Defines a feedforward DNN that represents a DQN approximating merit values.
#
#


class DQN(nn.Module):
    """
    A feedforward NN implementing a DQN.

    Parameters:
        dim_percept:    int
                        the dimension of the percept space
        num_actions:    int
                        the size of the action space, i.e. #output neurons
        dim_hidden_:    list of int
                        the dimensions of hidden layers
        dropout_rate_   list of float
                        the probability of a hidden neuron to be dropped in each hidden layer
        norm:           bool
                        if set to True, percept units are normalized
    """
    def __init__(self, dim_percept, num_actions, dim_hidden_=[32], dropout_rate_=[0.]):
        super(DQN, self).__init__()
        #initializing all the weights
        init_kaiman = True

        #input layer
        self.input = nn.Linear(dim_percept, dim_hidden_[0])
        self.input.apply(self._init_weights) if init_kaiman else None

        #hidden layers
        self.layers = nn.ModuleList()
        for l in range(len(dim_hidden_)-1):
            self.layers.append(nn.Linear(dim_hidden_[l], dim_hidden_[l+1]))

        for l in range(len(dim_hidden_)-1):
            self.layers[l].apply(self._init_weights) if init_kaiman else None

        #output layer
        self.output = nn.Linear(dim_hidden_[-1], num_actions)


        #dropout layers
        self.dropout = nn.ModuleList()
        for l in range(len(dropout_rate_)):
            self.dropout.append(nn.Dropout(p=dropout_rate_[l]))


    def _init_weights(self, layer):
        """
        Initializes weights with kaiming_uniform method for ReLu from PyTorch.
        """
        if type(layer) == nn.Linear:

            nn.init.kaiming_uniform_(layer.weight, a=0, mode='fan_in', nonlinearity='relu')

    def forward(self, percept):

        out = F.relu(self.input(percept))
        for l in range(len(self.layers)):
            out = self.dropout[l](out)
            out = F.relu(self.layers[l](out))
        out = self.dropout[-1](out)

        out = self.output(out)
        return out


#########################################################
#   DQN Agent
# ----------------
#
# Defines the agent that interacts with the environment.
#
#

class DQNAgent():
    """
    The agent that uses a DQN (represented by a feedforward NN) to predict  merit values.
    Parameters:
        dim_percept     int
                        the dimension of the percept space, i.e. #percept neurons
        num_actions:    int
                        the size of the action space, i.e. #output neurons
        dim_hidden:     list of int
                        the dimensions of hidden layers
        dropout_rate:   list of float
                        the probability of a hidden neuron to be dropped in each layer
        target_update:  int
                        the number of trials that have to be passed before the
                        policy network is copied to the target network
        device:         str
                        the device on which the DQN is run, either "cpu" or "cuda"
        learning_rate:  float
                        learning rate >=0. for optimizer
        capacity:       int
                        the maximum size of the memory on which the DQN is trained
        gamma:           float
                        the gamma parameter
        batch_size:     int
                        the size >1 of the batches sampled from the memory after each
                        trial to train the DQN
        replay_time:    int
                        number of interactions with the environment before experience replay
                        is invoked
    """
    def __init__(self, dim_percept, num_actions, dim_hidden=[32], dropout_rate=[0.], target_update=50, device="cpu", learning_rate=0.01,
                 capacity=1000, gamma=0.9, batch_size=100, replay_time=10, off_policy = False):
        #ERRORS
        if batch_size <= 1:
            raise ValueError("Invalid batch size: batch_size={}.".format(batch_size))
        if device != "cpu":
            raise NotImplementedError("GPU usage is not supported yet: device=\"{}\".".format(device)
                                     )
        if learning_rate < 0.:
            raise ValueError("Invalid learning rate: learning_rate={}.".format(learning_rate))
        if len(dim_hidden) != len(dropout_rate):
            raise ValueError("Number of hidden layers \"{}\" should equal".format(len(dim_hidden))+\
                             " number of dropout layers \"{}\".".format(len(dropout_rate))
                            )
        if gamma > 1. or gamma < 0.:
            raise ValueError("Invalid gamma value: gamma=\"{}\".".format(gamma))
        ##############

        #the policy network for choosing actions
        train_output_weights = False
        if off_policy:
            train_output_weights = True
        self.policy_net = DQN(dim_percept, num_actions, dim_hidden_=dim_hidden,
                              dropout_rate_=dropout_rate).to(device)

        #PARAMETERS
        #parameters as in class description
        self.num_actions = num_actions
        self.target_update = target_update
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_count = 0
        self.replay_time = replay_time
        self.gamma_ps = 1
        self.off_policy = off_policy

        #INTERNAL VARS
        #the optimizer for the policy network
        self._optimizer = optim.Adam(filter(lambda p: p.requires_grad, 
                                            self.policy_net.parameters()),
                                     lr=learning_rate, amsgrad=True#, weight_decay=1e-5
                                    )
        #the target network for learning
        self._target_net = DQN(dim_percept, num_actions, dim_hidden_=dim_hidden,
                               dropout_rate_=dropout_rate).to(device)
        self._target_net.load_state_dict(self.policy_net.state_dict())
        #the memory of the agent
        self._memory = deque(maxlen=capacity)
        if self.off_policy:
            #the percepts and actions of the current trial
            self._trial_data = deque()
            #the rewards of the current trial
            self._trial_rewards = torch.empty(0)
        #previous percept
        self._prev_s = None
        #previous action
        self._prev_a = None
        if not(self.off_policy):
            #save the current episode in buffer_memory before calculating the accumulated reward
            self._buffer_memory = deque(maxlen=2*replay_time)
        #number of interactions
        self._num_interactions = 0


    def deliberate_and_learn(self, percept, action, reward, eta_e, done, softmax_e=1):
        """
        This is the primary function that represents the agent's reaction to an interaction with
        the environment.
        Following an interaction with the environment, the agent receives a percept (representing
        the current state of the environment) and a reward.
        (1) The reward is saved alongside the previous percept and preceding action to the trial
            data.
        (2) At the end of a trial, the accumulated data from the trial is saved to the agent's
            memory and the agent is trained. In addition, the agent is trained whenever
            X (=replay_time) number of interactions have passed since the last training.
            The network receives a tensor of torch.size([n+m]) where
            n=#percept neurons and m=#action neurons (ideally in one-hot encoding)
        (3) During a trial, the agent chooses and returns an action in response to the current
            percept.
            This action should be in a one-hot encoding.
        Parameters:
            percept:    torch.Tensor
                        the current percept issued by the environment as a tensor
                        of torch.size([1,n]) which should be rescaled such that
                        components are real values in [0,1] for optimal performance
                        of the DQN
            reward:     float
                        the current reward >1.e-20 issued by the environment
            done:       bool
                        True if the trial has ended, False otherwise
        Returns:
            action:     torch.Tensor
                        the action that is to be performed as a tensor of torch.size([1,m]),
                        ideally in one-hot encoding
        """

        #(1)
        if self._prev_s is not None:
            self._save_data(reward, eta_e)

        #(2)
        self._num_interactions += 1
        if self._num_interactions % self.replay_time == 0:
            self._experience_replay()

        if done:
            self._save_memory()
            self._experience_replay()

            self._prev_a = None
            self._prev_s = None

            return None


        self._prev_a = torch.LongTensor([np.array([action])])
        self._prev_s = percept

        return None

    def deliberate(self, percept, softmax_e):
        """
        This is the primary function that represents the agent's choice of an action.
         During a trial, the agent chooses and returns an action in response to the current
            percept.
            This action should be in a one-hot encoding.

        Parameters:
            percept:    torch.Tensor
                        the current percept issued by the environment as a tensor
                        of torch.size([1,n]) which should be rescaled such that
                        components are real values in [0,1] for optimal performance
                        of the DQN

        """

        action = self._choose_action(percept, softmax_e)

        return action

    def learn(self, percept, action, reward, done):
        """
        Following is the learning process of the agent:
        (1) The reward is saved alongside the previous percept and preceding action to the trial
            data.
        (2) At the end of a trial, the accumulated data from the trial is saved to the agent's
            memory and the agent is trained. In addition, the agent is trained whenever
            X (=replay_time) number of interactions have passed since the last training.
            The network receives a tensor of torch.size([n+m]) where
            n=#percept neurons and m=#action neurons (ideally in one-hot encoding)

        Parameters:

            percept:    torch.Tensor
                        the current percept issued by the environment as a tensor
                        of torch.size([1,n]) which should be rescaled such that
                        components are real values in [0,1] for optimal performance
                        of the DQN
            reward:     float
                        the current reward >1.e-20 issued by the environment
            done:       bool
                        True if the trial has ended, False otherwise

        Returns:

            action:     torch.Tensor
                        the action that is to be performed as a tensor of torch.size([1,m]),
                        ideally in one-hot encoding

        """
        self._prev_a = torch.LongTensor([action])
        self._prev_s = percept

        self._num_interactions += 1

        if self._prev_a is not None:
            self._save_buffer_data(reward)

        #check if enough new entries are in the buffer_memory before calculating glow and adding elements to the memory:
        if self._num_interactions % self.replay_time and len(self._buffer_memory) >= 2*self.replay_time:
            self._save_memory()
            self._experience_replay()

        if done:
            self._prev_a = None
            self._prev_s = None
            self._num_interactions = 0.

        return None

    def _save_buffer_data(self, reward):
        """
        saves, state, action and reward in the buffer memory
        Parameters:
             reward: reward of current epoisode

        Returns: None
        """
        data = (self._prev_s, self._prev_a, reward)
        self._buffer_memory.append(data)
        return None

    def reward_glow(self, n):
        """
        Caclulates the contirbutions of glow from future interactions and adds them to the memory so that the network can be trained on
        these episodes

        Parameters:
            n: integer set to n and then it calculates all the accumulated reward for the entries in n to self.replay_time
               in the self._buffer_memory

        Returns:
            r: (float) accumulated reward

        """
        if n == self.replay_time:
            accumulated_reward = 0.
            for i in range(self.replay_time):
                glow_contribution = self._buffer_memory[self.replay_time + i][2] * (self.gamma) ** i
                if glow_contribution > 10**(-20):
                    accumulated_reward += glow_contribution
            r = accumulated_reward
            reward_tensor = torch.Tensor([[r]])
            data = _INTERACTION(self._buffer_memory[n][0], self._buffer_memory[n][1], reward_tensor)
            self._memory.append(data)
            return r
        else:
            r = self._buffer_memory[n][2] + self.gamma * self.reward_glow(n + 1)
            reward_tensor = torch.Tensor([[r]])
            data = _INTERACTION(self._buffer_memory[n][0], self._buffer_memory[n][1], reward_tensor)
            self._memory.append(data)
            return r


    def _save_data(self, reward, eta_e):
        """
        Saves data of the current instance of the agent-environment interaction.
        The data is distributed as follows.
            (previous percept, previous action) --> _trial_data
            reward --> _trial_rewards
        gamma is taken into account for the reward.
        Parameters:
            reward: float
                    the current reward issued by the environment
        Returns:
            None
        """

        data = (self._prev_s, self._prev_a)
        self._trial_data.append(data)
        r_val = reward
        if reward != 0:
            for i in range(len(self._trial_rewards)-1, -1, -1):
                r_val = r_val*eta_e
                if r_val < 1.e-20:
                    break
                self._trial_rewards[i] += r_val
        self._trial_rewards = torch.cat((self._trial_rewards, torch.Tensor([[reward]])))

        return None


    def _save_memory(self):
        """
        Saves the data of the current trial to the agent's memory from which it is trained.
        The memory is structured as follows, (percept, action, reward).
        The _trial_data is then emptied.
        Parameters:
            None
        Returns:
            None
        """

        global _INTERACTION
        if self.off_policy:
            for i, data in enumerate(self._trial_data):
                data = _INTERACTION(data[0], data[1], self._trial_rewards[i])
                self._memory.append(data)

            self._trial_data = deque()
            self._trial_rewards = torch.empty(0)

        else:
            self.reward_glow(0)

        return None


    def _experience_replay(self):
        """
        Trains the DQN on batches of the form (s, a, r, s , a) sampled from its memory.
        (1) The policy network is copied to the target network after experience
            replay has been invoked X (=target_update) times since the last update.
        (2) A batch is sampled from the memory. The s, a, r, s and a are batched together
            so all can be trained together.
        (3) The merit values M(s,a) are predicted by the policy and target network.
        (4) The policy network is updated to predict M(s,a)+r
            (where r is already taking gamma into account).
            Therefore,
            (4.1) the loss is calculated and
            (4.2) the model is optimized with an Adam optimizer.
        Parameters:
            None
        Returns:
            None
        """

        if len(self._memory) < self.batch_size:
            return None

        #(1)
        self.target_count += 1
        if self.target_count % self.target_update == 0:
            self._target_net.load_state_dict(self.policy_net.state_dict())

        #(2)
        global _INTERACTION

        batch = random.sample(self._memory, self.batch_size)
        batch = _INTERACTION(*zip(*batch))

        batch_percept = torch.cat(batch.percept)
        batch_action = torch.cat(batch.action)
        batch_reward = torch.cat(batch.reward)
        batch_reward = batch_reward.resize_(len(batch_reward), 1)

        #(3)
        m_values = self.policy_net(batch_percept).gather(1, batch_action)
        target_m_values = self._target_net(batch_percept).gather(1, batch_action).detach()

        #(4)
        #(4.1)
        loss =  F.mse_loss(m_values, (target_m_values+batch_reward-self.gamma_ps*m_values.detach()))
        #(4.2)
        self._optimizer.zero_grad()
        loss.backward()

        # #Gradient clipping:
        # for param in self.policy_net.parameters():
        #     if not param.grad is None:
        #         param.grad.data.clamp_(-100, 100)

        self._optimizer.step()

        return None

    def _choose_action(self, percept, softmax_e):
        """
        Chooses an action according to the policy networks h-value predictions.

        Parameters:
            percept:    torch.Tensor
                        the current percept issued by the environment

        Returns:
            action:     torch.Tensor
                        the action that is to be performed as a tensor of torch.size([m])
        """
        with torch.no_grad():
            m_values = self.policy_net(percept[0]).numpy()
        rescale = np.amax(m_values)
        m_values = np.exp(softmax_e*(m_values-rescale))

        m_values = m_values/np.sum(m_values)

        action = np.random.choice(range(self.num_actions), p=m_values)
        action = np.array([action])
        return action