import torch
import gym
import os
import sys
from sys import argv
import numpy as np
import pathlib
import argparse
import pickle

sys.path.insert(0, 'agents')

from debn_ps import *

'''
Usage: run command line of the form `python run_debn_cartpole.py --hidden_layers (int) --hidden_units (int) --learning_rate (float) --target_update (int) --batch_size (int) --agent_number (int)` 
e.g. `python run_debn_cartpole.py --hidden_layers 1 --hidden_units 73 --learning_rate 0.01 --target_update 5000 --batch_size 200 --agent_number 0` 
     `python run_debn_cartpole.py --hidden_layers 2 --hidden_units 19 --learning_rate 0.01 --target_update 5000 --batch_size 200 --agent_number 0`
     `python run_debn_cartpole.py --hidden_layers 5 --hidden_units 10 --learning_rate 0.001 --target_update 10000 --batch_size 100 --agent_number 0` 
Run code with --agent_number n for n in {0,..,50} to gather the statistics for 50 different agents 
'''

# Allow only one thread/CPU for pytorch (no gain by multiprocessing)
torch.set_num_threads(1)

#####################################################################################
def get_args(argv):
    """
    Passes the arguments from the runfile to the main file or passes the default values to the main file.
    These arguments define the filename under which the data is saved.
    Returns a namespace object that makes each element callable by args.name.
    Args:
        argv  (list) list of arguments that are passed through with a --name
    Returns:
        args  (namespace) namespace of all arguments passed through
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=2000, help='number of episodes')
    parser.add_argument('--hidden_layers', type=int, default=2, help='number of hidden layers')
    parser.add_argument('--hidden_units', type=int, default=19, help='number of hidden units in each hidden layer')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate of the optimizer')
    parser.add_argument('--target_update', type=int, default=5000, help='update interval of the target network')
    parser.add_argument('--batch_size', type=int, default=200, help='size of the training batch for experience replay')
    parser.add_argument('--agent_number', type=int, default=1, help='index for the agent to gather statistics')

    args = parser.parse_args(argv)
    return args

args = get_args(argv[1:])

# INTERACTION PARAMETERS
EPISODES = args.episodes  # number of epochs/episodes

# AGENT PARAMETERS
AGENT_NUMBER = args.agent_number
DEVICE = 'cpu'
AGENT_NAME = 'debn'

GAMMA = 0.99 # discount factor
hidden_layers = args.hidden_layers # number of hidden layer
hidden_units_layer = args.hidden_units # number of hidden units
NUM_HIDDEN = [int(round(hidden_units_layer))]*hidden_layers # list of hidden unit numbers list
DROPOUT = [0.]*hidden_layers # dropout rate list
LEARNING_RATE = args.learning_rate # learning rate
CAPACITY = 2000 # size of memory
BATCH_SIZE = args.batch_size # size of the training batch for experience replay
REPLAY_TIME = 5 # the time interval between each experience replay
TARGET_UPDATE = args.target_update # update interval for the target network
BETA = 0.01 # beta parameter for softmax schedule
beta = np.tanh(np.linspace(BETA, 10, EPISODES)) # softmax schedule
SAVE_MODEL = False # set to true to save state dict

# ENVIRONMENT PARAMETERS
ENV_NAME = 'CartPole-v1' # environment name
env = gym.make(ENV_NAME) # generate environment
percept_size = env.observation_space.shape[0] # size of the percept space
action_size = env.action_space.n # size of the action space

# action encoding
all_actions = torch.empty(0)
for i in range(action_size):
    a = torch.zeros((1, 1, action_size))
    a = a.new_full((1, 1, action_size), 0.)
    a[0, 0, i] = 1.
    all_actions = torch.cat((all_actions, a))


# DEFINE FILENAME TO SAVE RESULTS
foldername = 'results/'
pathlib.Path(foldername).mkdir(parents=True, exist_ok=True)
filename = ENV_NAME+'_'+AGENT_NAME+'_E='+str(EPISODES)+'_HL='+str(hidden_layers)+'_HU='+str(hidden_units_layer)+'_LR='+str(LEARNING_RATE)+'_TU='+str(TARGET_UPDATE)+'_BS='+str(BATCH_SIZE)+'_'+str(AGENT_NUMBER)
print(filename)

#####################################################################################
if __name__ == "__main__":
    agent = DEBNAgent(percept_size, action_size, all_actions, dim_hidden=NUM_HIDDEN, dropout_rate=DROPOUT,
                     device = DEVICE, learning_rate=LEARNING_RATE, capacity=CAPACITY, batch_size=BATCH_SIZE, replay_time=REPLAY_TIME,
                     target_update = TARGET_UPDATE, gamma = GAMMA)

    # list to save data:
    timesteps = []

    for e in range(EPISODES):
        #reset the environment
        percept = env.reset()
        percept = np.reshape(percept, [1, percept_size])
        percept = torch.Tensor(percept)
        reward = 0.
        done = False
        t=0

        while not done:
            t += 1
            action = agent.deliberate_and_learn(percept, reward, GAMMA, beta[e], done)
            action = (action[0] == 1).nonzero().item()
            percept, reward, done, _ = env.step(action)
            percept = np.reshape(percept, [1, percept_size])
            percept = torch.Tensor(percept)
            if done:
                timesteps.append(t)
                agent.deliberate_and_learn(percept, reward, GAMMA, beta[e], done)


        if e%1 == 0:
            print("Last score (timesteps per episode) the agent achieved at " + str(e) + ": ", np.mean(timesteps[-1:]))
            # save data to file
            pickle.dump(timesteps, open(foldername+filename, "wb"))

    # Save model
    if SAVE_MODEL:
        torch.save(agent.policy_net.state_dict(), 'saved_models/state_dict_' + filename + '.pckl')

env.close()