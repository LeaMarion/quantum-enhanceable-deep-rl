import torch
import gym
import os
import sys
import numpy as np
import pathlib
from sys import argv
import argparse
import pickle


sys.path.insert(0, 'agents')
sys.path.insert(0, 'environments')

from env_gridworld import *
from debn_ps import *

'''
Usage: run command line of the form `python run_debn_gridworld.py --agent_number (int)`
e.g. `python run_debn_gridworld.py --agent_number 0` 
Run code with --agent_number n for n in {0,..,50} to gather data for 50 different agents 
'''

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
    parser.add_argument('--beta_f', type=int, default=0.8, help='number of episodes')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate of the optimizer')
    parser.add_argument('--target_update', type=int, default=100, help='update interval of the target network')
    parser.add_argument('--schedule', type=str, default='htan', help='interval between each experience replay')
    parser.add_argument('--agent_number', type=int, default=0, help='index for the agent to gather statistics')

    args = parser.parse_args(argv)
    return args

args = get_args(argv[1:])


# INTERACTION PARAMETERS
EPISODES = 1200  # number of episodes for each agent 2000
MAX_STEPS_PER_TRIAL = 10  # number of allowed timesteps before reset 20000

#agent parameters
AGENT_NUMBER = args.agent_number
DEVICE = 'cpu'
AGENT_NAME = 'ebm'

GAMMA = 0.99 # discount factor
hidden_layers = 1 # number of hidden layers
hidden_units_layer = 64 # number of hidden units
NUM_HIDDEN = [int(round(hidden_units_layer))]*hidden_layers # list of hidden unit numbers list
DROPOUT = [0.]*hidden_layers # dropout rate list
LEARNING_RATE = args.learning_rate # learning rate
CAPACITY = 5000 # size of the memory
BATCH_SIZE = 100 # size of the training batch for experience replay
REPLAY_TIME = 100 # the time interval between each experience replay
TARGET_UPDATE = args.target_update # update interval for the target network
SAVE_MODEL = False #set to true to save state dict

BETA_i = 0.001 # initial beta parameter for schedule
BETA_f = args.beta_f # final beta parameter for schedule
SCHEDULE = args.schedule # name of the schedule
if SCHEDULE == 'htan':
    beta = np.tanh(np.linspace(BETA_i, BETA_f, EPISODES)) # tanh schedule
elif SCHEDULE =='lin':
    beta = np.linspace(BETA_i, BETA_f, EPISODES) # linear scchedule

# ENVIRONMENT PARAMETERS
ENV_NAME = 'gridworld' # environment name
DIMENSIONS = [100,100] # 2D grid of size [100,100]
env = TaskEnvironment(DIMENSIONS) #generate environment
percept_size = DIMENSIONS[0]+DIMENSIONS[1] #size of the percept space
action_size = 4 # size of the action space

#action encoding
all_actions = torch.empty(0)
for i in range(action_size):
    a = torch.zeros((1, 1, action_size))
    a = a.new_full((1, 1, action_size), 0.)
    a[0, 0, i] = 1.
    all_actions = torch.cat((all_actions, a))

#percept encoding
def to_two_hot(percept, dim):
    """
    Two-hot encodes the 2D percept of positions.
    """
    one_hot = np.zeros(dim[0]+dim[1])
    one_hot[percept[0]] = 1
    one_hot[dim[0]+percept[1]] = 1
    return one_hot

# DEFINE FILENAME TO SAVE RESULTS
foldername = 'results/'
pathlib.Path(foldername).mkdir(parents=True, exist_ok=True)
filename = ENV_NAME+'_'+AGENT_NAME+'_Bf='+str(BETA_f)+'_LR='+str(LEARNING_RATE)+'_TU='+str(TARGET_UPDATE)+'_S='+SCHEDULE+'_'+str(AGENT_NUMBER)
print(filename)

if __name__ == "__main__":
    agent = DEBNAgent(percept_size, action_size, all_actions, dim_hidden=NUM_HIDDEN, dropout_rate=DROPOUT,
                     device = DEVICE, learning_rate=LEARNING_RATE, capacity=CAPACITY, batch_size=BATCH_SIZE, replay_time=REPLAY_TIME,
                     target_update = TARGET_UPDATE, gamma = GAMMA)

    timesteps = []
    for e in range(EPISODES):
        counter = 0
        #reset the environment
        percept = env.reset()
        percept = to_two_hot(percept,DIMENSIONS)
        percept = np.reshape(percept, [1, percept_size])
        percept = torch.Tensor(percept)
        reward = 0.
        done = False
        for t in range(1, MAX_STEPS_PER_TRIAL + 1):
            action = agent.deliberate_and_learn(percept, reward, GAMMA, beta[e], done)
            action = (action[0] == 1).nonzero().item()
            percept, reward, done, _ = env.step(action)
            percept = to_two_hot(percept, DIMENSIONS)
            percept = np.reshape(percept, [1, percept_size])
            percept = torch.Tensor(percept)
            if t==MAX_STEPS_PER_TRIAL:
                reward = -1
                done = True
            if done:
                agent.deliberate_and_learn(percept, reward, GAMMA, beta[e], done)
                timesteps.append(t)
                break

        if e%100 == 0:
            print("Average last 100 scores (timesteps per episode) the agent achieved at " + str(e) + ": ", np.mean(timesteps[-100:]))
            # save data to file
            pickle.dump(timesteps, open(foldername+filename, "wb"))

    # Save model
    if SAVE_MODEL:
        torch.save(agent.policy_net.state_dict(), 'saved_models/state_dict_'+filename+'.pckl')
