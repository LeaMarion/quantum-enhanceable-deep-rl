import os
import sys
from sys import argv
import pathlib
import argparse
import pickle

sys.path.insert(0, 'environments')
sys.path.insert(0, 'agents')

import numpy as np
from env_gridworld import *
from flexible_ps import *

'''
Usage: run command line of the form `python run_tabular_gridworld.py --agent_number (int)`
e.g. `python run_tabular_gridworld.py --agent_number 0`
Run code with --agent_number n for n in {0,..,50} to gather the statistics for 50 different agents 
'''

#####################################################################################
def get_args(argv):
    """
    Passes the arguments from the runfile to the main file or passes the default values to the main file.
    The given argument defines the filename under which the data is saved.
    Returns a namespace object that makes each element callable by args.name.
    Args:
        argv  (list) list of arguments that are passed through with a --name

    Returns:
        args  (namespace) namespace of all arguments passed through
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent_number', type=int, default=0, help='index for the agent to gather statistics')

    args = parser.parse_args(argv)
    return args

# getting parameters
args = get_args(argv[1:])

# INTERACTION PARAMETERS
EPISODES = 1200 # number of episodes for each agent
MAX_STEPS_PER_TRIAL = 20000 # number of allowed timesteps before reset

# AGENT PARAMETERS
AGENT_NAME = 'flexible_PS' # tabular agent type
AGENT_NUMBER = args.agent_number
GAMMA_DAMPING = 0.0 # damping of the learning values
GLOW = 0.1 # glow parameter
POLICY = 'softmax' # policy
BETA = 0.1 # beta parameter of the softmax policy


# ENVIRONMENT PARAMETERS
ENV_NAME = 'gridworld'
DIMENSION = [100, 100] # grid size
env = TaskEnvironment(DIMENSION) # define environment
action_size = 4 # size of the action space

# DEFINE FILENAME TO SAVE RESULTS
foldername = 'results/'
pathlib.Path(foldername).mkdir(parents=True, exist_ok=True)
filename = ENV_NAME+'_'+str(DIMENSION[0])+'_'+str(DIMENSION[1])+'_'+AGENT_NAME+'_'+str(AGENT_NUMBER)
print(filename)

if __name__ == "__main__":
    agent = FlexiblePSAgent(action_size, GAMMA_DAMPING, GLOW, POLICY, BETA)

    # list to save data:
    timesteps = []

    for e in range(EPISODES):
        percept = env.reset()
        reward = 0
        done = False
        for t in range(1, MAX_STEPS_PER_TRIAL+1):
            action = agent.deliberate_and_learn(percept, reward, done)
            percept, reward, done, _ = env.step(action)
            if t == MAX_STEPS_PER_TRIAL:
                reward = 0
                done = True
            if done:
                agent.deliberate_and_learn(percept, reward, done)
                timesteps.append(t)
                break

        if e%100 == 0:
            print("Average last 100 scores (timesteps per episode) the agent achieved at " + str(e) + ": ", np.mean(timesteps[-100:]))
            # save data to file
            pickle.dump(timesteps, open(foldername+filename+'.pkl', "wb"))
