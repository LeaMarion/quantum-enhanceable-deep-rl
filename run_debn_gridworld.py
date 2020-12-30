import torch
import gym
import os
import sys
import numpy as np

sys.path.insert(0, 'agents')
sys.path.insert(0, 'environments')

from env_grid_world import *
from ebm_ps_stable import *

torch.set_num_threads(1)
#####################################################################################


args = sys.argv

# environment
ENV = 'gridworld'

# stats
EPISODES = 3000  # number of episodes for each agent 2000
MAX_STEPS_PER_TRIAL = 20000  # number of allowed timesteps before reset

#agent parameters
AGENT_NAME = 'ebm'
AGENT_NUMBER = 0
hidden_layers = 1 #
hidden_units_layer = 64 #f
NUM_HIDDEN = [int(round(hidden_units_layer))]*hidden_layers
DROPOUT = [0.]*hidden_layers
DEVICE = 'cpu'
LEARNING_RATE = 0.001#float(args[3]) #float(args[3])
CAPACITY = 5000  # size of memory | Acrobot: 5000 |
BATCH_SIZE = 100 #int(args[4]) # size of training batch | Acrobot: 1000 |
REPLAY_TIME = 10
TARGET_UPDATE = 10000#int(args[2]) #float(args[2])
GAMMA = 0.99

BETA_i = 0.001
BETA_f = 1.0 #float(args[1])
schedule = 'htan' #str(args[4])
if schedule == 'htan':
    beta = np.tanh(np.linspace(BETA_i, BETA_f, EPISODES))
elif schedule =='lin':
    beta = np.linspace(BETA_i, BETA_f, EPISODES)

# ENVIRONMENT PARAMETERS
env_name = ENV
DIMENSIONS = [100,100]
env = TaskEnvironment(DIMENSIONS) #generate environment
percept_size = DIMENSIONS[0]+DIMENSIONS[1] #size of the percept space
action_size = 4 #size of the action space

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

#perparations for saving results
foldername = 'results/cluster-gridworld-results/'
filename = ENV+'_'+AGENT_NAME+'_Bf='+str(BETA_f)+'_LR='+str(LEARNING_RATE)+'_TU='+str(TARGET_UPDATE)+'_S='+schedule+'_'+str(AGENT_NUMBER)+'.txt'
print(filename)

if __name__ == "__main__":
    agent = DEBNAgent(percept_size, action_size, all_actions, dim_hidden=NUM_HIDDEN, dropout_rate=DROPOUT,
                     device = DEVICE, learning_rate=LEARNING_RATE, capacity=CAPACITY, batch_size=BATCH_SIZE, replay_time=REPLAY_TIME,
                     target_update = TARGET_UPDATE, gamma = GAMMA)
    #track running average
    # overwrite existing resultfile
    if os.path.exists(foldername+filename):
        os.remove(foldername+filename)
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
            action = agent.deliberate(percept, beta[e])
            action_index = (action[0] == 1).nonzero().item()
            percept, reward, done, _ = env.step(action_index)
            percept = to_two_hot(percept, DIMENSIONS)
            percept = np.reshape(percept, [1, percept_size])
            percept = torch.Tensor(percept)
            if t==MAX_STEPS_PER_TRIAL:
                reward = -1
            agent.learn(percept, action, reward, done)
            if done or t==MAX_STEPS_PER_TRIAL:
                print("episode: {}/{}, time until done: {}".format(e, EPISODES, t))
                sys.stdout.flush()
                #save results to a file
                with open(foldername+filename, 'a') as results_logger:
                    results_logger.write('%a\n' % (t))
                    results_logger.flush()
                break