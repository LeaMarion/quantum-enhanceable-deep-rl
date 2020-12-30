import torch
import gym
import os
import sys
import numpy as np
from datetime import datetime

sys.path.insert(0, 'agents')

from debn_ps import *

torch.set_num_threads(1)
#####################################################################################


args = sys.argv

# environment
ENV = 'CartPole-v1'  # | 'Acrobot-v1' | 'CartPole-v1' | 'Pendulum-v0' |

# stats
EPISODES = 2000  # number of episodes for each agent 2000
NUM_AGENTS = 1  # number of agents 50
MAX_STEPS_PER_TRIAL = 10000  # number of allowed timesteps before reset 1000

#agent parameters
AGENT_NAME = 'ebm'
hidden_layers = int(args[1])
hidden_units_layer = float(args[2])

NUM_HIDDEN = [int(round(hidden_units_layer))]*hidden_layers
DROPOUT = [0.]*hidden_layers  # dropout rate | Acrobot: [0., 0., 0.] |

DEVICE = 'cpu'
LEARNING_RATE = float(args[3])
CAPACITY = 2000  # size of memory | Acrobot: 5000 |
BATCH_SIZE = 100 #int(args[4]) # size of training batch | Acrobot: 1000 |
REPLAY_TIME = 10
TARGET_UPDATE = 1000 #float(args[2])

ETA_i=0.99
ETA_f=0.99
ETA_GLOW = 1.
#constant at the moment
eta = np.exp(np.linspace(np.log(ETA_i),np.log(ETA_f),EPISODES))

GAMMA = 0.0 #float(args[3]) #0.0  #decay of the h-values

#SOFTMAX SCHEDULE
BETA = 0.01
beta = np.tanh(np.linspace(BETA, 10, EPISODES))
print(beta)
# environment parameters
env_name = ENV
env = gym.make(env_name) #generate environment
percept_size = env.observation_space.shape[0] #size of the percept space
print('percept size', percept_size)
action_size = env.action_space.n #size of the action space
print('action size', action_size)

#count all actions
all_actions = torch.empty(0)
for i in range(action_size):
    a = torch.zeros((1, 1, action_size))
    a = a.new_full((1, 1, action_size), 0.)
    a[0, 0, i] = 1.
    all_actions = torch.cat((all_actions, a))

#perparations for saving results
AGENT_NUMBER = 0#int(args[4])
filename = 'cluster-cartpole-results/'+ENV+'_'+AGENT_NAME+'_HL='+str(hidden_layers)+'_HU='+str(hidden_units_layer)+'_LR='+str(LEARNING_RATE)+'_TU='+str(TARGET_UPDATE)+'_RT='+str(REPLAY_TIME)+'_G='+str(GAMMA)+'_'+str(AGENT_NUMBER)+'.txt'
print(filename)


#####################################################################################
if __name__ == "__main__":
    now = datetime.now()
    for i in range(NUM_AGENTS):
        print("Agent #{}".format(i))
        agent = EBMAgent(percept_size, action_size, all_actions, dim_hidden=NUM_HIDDEN, dropout_rate=DROPOUT,
                         device = DEVICE, learning_rate=LEARNING_RATE, capacity=CAPACITY, batch_size=BATCH_SIZE, replay_time=REPLAY_TIME,
                         target_update = TARGET_UPDATE, eta_glow= ETA_GLOW, gamma = GAMMA)
        #track running average
        average = 0.
        params = count_parameters(agent.policy_net)
        print(params)
        # overwrite existing resultfile
        if os.path.exists(filename):
            os.remove(filename)
        for e in range(EPISODES):
            counter = 0
            #reset the environment
            percept = env.reset()
            percept = np.reshape(percept, [1, percept_size])
            percept = torch.Tensor(percept)
            reward = 0.
            done = False
            for t in range(1, MAX_STEPS_PER_TRIAL + 1):
                action = agent.deliberate(percept, beta[e])
                action = (action[0] == 1).nonzero().item()
                percept, reward, done, _ = env.step(action)
                percept = np.reshape(percept, [1, percept_size])
                percept = torch.Tensor(percept)
                agent.learn(reward, eta[e], done)
                if done:
                    #calculate running average
                    average = (e*average + t)/(e+1)
                    print("episode: {}/{}, time until done: {}".format(e, EPISODES, t))
                    if average > 350:
                        print("average time: {}".format(average))
                    sys.stdout.flush()
                    #save results to a file
                    with open(filename, 'a') as results_logger:
                        results_logger.write('%a\n' % (t))
                        results_logger.flush()
                    break

    env.env.close()

later = datetime.now()
print(later - now)