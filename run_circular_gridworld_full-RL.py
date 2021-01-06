import os
import sys
import numpy as np
import torch
import pickle
from environments.circular_gridworld import Circular_GridWorld

'''
Usage: run command line of the form `python run_circular_gridworld_full-RL.py {sarsa-ps} {debn-dqn} {17-21-25-29} 0.9 True {0->19}`
e.g. `python run_circular_gridworld_full-RL.py sarsa dqn 17 0.9 True 0`
'''


args = sys.argv
if str(args[1])=="sarsa":
	if str(args[2])=="debn":
		from agents.debn_sarsa import *
	elif str(args[2])=="dqn":
		from agents.dqn_sarsa import *
elif str(args[1])=="ps":
	if str(args[2])=="debn":
		from agents.debn_ps import *
	elif str(args[2])=="dqn":
		from agents.dqn_ps import *

# Allow only one thread/CPU for pytorch (no gain by multiprocessing)
torch.set_num_threads(1)

### INTERACTION PARAMETERS
INTERACTIONS = 200000 # number of interactions for each agent
NUM_AGENTS = 1 # number of agents

###  AGENT PARAMETERS
CAPACITY = 5000  # size of memory
BATCH_SIZE = 100  # size of training batch
LEARNING_RATE = 0.01  # learning rate for optimizer
REPLAY = 10 # every X interactions, the network is trained
TARGET_UPDATE = 50  # every X experience replays the target net is updated
DROPOUT_RATE = [0.,0.]  # dropout rate
GAMMA = float(args[4]) # discount factor
DEVICE = "cpu"
#SOFTMAX SCHEDULE
b_i=1
b_f=100
softmax = np.linspace(b_i, b_f, INTERACTIONS)

###  ENVIRONMENT PARAMETERS
one_all = (str(args[5]) == "True") # when True, only shortest action rewarded
env = Circular_GridWorld(grid_size=np.array([int(args[3])]),one_action=one_all)
percept_size = env.gridlength
all_actions = env.generate_all_actions()
action_size = int(env.num_actions)
action_length = int(env.longest_sequence * 2)

### AGENT-ENV PARAMETERS
num_percepts = percept_size
max_num_percepts = max(num_percepts,29) # change when considering larger env
dim_percept = num_percepts
n_action_units = action_length
num_actions = action_size

n_hidden_units_layer = 10 # at largest env size
nb_hidden_layers = 2
max_num_actions = int(((max_num_percepts+1)/2)**2)

# Compute nb weights of debn and dqn
n_weights = (max_num_percepts + nb_hidden_layers + max_num_actions)*n_hidden_units_layer+(n_hidden_units_layer**2)*(nb_hidden_layers-1)
print("Nb weights: ", n_weights)
b = dim_percept + nb_hidden_layers + n_action_units
delta = b**2 + 4*(nb_hidden_layers-1)*(n_weights - dim_percept - n_action_units)
n_hidden_units_layer_debn = (-b+np.sqrt(delta))/(2*(nb_hidden_layers-1))
b = dim_percept + nb_hidden_layers + num_actions
delta = b**2 + 4*(nb_hidden_layers-1)*n_weights
n_hidden_units_layer_dqn = (-b+np.sqrt(delta))/(2*(nb_hidden_layers-1))
NUM_HIDDEN_DEBN = [int(round(n_hidden_units_layer_debn))]*nb_hidden_layers
NUM_HIDDEN_DQN = [int(round(n_hidden_units_layer_dqn))]*nb_hidden_layers
print("Hidden units DEBN/DQN:", NUM_HIDDEN_DEBN,NUM_HIDDEN_DQN)

filename = "results/"+str(args[1])+"_"+str(args[2])+"_onegoodaction"*one_all+"_circular_gridworld_RL_gamma"+str(args[4])+"_grid"+str(env.gridlength)+'_'+str(args[6])+'.pckl'
print("Filename:", filename)


#PERCEPT ENCODING
def one_hot(percept):
	one_hot = np.zeros(env.gridlength)
	one_hot[int(percept[0])]=1.
	return one_hot


if __name__ == "__main__":
	for ag in range(NUM_AGENTS):
		print("Agent #{}".format(ag))
		sys.stdout.flush()
		#reset the environment
		env.reset()
		#initialize agent
		if str(args[1])=="sarsa":
			if str(args[2])=="debn":
				agent = DEBNAgent(percept_size, action_length, all_actions,
							 capacity=CAPACITY, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE,
							 dim_hidden=NUM_HIDDEN_DEBN, dropout_rate=DROPOUT_RATE, gamma=GAMMA,
							 target_update = TARGET_UPDATE, replay_time = REPLAY, episodic = False)
			elif str(args[2])=="dqn":
				agent = DQNAgent(percept_size, action_size, dim_hidden=NUM_HIDDEN_DQN, dropout_rate=DROPOUT_RATE, target_update=TARGET_UPDATE,
							 device=DEVICE, learning_rate=LEARNING_RATE, capacity=CAPACITY,
							 batch_size=BATCH_SIZE, gamma = GAMMA, replay_time = REPLAY, episodic = False)
		if str(args[1])=="ps":
			if str(args[2])=="debn":
				agent = DEBNAgent(percept_size, action_length, all_actions,
							 capacity=CAPACITY, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE,
							 dim_hidden=NUM_HIDDEN_DEBN, dropout_rate=DROPOUT_RATE, gamma=GAMMA,
							 target_update = TARGET_UPDATE, replay_time = REPLAY, episodic = False)
			elif str(args[2])=="dqn":
				agent = DQNAgent(percept_size, action_size, dim_hidden=NUM_HIDDEN_DQN, dropout_rate=DROPOUT_RATE, target_update=TARGET_UPDATE,
							 device=DEVICE, learning_rate=LEARNING_RATE, capacity=CAPACITY,
							 batch_size=BATCH_SIZE, gamma = GAMMA, replay_time = REPLAY, episodic = False)
		percept = env.state
		percept = one_hot(percept)
		percept = np.reshape(percept, [1, percept_size])
		percept = torch.Tensor(percept)
		rewards = []
		for e in range(INTERACTIONS+1):
			action = agent.deliberate(percept, softmax[e])
			if str(args[2])=="dqn":
				action_tensor = all_actions[action[0]]
				next_percept, reward, done = env.step(action_tensor)
			elif str(args[2])=="debn":
				next_percept, reward, done = env.step(action)
			agent.learn(percept, action, reward, done)

			percept = next_percept
			percept = one_hot(percept)
			percept = np.reshape(percept , [1, percept_size])
			percept = torch.Tensor(percept)
			rewards += [reward]
			if e%1000 == 0:
				print("Average last 1000 rewards at "+str(e)+": ", np.mean(rewards[-1000:]))
				# dump rewards in the file
				pickle.dump(rewards,open(filename, "wb"))

		#Save models
		torch.save(agent.policy_net.state_dict(), 'saved_models/'+str(args[1])+"_"+str(args[2])+"_onegoodaction"*one_all+"_circular_gridworld_RL_gamma"+str(args[4])+"_grid"+str(env.gridlength)+'_'+str(args[6])+ '.pckl')


