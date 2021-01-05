import os
import sys
#matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import math
import torch
import pickle
from environments.circular_gridworld import Circular_GridWorld

'''
Usage: run command line of the form `run_circular_gridworld_reward-discounting.py {sarsa-ps} {5-9-13-17-21-25} {0-0.9} True 20/$i {1->$i}`
e.g. `python run_circular_gridworld_reward-discounting.py sarsa 5 0.9 True 20 1`
'''

args = sys.argv
if str(args[1])=="sarsa":
	from agents.debn_sarsa import *
	from agents.dqn_sarsa import *
elif str(args[1])=="ps":
	from agents.debn_ps import *
	from agents.dqn_ps import *

# Allow only one thread/CPU for pytorch (no gain by multiprocessing)
torch.set_num_threads(1)

args = sys.argv

### INTERACTION PARAMETERS
EPISODES = 500 # number of episodes for each agent
NUM_AGENTS = int(args[5])
MAX_STEPS_PER_TRIAL = 100

###  AGENT PARAMETERS
CAPACITY = 2000  # size of memory
BATCH_SIZE = 10  # size of training batch
GAMMA = float(args[3])
LR = 0.01  # learning rate for optimizer
REPLAY = 10  # every X interactions, the network is trained
UPDATE = 10  # every X experience replays the target net is updated
DROPOUT = [0.,0.]  # dropout rate

###  ENVIRONMENT PARAMETERS
p = 0.99 # probability of good action
reward = 1
good = reward*(1-GAMMA+GAMMA*p)/(1-GAMMA)
bad = reward*GAMMA*p/(1-GAMMA)
#print(good,bad)
one_all = (str(args[4]) == "True") # when True, only shortest action rewarded

### AGENT-ENV PARAMETERS
num_percepts = int(args[2])
max_num_percepts = max(num_percepts,25) # change when considering larger env
dim_percept = num_percepts
n_action_units = num_percepts+1
num_actions = int((n_action_units/2)**2)

n_hidden_units_layer = 10 # at largest env size
nb_hidden_layers = 2
max_num_actions = int(((max_num_percepts+1)/2)**2)

# Compute nb weights debn and dqn
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

### EXECUTION PARAMETERS
plot = True
dump = True
save_agent = True

filename = "results/"+str(args[1])+"_circular_gridworld"+"_onegoodaction"*one_all+"_target"+str(UPDATE)+"_agents"+str(args[5])+"_gamma"+str(args[3])+"_percept"+str(dim_percept)+"_layers"+str(nb_hidden_layers)+"_units"+str(n_hidden_units_layer)+"_actionunits"+str(n_action_units)+"-"+str(args[6])+".pckl"
print("Filename:", filename)


def one_hot(percept):
	one_hot = np.zeros(env.gridlength)
	one_hot[int(percept)]=1.
	return one_hot

def evaluate_m_values_debn(percept,all_actions,agent):
	m_values = []
	for action in all_actions:
		m_values+=[agent.policy_net(percept,action)[0].detach().numpy()[0]]
	m_values = np.array(m_values)
	return(m_values)

def evaluate_m_values_dqn(percept,agent):
	m_values = agent.policy_net(percept)[0].detach().numpy()
	m_values = np.array(m_values)
	return(m_values)

def mse(vector_1,vector_2):
	return(np.mean((vector_1-vector_2)**2))

if __name__ == "__main__":
	env = Circular_GridWorld(grid_size=np.array([dim_percept]),one_action=one_all)
	percept_size = env.gridlength
	all_actions = env.generate_all_actions()
	action_size = int(env.num_actions)
	action_length = int(percept_size+1) # Composite movement (i to left, j to right), 2-hot encoding
	print('Nb states: ', percept_size)
	print('Nb actions: ', action_size)
	
	# Compute env immediate rewards
	rewards = np.zeros((percept_size,action_size))
	for percept in range(env.gridlength):
		state = np.array([float(percept)])
		for i in range(all_actions.size(0)):
			action = all_actions[i]
			rewards[percept][i]+=[env.step(action,state)[1]]

	# Train DEBN and DQN off-policy
	all_full_losses_debn = []
	all_full_losses_dqn = []
	for ag in range(NUM_AGENTS):
		print("Agent #{}".format(ag))
		sys.stdout.flush()


		agent = DEBNAgent(percept_size, action_length, all_actions,
						 capacity=CAPACITY, batch_size=BATCH_SIZE, gamma=GAMMA, learning_rate=LR,
						 dim_hidden=NUM_HIDDEN_DEBN, replay_time=REPLAY,
						 dropout_rate=DROPOUT, target_update = UPDATE)
		dqn_agent = DQNAgent(percept_size, action_size,
						 capacity=CAPACITY, batch_size=BATCH_SIZE, gamma=GAMMA, learning_rate=LR,
						 dim_hidden=NUM_HIDDEN_DQN, replay_time=REPLAY,
						 dropout_rate=DROPOUT, target_update = UPDATE)

		full_losses_debn = []
		full_losses_dqn = []
		for e in range(EPISODES):
			reward = 0
			percept = float(np.random.choice(env.gridlength))
			state = np.array([percept])
			percept = one_hot(percept)
			percept = np.reshape(percept , [1, percept_size])
			percept = torch.Tensor(percept)
			rewarded = rewards[int(state[0])]
			losses_debn = []
			losses_dqn = []
			for t in range(1, MAX_STEPS_PER_TRIAL+1):
				policy = rewarded*p/np.sum(rewarded)+(1-rewarded)*(1-p)/np.sum(1-rewarded) # off-policy
				action_id = np.random.choice(range(action_size),p=policy)
				action = all_actions[action_id]
				agent.deliberate_and_learn(percept, action, reward, GAMMA, done=(t==MAX_STEPS_PER_TRIAL))
				dqn_agent.deliberate_and_learn(percept, action_id, reward, GAMMA, done=(t==MAX_STEPS_PER_TRIAL))
				reward = rewarded[action_id]
				percept = float(np.random.choice(env.gridlength))
				state = np.array([percept])
				percept = one_hot(percept)
				percept = np.reshape(percept , [1, percept_size])
				percept = torch.Tensor(percept)
				rewarded = rewards[int(state[0])]
				# monitoring test performance 
				if t>=9*MAX_STEPS_PER_TRIAL/10:
					target_m_values = rewarded*good+(1-rewarded)*bad
					loss_debn = mse(target_m_values,evaluate_m_values_debn(percept,all_actions,agent))
					loss_dqn = mse(target_m_values,evaluate_m_values_dqn(percept,dqn_agent))
					losses_debn += [loss_debn]
					losses_dqn += [loss_dqn]
			full_losses_debn += [np.mean(losses_debn)]
			full_losses_dqn += [np.mean(losses_dqn)]
			if e%10 == 0:
				print(e, full_losses_debn[-1])
				print("dqn", full_losses_dqn[-1])
				print("")
				if save_agent and (ag==0):
					file = open('saved_models/'+str(args[1])+"_circular_gridworld"+"_onegoodaction"*one_all+"_target"+str(UPDATE)+"_agents"+str(args[5])+"_gamma"+str(args[3])+"_percept"+str(dim_percept)+"_layers"+str(nb_hidden_layers)+"_units"+str(n_hidden_units_layer)+"_actionunits"+str(n_action_units)+"-"+str(args[6])+".pckl", "wb")
					pickle.dump([agent.policy_net, dqn_agent.policy_net], file)
		all_full_losses_debn += [full_losses_debn]
		all_full_losses_dqn += [full_losses_dqn]

if dump:
	pickle.dump([all_full_losses_debn,all_full_losses_dqn],open(filename, "wb"))

if plot:
	w = 20
	h = 15
	d = 130
	plt.figure(figsize=(w, h), dpi=d)

	all_full_losses_debn = np.array(all_full_losses_debn)
	all_full_losses_dqn = np.array(all_full_losses_dqn)
	mean_debn = np.mean(all_full_losses_debn, axis=0)
	std_debn = np.std(all_full_losses_debn, axis=0)
	mean_dqn = np.mean(all_full_losses_dqn, axis=0)
	std_dqn = np.std(all_full_losses_dqn, axis=0)
	
	x = range(len(mean_dqn))

	plt.subplot(211)
	plt.title('Testing loss during off-policy training')
	plt.grid(True)
	plt.yscale('log')
	plt.fill_between(x, mean_dqn+std_dqn, mean_dqn-std_dqn,alpha=0.3,color='r')
	plt.plot(x,mean_dqn,c='r')
	plt.fill_between(x, mean_debn+std_debn, mean_debn-std_debn,alpha=0.3,color='g')
	plt.plot(x,mean_debn,c='g')
	
	plt.subplot(212)
	plt.grid(True)
	plt.fill_between(x, mean_dqn+std_dqn, mean_dqn-std_dqn,alpha=0.3,color='r')
	plt.plot(x,mean_dqn,c='r')
	plt.fill_between(x, mean_debn+std_debn, mean_debn-std_debn,alpha=0.3,color='g')
	plt.plot(x,mean_debn,c='g')
	#plt.savefig(filename[:-5]+'.png')
	plt.show()



