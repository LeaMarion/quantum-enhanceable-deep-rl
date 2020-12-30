import sys
import numpy as np
import scipy.stats
from sympy.combinatorics.graycode import bin_to_gray
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pickle
import matplotlib.pyplot as plt

'''
Usage: run command line of the form `run_debn-dqn_compare.py {4-10} {False-True} 20/$i {1->$i}`
e.g. `python run_debn-dqn_compare.py 4 True 20 0`
'''

np.set_printoptions(suppress=True)
# Allow only one thread/CPU for pytorch (no gain by multiprocessing)
torch.set_num_threads(1)

# DEBN and DQN classes copied here for more readability
class DEBN(nn.Module):
	def __init__(self, dim_percept, n_action_units, dim_hidden_=[32], dropout_rate_=[0.]):
		super(DEBN, self).__init__()

		self.all_actions = [torch.from_numpy(np.array([float(digit) for digit in bin_to_gray(bin(i)[2:].zfill(n_action_units))])).float() for i in range(2**n_action_units)]
		
		#initialisation
		init_kaiman = True

		#input layer
		self.visible = nn.Linear(dim_percept+n_action_units, dim_hidden_[0])
		self.visible.apply(self._init_weights) if init_kaiman else None
		
		self.b_input = nn.Linear(dim_percept+n_action_units, 1)
		nn.init.constant_(self.b_input.bias, 0.)
		self.b_input.bias.requires_grad = False

		#hidden layers
		self.hidden = nn.ModuleList()
		for l in range(len(dim_hidden_)-1):
			self.hidden.append(nn.Linear(dim_hidden_[l], dim_hidden_[l+1]))

		for l in range(1, len(dim_hidden_) - 1):
			self.hidden[l].apply(self._init_weights) if init_kaiman else None

		#output layer
		self.output = nn.Linear(dim_hidden_[-1]+1, 1)

	def _init_weights(self, layer):
		"""
		Initializes weights with kaiming_uniform method for ReLu from PyTorch.
		"""
		if type(layer) == nn.Linear:
			nn.init.kaiming_uniform_(layer.weight, a=0, mode='fan_in', nonlinearity='relu')

	def forward(self, percept):
		outs = []
		for action in self.all_actions:
			out = torch.cat((percept, action.repeat(percept.size(0),1)), dim=1).float()
			#feedforward
			bias = self.b_input(out)
			out = F.relu(self.visible(out))
			for l in range(len(self.hidden)):
				out = F.relu(self.hidden[l](out))
			out = torch.cat((out,bias),1)
			out = self.output(out)
			outs += [out]
		return torch.cat(outs,dim=1)

class DQN(nn.Module):
	def __init__(self, dim_percept, dim_action, dim_hidden_=[32], dropout_rate_=[0.]):
		super(DQN, self).__init__()
		
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
		self.output = nn.Linear(dim_hidden_[-1], dim_action)

	def _init_weights(self, layer):
		"""
		Initializes weights with kaiming_uniform method for ReLu from PyTorch.
		"""
		if type(layer) == nn.Linear:
			nn.init.kaiming_uniform_(layer.weight, a=0, mode='fan_in', nonlinearity='relu')

	def forward(self, percept):
		out = percept.float()

		#feedforward
		out = F.relu(self.input(percept))
		for l in range(len(self.layers)):
			out = F.relu(self.layers[l](out))
		out = self.output(out)
		return out
	
def sample(values):
	values = values.detach().numpy()
	rescale = np.amax(values, axis=1)
	policy = np.exp(values-rescale[:,None])
	policy = policy / np.sum(policy, axis=1)[:,None]
	return([np.random.choice(range(len(policy[i])), p=policy[i]) for i in range(len(policy))])
	

def select(tensor,indices,l):
	return(torch.take(tensor,torch.tensor([l*i+indices[i] for i in range(len(indices))]).long()))
	
args = sys.argv

###  ENVIRONMENT PARAMETERS
dim_percept = 8
num_percepts = 2**dim_percept
n_action_units = int(args[1])
num_actions = 2**n_action_units
max_n_action_units = max(n_action_units,10) # change when considering larger env
sampling = (str(args[2]) == 'True') # when True, policy-sampling aspect is taking effect

###  AGENT PARAMETERS
batch_size = 10
learning_rate = 0.001
n_hidden_units_layer = 10 # at largest env size
nb_hidden_layers = 2

### EXECUTION PARAMETERS
simulate = True
dqn_only = False
dump = True
plot = True
save_agent = True

### AGENT-ENV PARAMETERS
# Computing nb weights
n_weights = (dim_percept + nb_hidden_layers + 2**max_n_action_units)*n_hidden_units_layer+(n_hidden_units_layer**2)*(nb_hidden_layers-1)
print("Nb weights: ", n_weights)
# Computing nb hidden units of debn and dqn
b = dim_percept + nb_hidden_layers + n_action_units
delta = b**2 + 4*(nb_hidden_layers-1)*(n_weights - dim_percept - n_action_units)
n_hidden_units_layer_debn = (-b+np.sqrt(delta))/(2*(nb_hidden_layers-1))
b = dim_percept + nb_hidden_layers + 2**n_action_units
delta = b**2 + 4*(nb_hidden_layers-1)*n_weights
n_hidden_units_layer_dqn = (-b+np.sqrt(delta))/(2*(nb_hidden_layers-1))
dim_hidden_debn = [int(round(n_hidden_units_layer_debn))]*nb_hidden_layers
dim_hidden_dqn = [int(round(n_hidden_units_layer_dqn))]*nb_hidden_layers
print("Hidden units DEBN/DQN:", dim_hidden_debn,dim_hidden_dqn)

filename = "results/multimodal5"+"_agents"+str(args[3])+(1-sampling)*"_nosample"+"_percept"+str(dim_percept)+"_layers"+str(nb_hidden_layers)+"_units"+str(n_hidden_units_layer)+"_actionunits"+str(n_action_units)+"-"+str(args[4])+".pckl"
print("Filename:", filename)


# Generate target function
x, y = np.mgrid[-1:1:2/num_percepts, -1:1:2/num_actions]
pos = np.empty(x.shape + (2,))
pos[:, :, 0] = x; pos[:, :, 1] = y
offset = 0.7
width = 0.15
rv = scipy.stats.multivariate_normal([offset, -offset],[width, width])
rv2 = scipy.stats.multivariate_normal([-offset, offset],[width, width])
rv3 = scipy.stats.multivariate_normal([-offset, -offset],[width, width])
rv4 = scipy.stats.multivariate_normal([offset, offset],[width, width])
rv5 = scipy.stats.multivariate_normal([0, 0],[width, width])
m_values = rv.pdf(pos)+rv2.pdf(pos)+rv3.pdf(pos)+rv4.pdf(pos)+0.9*rv5.pdf(pos)

all_full_losses_debn = []
all_full_losses_dqn = []
if simulate:
	for i in range(int(args[3])):
		print("Agent #{}".format(i))
		if not(dqn_only):
			debn = DEBN(dim_percept, n_action_units, dim_hidden_=dim_hidden_debn).to("cpu")
		dqn = DQN(dim_percept, num_actions, dim_hidden_=dim_hidden_dqn).to("cpu")
		if not(dqn_only):
			optimizer_debn = optim.Adam(filter(lambda p: p.requires_grad, debn.parameters()), lr=learning_rate, amsgrad=True)
		optimizer_dqn = optim.Adam(filter(lambda p: p.requires_grad, dqn.parameters()), lr=learning_rate, amsgrad=True)
		if not(dqn_only):
			full_losses_debn = []
		full_losses_dqn = []
		for j in range(5001):
			percepts = np.random.choice(num_percepts,size=batch_size)
			batch_percepts = torch.from_numpy(np.array([[float(digit) for digit in bin_to_gray(bin(i)[2:].zfill(dim_percept))] for i in percepts])).float()
			target_m_values = torch.from_numpy(m_values[percepts]).float()
			m_values_dqn = dqn(batch_percepts)
			if not(dqn_only):
				m_values_debn = debn(batch_percepts)
			if sampling:
				sample_dqn = sample(m_values_dqn)
				if not(dqn_only):
					sample_debn = sample(m_values_debn)
				loss_dqn = F.smooth_l1_loss(select(m_values_dqn,sample_dqn,num_actions), select(target_m_values,sample_dqn,num_actions))
				if not(dqn_only):
					loss_debn = F.smooth_l1_loss(select(m_values_debn,sample_dqn,num_actions), select(target_m_values,sample_dqn,num_actions))
			full_loss_dqn = F.smooth_l1_loss(m_values_dqn, target_m_values)
			if not(dqn_only):
				full_loss_debn = F.smooth_l1_loss(m_values_debn, target_m_values)
			full_losses_dqn += [full_loss_dqn.detach().numpy()]
			if not(dqn_only):
				full_losses_debn += [full_loss_debn.detach().numpy()]
			if j%500 == 0:
				if not(dqn_only):
					print(i, j, 'debn ', full_loss_debn.detach().numpy(), 'dqn ', full_loss_dqn.detach().numpy(), 'std ', np.mean(np.std(target_m_values.detach().numpy(),axis=1)))
				else:
					print(i, j, 'dqn ', full_loss_dqn, 'std ', np.mean(np.std(target_m_values.detach().numpy(),axis=1)))
			optimizer_dqn.zero_grad()
			if not(dqn_only):
				optimizer_debn.zero_grad()
			if sampling:
				loss_dqn.backward(retain_graph=True)
				if not(dqn_only):
					loss_debn.backward(retain_graph=True)
			else:
				full_loss_dqn.backward(retain_graph=True)
				if not(dqn_only):
					full_loss_debn.backward(retain_graph=True)
			optimizer_dqn.step()
			if not(dqn_only):
				optimizer_debn.step()
			if save_agent and (i==0) and (j%1000==0):
				file = open("saved_models/multimodal5_"+str(j)+(1-sampling)*"_nosample"+"_percept"+str(dim_percept)+"_layers"+str(nb_hidden_layers)+"_units"+str(n_hidden_units_layer)+"_actionunits"+str(n_action_units)+"-"+str(args[4])+".pckl", "wb")
				if not(dqn_only):
					pickle.dump([debn, dqn], file)
				else:
					pickle.dump([None, dqn], file)
				file.close()
		if not(dqn_only):
			all_full_losses_debn += [full_losses_debn]
		all_full_losses_dqn += [full_losses_dqn]

if dump:
	if dqn_only:
		pickle.dump(all_full_losses_dqn,open(filename, "wb"))
	else:
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
	plt.title('Testing loss during (semi-)supervised training')
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