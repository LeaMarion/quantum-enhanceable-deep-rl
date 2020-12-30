import numpy as np
#from scipy.spatial import distance
from itertools import product
import torch
import random
from functools import reduce


class Circular_GridWorld():
    def __init__(self, **kwargs):
        """
            This environment is a d-dimenional gridworld of size (n_0,n_1,...).
            The agent perceives its position in the grid.
            It does not see the reward, which is fixed at the corner of the world.
            The agents goal is to find the reward in one step by choosing the correct action
            sequence. The reward is scaled by the numbers of actions the agent takes to get to
            the goal.
             **kwargs:
                    grid_size (:obj:`np.array` of :obj:`int`): The size of the grid.
                                                           Defaults to [10, 10].
                    reward_pos (:obj:`list` of :obj:`int`): The position of the
                                                            reward in the grid.
                                                            Defaults to [0, 0].
            """
        if 'grid_size' in kwargs:
            setattr(self, 'gridsize', kwargs['grid_size'])
        else:
            setattr(self, 'gridsize', np.array([21]))
        if 'one_action' in kwargs:
            setattr(self, 'oneaction', kwargs['one_action'])
        else:
            setattr(self, 'oneaction', True)
    #    if 'reward_pos' in kwargs and type(kwargs['reward_pos']) is np.array:
    #        setattr(self, 'goal', kwargs['goal'])
    #    else:
    #        setattr(self, 'goal', np.array([]))
        #goal state
        self.goal = np.array([int(self.gridsize/2.)])
        #dimension of the grid world
        self.dim = np.size(self.gridsize)
        #length of the grid
        self.gridlength = self.gridsize[0]
        #print(self.gridlength)
        #the length of the longest path to goal
        self.longest_sequence = self.calculate_longest_sequence()
        #all actions
        self.actions = self.generate_all_actions()
        #print(len(self.actions))
        #number of actions
        self.num_actions = len(self.actions)
        #origin
        self.origin = np.zeros(np.size(self.gridsize))
        #print('goal:',self.goal)
        #current state of the environment
        self.state = self.reset()


    def step(self, action_sequence, state=None):
        """
        checks if the goal was reached by the action sequence, if it was reached the agent
        receives a reward that is discounted by the number of steps the agent took
        :param action_sequence (torch tensor): the action sequence chosen by the agent
        :return: done   (bool) set to True after each step
                 reward (float) if goal is reached it is given by d/all_taken_actions
        """
        if state!=None:
            self.state = state
        #calculate minimum distance to the reward from current state
        d_min = np.sum(np.abs(np.array(self.state)-np.array(self.goal)))#distance.cityblock(self.state,self.goal)
        #convert the received action into numpy array
        action_sequence = action_sequence.numpy()[0]
        #the action array keeps track of all the performed actions
        actions = np.nonzero(action_sequence)[0]
        #the second action has to be shifted so that it is a number between 0-gridsize
        actions[1] = actions[1]-self.longest_sequence
        #actions sum caclulates the steps taken by the actions
        action_sum = sum(actions)
        #the direction only tracks the
        directions = actions[0]-actions[1]
        #the agent is rewarded if the goal was reached with the action sequence
        reward = 0.
        new_state = self.state+np.array(directions)
        new_state = np.mod(new_state,self.gridlength)

        if (new_state == self.goal).all():
            if d_min == action_sum or not(self.oneaction):
                reward = 1.
            if self.state[0]+1 == self.goal:
                self.state = np.array([self.state[0]+2])
            else:
                self.state = self.state+np.array([1])
                self.state = np.mod(self.state, self.gridlength)
        else:
            reward = 0.0
            self.state = new_state
        #print(reward)
        # print('new state',new_state, 'reward', reward)
        # print('state after',self.state)
        #print(reward)
            #reward = (d_min/action_sum)
        #one step environment, set to true after every step
        done = False
        return self.state, reward, done

    def loop(self,state):
        self.state = state

    def reset(self):
        """
        initializes self.state to a random state that is not the goal state,
        hereby we assume that the goal is in the furthers corner of the grid
        and that there are no walls.
        :return: state (np.array)
        """
        self.state = np.array([0])
        return self.state

#-------------------helper function-------------------------------------------------
    def generate_all_actions(self):
        """
        generates the set of all action sequences
        :return: all_actions (list) of torch tensors action sequences
        """
        a = []
        for j in range(self.longest_sequence):
            b = np.zeros([self.longest_sequence])
            b[j]=1.0
            a.append(b)

        all_actions = torch.empty(0)
        ind = product(range(len(a)), repeat=2)
        for i in ind:
            action = []
            for j in range(len(i)):
                action.append(a[i[j]])
            action = np.concatenate(action)
            action = np.reshape(action, [1, int(self.longest_sequence*2.)])
            action = torch.Tensor([action])
            all_actions = torch.cat((all_actions, action))
        return all_actions

    def one_hot(self, percept):
        one_hot = np.zeros(self.gridlength)
        one_hot[int(percept[0])] = 1.
        return one_hot

    def generate_dataset(self):
        dataset = []
        for state in range(self.gridlength):
            if state != self.goal[0]:
                state = np.array([float(state)])
                #print(state
                for action in self.actions:
                   # print('action', action)
                    self.state = state
                    new_state, reward, done = self.step(action)
                    #percept = self.one_hot(state)
                    #percept = np.reshape(percept, [1, self.gridlength])
                    #percept = torch.Tensor(percept)

                   # print('reward', reward)
                    dataset.append([state, action, reward])
        return dataset

    def generate_reward(self):
        dataset = []
        for state in range(self.gridlength):
            if state != self.goal[0]:
                state = np.array([float(state)])
                #print(state)
                for action in self.actions:
                    self.state = state
                   # print('action', action)
                    new_state, reward, done = self.step(action)
                    #percept = self.one_hot(state)
                    #percept = np.reshape(percept, [1, self.gridlength])
                    #percept = torch.Tensor(percept)

                   # print('reward', reward)
                    if reward == 1.0:
                        dataset.append([state, action, reward])
        return dataset

    def generate_dnn_dataset(self):
        dataset = []
        for state in range(self.gridlength):
            if state != self.goal[0]:
                for i in range(len(self.actions)):
                  #  print('action', self.actions[i], i)
                    self.state = np.array([float(state)])
                    new_state, reward, done = self.step(self.actions[i])
                    #percept = self.one_hot(state[0])
                    #percept = np.reshape(percept, [1, self.gridlength])
                    #percept = torch.Tensor(percept)
                    #print('reward', reward)
                    dataset.append([state, [int(i)], reward])
        return dataset

    def generate_dnn_reward(self):
        dataset = []
        for state in range(self.gridlength):
            if state != self.goal[0]:
                #print(state)
                state = np.array([float(state)])
                #print(state)
                #print(self.state)
                for i in range(len(self.actions)):
                    self.state = state
                  #  print('action', self.actions[i], i)
                    new_state, reward, done = self.step(self.actions[i])
                    #percept = self.one_hot(state[0])
                    #percept = np.reshape(percept, [1, self.gridlength])
                    #percept = torch.Tensor(percept)
                    #print('reward', reward)
                    if reward == 1.0:
                        dataset.append([state, [int(i)], reward])
        return dataset

    def all_states(self):
        states = []
        for i in range(self.gridlength-1):
            states.append(np.array([float(i)]))
        return states

    def calculate_longest_sequence(self):
        if self.gridlength-self.goal[0] > self.goal[0]:
            return self.gridlength-self.goal[0]
        else:
            return self.goal[0]+1




#------------------------------testing-------------------------------------
if __name__ == "__main__":
    env = Circular_GridWorld(grid_size=np.array([21]))
    print(env.gridlength)
    print(len(env.actions))
    print(env.actions)


