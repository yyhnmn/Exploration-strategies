import numpy as np
import torch
import torch.nn.functional as f
import torch.optim as optim

from agents.Network import Network
from utils.ReplayBuffer import ReplayBuffer
from utils.torch import device
from environments.Gridworld import Gridworld

env = Gridworld(10)
class QTable:
    def __init__(self, features, actions, params):
        self.features = features
        self.actions = actions
        self.params = params

        # define parameter contract
        self.alpha = params['alpha']
        self.epsilon = params['epsilon']
        self.q_table = dict() # Store all Q-values in dictionary of dictionaries 
        for x in range(env.width): # Loop through all possible grid spaces, create sub-dictionary for each
            for y in range(env.height):
                self.q_table[(x,y)] = {}
                for a in range(self.actions):
                    self.q_table[(x,y)][a]=10  # Populate sub-dictionary with zero values for possible moves

    def selectAction(self, s):
        x,y = s.numpy()[0]

        s = (x,y)
        # take a random action about epsilon percent of the time
        if np.random.rand() < self.epsilon:
            a = np.random.randint(self.actions)
        else:
            q_values_of_state = self.q_table[s]

            maxValue = max(q_values_of_state.values())
            a = np.random.choice([k for k, v in q_values_of_state.items() if v == maxValue])

        return a


    def update(self, s, a, r, sp, gamma):
        if sp is None:
            pass
        else:
            sp = (sp.numpy()[0][0],sp.numpy()[0][1])
            s = (s.numpy()[0][0],s.numpy()[0][1])
            q_values_of_state = self.q_table[sp]
            max_q_value_in_new_state = max(q_values_of_state.values())
            self.q_table[s][a] += self.alpha * (r + gamma * max_q_value_in_new_state - self.q_table[s][a])

