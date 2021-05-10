import torch
import numpy as np
import torch.nn.functional as f
from agents.BaseAgent import BaseAgent
from utils.torch import device, getBatchColumns
import math
import random
from environments.Gridworld import Gridworld


env = Gridworld(10)
class Pursuit(BaseAgent):
    def __init__(self, features, actions, params):
        super().__init__(features, actions, params)
        # self.probs=[0.25,0.25,0.25,0.25]
        self.probs = np.ones((env.width,env.height,env.num_actions))/4


    def selectAction(self, s):
        x,y = s.numpy()[0]
        # take a random action about epsilon percent of the time
        # if np.random.rand() < self.epsilon:
        #     a = np.random.randint(self.actions)
        #     return torch.tensor(a, device=device)

        # otherwise take a greedy action
        q_s, _ = self.policy_net(s)
        
        qvalues = q_s.detach().numpy()[0]
        max_index = np.argmax(qvalues)
        # for i in range(4):
        #     if i == max_index:
        #         self.probs[i]=self.probs[i]+0.1*(1-self.probs[i])
        #     else:
        #         self.probs[i]=self.probs[i]+0.1*(0-self.probs[i])

        for i in range(4):
            if i == max_index:
                self.probs[x][y][i]=self.probs[x][y][i]+0.1*(1-self.probs[x][y][i])
            else:
                self.probs[x][y][i]=self.probs[x][y][i]+0.1*(0-self.probs[x][y][i])

        actions = [i for i in range(len(qvalues))]
        action = random.choices(actions, weights=self.probs[x][y], k=1)[0]
        return torch.tensor(action)
        # return q_s.argmax().detach()

    def updateNetwork(self, samples):
        # organize the mini-batch so that we can request "columns" from the data
        # e.g. we can get all of the actions, or all of the states with a single call
        batch = getBatchColumns(samples)

        # compute Q(s, a) for each sample in mini-batch
        Qs, x = self.policy_net(batch.states)
        Qsa = Qs.gather(1, batch.actions).squeeze()

        # by default Q(s', a') = 0 unless the next states are non-terminal
        Qspap = torch.zeros(batch.size, device=device)

        # if we don't have any non-terminal next states, then no need to bootstrap
        if batch.nterm_sp.shape[0] > 0:
            Qsp, _ = self.target_net(batch.nterm_sp)

            # bootstrapping term is the max Q value for the next-state
            # only assign to indices where the next state is non-terminal
            Qspap[batch.nterm] = Qsp.max(1).values


        # compute the empirical MSBE for this mini-batch and let torch auto-diff to optimize
        # don't worry about detaching the bootstrapping term for semi-gradient Q-learning
        # the target network handles that
        target = batch.rewards + batch.gamma * Qspap.detach()
        td_loss = 0.5 * f.mse_loss(target, Qsa)


        # make sure we have no gradients left over from previous update
        self.optimizer.zero_grad()
        self.target_net.zero_grad()

        # compute the entire gradient of the network using only the td error
        td_loss.backward()

        # update the *policy network* using the combined gradients
        self.optimizer.step()