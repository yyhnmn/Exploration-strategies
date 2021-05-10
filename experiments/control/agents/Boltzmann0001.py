import torch
import numpy as np
import torch.nn.functional as f
from agents.BaseAgent import BaseAgent
from utils.torch import device, getBatchColumns
import random
import math

class Boltzmann0001(BaseAgent):
    def __init__(self, features, actions, params):
        super().__init__(features, actions, params)
        self.alpha = 0.001

    def selectAction(self, x):

        probs= [] 
        tau = max(0.5*(0.9999**self.steps),0.001)
        # otherwise take a greedy action
        q_s, _ = self.policy_net(x)
        qvalues = q_s.detach().numpy()[0]
        boltzmann = np.zeros((len(qvalues)))

        for q in range(len(qvalues)):
            num = math.exp((qvalues[q]-max(qvalues))/tau)
            boltzmann[q] = num
        probs = boltzmann/sum(boltzmann)

        actions = [i for i in range(len(qvalues))]
        action = random.choices(actions, weights=probs, k=1)[0]
        # action = probs.argmax()
        return torch.tensor(action)

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