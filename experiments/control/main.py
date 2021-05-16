import numpy as np
import torch
from RlGlue import RlGlue
import matplotlib.pyplot as plt
from utils.plotting import plot


from agents.QLearning import QLearning
from agents.QRC import QRC
from agents.QC import QC
from agents.EpsilonGreedy import EpsilonGreedy

from agents.EpsilonGreedyWD import EpsilonGreedyWD
from agents.UCB import UCB
from agents.Greedy import Greedy

from agents.SA import SA
from agents.Boltzmann import Boltzmann
from agents.Boltzmann001 import Boltzmann001
from agents.Boltzmann0005 import Boltzmann0005
from agents.Boltzmann01 import Boltzmann01
from agents.Boltzmann05 import Boltzmann05
from agents.QTable import QTable

from agents.ThompsonSampling import ThompsonSampling
from agents.NoisyNetAgent import NoisyNetAgent

from agents.Pursuit import Pursuit
from agents.CBIR import CBIR
from environments.MountainCar import MountainCar
from environments.Gridworld import Gridworld
from utils.Collector import Collector
from utils.rl_glue import RlGlueCompatWrapper

RUNS = 20
EPISODES = 100
# LEARNERS = [QRC, QC, QLearning,DQN]
#here
LEARNERS = [EpsilonGreedy,CBIR,UCB]
COLORS = {
    'CBIR': 'red',
    'EpsilonGreedy': 'blue',
    'SA': 'green',
    'EpsilonGreedyWD': 'blue',
    'UCB': 'green',
    'Greedy': 'green',
    'ThompsonSampling': 'red',
    'NoisyNetAgent': 'red',
    'Pursuit':'green',
}

# use stepsizes found in parameter study
# 0.0009765
STEPSIZES = {
    'EpsilonGreedy': 0.001,
    'ThompsonSampling': 0.001,
    'CBIR': 0.001,
    'UCB':0.001,
    'EpsilonGreedyWD':0.001,
    'Pursuit':0.001,
    'Greedy':0.001,
    'Boltzmann':0.001,
    'SA':0.001,
}

OBS = [0,10,20,30,40,50,60]
for obs in OBS:
    collector = Collector()
    for run in range(RUNS):
        for Learner in LEARNERS:

            np.random.seed(run)
            torch.manual_seed(run)

            env = Gridworld(obs)

            learner = Learner(env.features, env.num_actions, {
                'alpha': STEPSIZES[Learner.__name__],
                'epsilon': 0.1,
                'beta': 1.0,
                'target_refresh': 1,
                'buffer_size': 4000,
                'h1': 32,
                'h2': 32,
            })

            agent = RlGlueCompatWrapper(learner, gamma=0.99)

            glue = RlGlue(agent, env)

            glue.start()
            for episode in range(EPISODES):
                glue.num_steps = 0
                glue.total_reward = 0

                glue.runEpisode(max_steps=1000)

                print(Learner.__name__, run, episode, glue.num_steps)
                #here
                f = open("EG_CBIR_UCB5x5_SOBS"+str(env.num_obs)+".txt", "a")
                content = str(Learner.__name__)+' ' + str(run) + \
                    ' '+str(episode)+' '+str(glue.num_steps)+'\n'
                f.write(content)
                f.close()

                collector.collect(Learner.__name__, glue.total_reward)

            collector.reset()


    ax = plt.gca()

    for Learner in LEARNERS:
        name = Learner.__name__
        data = collector.getStats(name)

        plot(ax, data, label=name, color=COLORS[name])

    plt.legend()
    plt.title('5x5 Gridworld -- num_obs='+str(env.num_obs))

    # here
    plt.savefig('figures/EG_CBIR_UCB5x5_SOBS'+str(env.num_obs)+'.pdf')
    plt.close()
