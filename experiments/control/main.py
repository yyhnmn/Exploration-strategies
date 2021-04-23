import numpy as np
import torch
from RlGlue import RlGlue

from agents.QLearning import QLearning
from agents.QRC import QRC
from agents.QC import QC
from agents.EpsilonGreedy import EpsilonGreedy
from agents.EpsilonGreedyWD import EpsilonGreedyWD
from agents.UCB import UCB
from agents.Greedy import Greedy
from agents.Boltzmann import Boltzmann
from agents.ThompsonSampling import ThompsonSampling
from agents.NoisyNetAgent import NoisyNetAgent
from agents.CountBasedIR import CountBasedIR

from environments.MountainCar import MountainCar
from environments.Gridworld import Gridworld
from utils.Collector import Collector
from utils.rl_glue import RlGlueCompatWrapper

RUNS = 20
EPISODES = 100
# LEARNERS = [QRC, QC, QLearning,DQN]
LEARNERS = [Boltzmann,EpsilonGreedy,UCB]
COLORS = {
    'EpsilonGreedy': 'red',
    'EpsilonGreedyWD': 'blue',
    'UCB': 'green',
    'Greedy': 'green',
    'Boltzmann': 'blue',
    'ThompsonSampling': 'blue',
    'NoisyNetAgent': 'red',
    'CountBasedIR': 'red',
}

# use stepsizes found in parameter study
STEPSIZES = {
    'EpsilonGreedy': 0.0009765,
    'EpsilonGreedyWD': 0.0009765,
    'UCB': 0.0009765,
    'Greedy': 0.0009765,
    'Boltzmann': 0.0009765,
    'ThompsonSampling': 0.0009765,
    'NoisyNetAgent': 0.0009765,
    'CountBasedIR': 0.0009765
}

OBS = [0,10,20,30,40,50,60,70,80,90]
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
                f = open("EG_BOL_UCB_10X10_SOBS"+str(env.num_obs)+".txt", "a")
                content = str(Learner.__name__)+' ' + str(run)+' '+str(episode)+' '+str(glue.num_steps)+'\n'
                f.write(content)
                f.close()

                collector.collect(Learner.__name__, glue.total_reward)


            collector.reset()


    import matplotlib.pyplot as plt
    from utils.plotting import plot

    ax = plt.gca()

    for Learner in LEARNERS:
        name = Learner.__name__
        data = collector.getStats(name)

        plot(ax, data, label=name, color=COLORS[name])

    plt.legend()
    plt.title('10x10 Gridworld -- num_obs='+str(env.num_obs))
    plt.savefig('figures/EG_BOL_UCB_10X10_SOBS'+str(env.num_obs)+'.pdf')
    plt.close()

