import matplotlib.pyplot as plt
from utils.plotting import plot
from utils.Collector import Collector
from agents.EpsilonGreedyWD import EpsilonGreedyWD
from agents.EpsilonGreedy import EpsilonGreedy
from agents.Greedy import Greedy
from agents.Pursuit import Pursuit
from agents.Boltzmann import Boltzmann
from agents.SA import SA
from agents.ThompsonSampling import ThompsonSampling
from agents.CBIR import CBIR
from agents.UCB import UCB
import numpy as np


FILES = ['EG_EGWD_GREEDY5x5_SOBS15.txt']
flag = 0
LEARNERS = [Greedy,EpsilonGreedy,EpsilonGreedyWD]
COLORS = {

    'Greedy': 'red',
    'EpsilonGreedy': 'green',
    'EpsilonGreedyWD': 'blue',
}

for filename in FILES:
    collector = Collector()
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            # print (line.split(' ')[1])
            if int(line.split(' ')[1]) == flag:
                collector.collect(line.split(' ')[0], -int(line.split(' ')[3]))
            elif int(line.split(' ')[1]) != flag:
                collector.reset()
                flag += 1
                collector.collect(line.split(' ')[0], -int(line.split(' ')[3]))
    f.close()
    collector.reset()
    ax = plt.gca()


    print(filename)
    for Learner in LEARNERS:
        name = Learner.__name__
        data = collector.getStats(name)
        mean,std,run = data
        txt = '\\\\'
        print(name,' & ',np.average(mean[-20],),txt)

    #     plot(ax, data, label=name, color=COLORS[name])

    # plt.legend(loc='lower right')
    # plt.title('10x10 Gridworld -- num_obs=10')
    # plt.savefig('test' + '.pdf')
    # plt.close()