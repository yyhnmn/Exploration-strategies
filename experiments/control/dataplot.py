import matplotlib.pyplot as plt
from utils.plotting import plot
from utils.Collector import Collector
from agents.EpsilonGreedyWD import EpsilonGreedyWD
from agents.EpsilonGreedy import EpsilonGreedy
from agents.Greedy import Greedy
collector = Collector()

filename = 'EG_EGWD_G_5x5_SOBS30.txt'
flag = 0
LEARNERS = [EpsilonGreedy,EpsilonGreedyWD,Greedy]
COLORS = {

    'EpsilonGreedy': 'red',

    'EpsilonGreedyWD': 'blue',

    'Greedy': 'green',

}

with open(filename, 'r') as f:
    lines = f.readlines()
    for line in lines:
        # print (line.split(' ')[1])
        if int(line.split(' ')[1]) == flag:
            collector.collect(line.split(' ')[0], -int(line.split(' ')[3])+1)
        elif int(line.split(' ')[1]) != flag:
            collector.reset()
            flag += 1
            collector.collect(line.split(' ')[0], -int(line.split(' ')[3])+1)
f.close()
collector.reset()
# print (collector.all_data["Boltzmann"])
ax = plt.gca()
for Learner in LEARNERS:
    name = Learner.__name__
    data = collector.getStats(name)
    plot(ax, data, label=name, color=COLORS[name])

plt.legend()
plt.title('test')
plt.savefig('savefig/test' + '.pdf')
plt.close()