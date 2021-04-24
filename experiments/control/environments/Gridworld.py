import numpy as np
from RlGlue import BaseEnvironment


UP = 0
LEFT = 1
DOWN = 2
RIGHT = 3

class Gridworld(BaseEnvironment):
    """
    Description:
    nxn gridworld. Agent reset to the center, four goals at the corner.

    Observation:
        Current position (x,y)
        x: width
        y: height

    Actions:
        0   up
        1   left
        2   down
        3   right

    Reward:
        -1  every move

    Starting State:
        top left of the gridworld

    Termination:
        bottom right of the gridworld

    """
    def __init__(self,obs):
        self.x = 0
        self.y = 0
        self.width = 10
        self.height = 10
        self.features = 2
        self.num_actions = 4
        self.num_obs = obs
        self.steps = 0
        self.env_refresh = 10
        self.obsList = []

    def start(self):
        self.x = 0
        self.y = 0
        for i in range(self.num_obs):
            obs = (np.random.randint(self.width),np.random.randint(self.height))
            while obs == (0,0) or obs == (self.width,self.height) or obs in self.obsList:
                obs = (np.random.randint(self.width),np.random.randint(self.height))
            self.obsList.append(obs)

        return (self.x, self.y)

    # give all actions for a given state
    def actions(self, s):
        return [UP, LEFT, DOWN, RIGHT]

    # give the rewards associated with a given state, action, next state tuple
    def rewards(self, s, a, sp):
        return -1

    # get the next state and termination status
    def next_state(self, s, a):
        x,y = s
        self.steps += 1

        if self.env_refresh:
            if self.steps % self.env_refresh == 0:
                self.obsList = []
                for i in range(self.num_obs):
                    obs = (np.random.randint(self.width),np.random.randint(self.height))
                    while obs == s or obs == (self.width,self.height) or obs in self.obsList:
                        obs = (np.random.randint(self.width),np.random.randint(self.height))
                    self.obsList.append(obs)

        if a == UP:
            obsFlag = 0
            for obs in self.obsList:
                if obs[0] == x and obs[1] == y-1:
                    obsFlag = 1
            if obsFlag == 0:
                y = max(0,y-1)

        elif a == LEFT:
            obsFlag = 0
            for obs in self.obsList:
                if obs[0] == x-1 and obs[1] == y:
                    obsFlag = 1
            if obsFlag == 0:
                x = max(0,x-1)

        elif a == DOWN:
            obsFlag = 0
            for obs in self.obsList:
                if obs[0] == x and obs[1] == y+1:
                    obsFlag = 1
            if obsFlag == 0:
                y = min(y+1,self.height-1)

        elif a == RIGHT:
            obsFlag = 0
            for obs in self.obsList:
                if obs[0] == x+1 and obs[1] == y:
                    obsFlag = 1
            if obsFlag == 0:
                x = min(x+1,self.width-1)

        if x == self.width-1 and y == self.height-1:
            return (x,y), True

        return (x,y), False


    def step(self, a):
        s = (self.x, self.y)
        sp, t = self.next_state(s, a)
        self.x = sp[0]
        self.y = sp[1]

        r = self.rewards(s, a, sp)

        return (r, sp, t)
