"""
A simple nxn Gridworld environment
"""

import math
import gym
from gym import spaces
import numpy as np


class Env(gym.Env):
    """
    Description:
    nxn gridworld. Agent reset to the center, four goals at the corner.

    Observation:
    Type: Box(2,)
        Current position (x,y)
        x: width
        y: height

    Actions:
        Type: Discrete(2)
        Num Observation
        0   up
        1   left
        2   down
        3   right

    Reward:
        -1  every move
        n//2+1 reaching the goal (expect reward converge to 0)

    Starting State:
        center of the gridworld

    Termination:
        four corners of the gridworld

    """

    def __init__(self):
        self.height = 7
        self.width = 7
        self.action_space = spaces.Discrete(4)

        high = np.array([self.width, self.height], dtype=int)

        self.observation_space = spaces.Box(0, high-1, shape=[2,],dtype=int)

        self.moves = {
            0: (0, -1),   # up
            1: (-1, 0),   # left
            2: (0, 1),  # down
            3: (1, 0),  # right
        }

        self.reset()

    def step(self, action):
        
        # move
        x, y = self.moves[action]
        self.state = self.state[0] + x, self.state[1] + y
        
        # boundaries
        self.state = max(0, self.state[0]), max(0, self.state[1])
        self.state = (min(self.state[0], self.width - 1),
                      min(self.state[1], self.height - 1))

        # reach the goal
        if self.state == (self.width - 1, self.height - 1) or self.state == (0,0) or self.state == (0, self.height - 1) or self.state == (self.width - 1, 0):
            return self.state, 5, True, {}
        
        # # fell off the cliff, state reset
        # elif self.state[0] != 0 and self.state[1] == self.height - 1:
        #     return self.reset(), -10, False, {}

        return self.state, -1, False, {}

    def reset(self):
        self.state = (self.width//2, self.height//2)
        return self.state


