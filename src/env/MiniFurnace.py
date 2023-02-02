import numpy as np
import torch
import torch.nn as nn
import pandas as pd


class MiniFurnace(nn.Module):
    def __init__(self):
        super(MiniFurnace, self).__init__()

        self.observation_space = {'low': -2, 'high': 2}
        self.action_space = {'low': -1, 'high': 1}  # [-1, 1]

        self.state_dim, self.action_dim = 5, 3

        self.state_order = 5
        self.action_order = 10

        self.initial_history = 30

        self.setting_parameter()

        self.past_xs = np.zeros((self.initial_history, self.state_dim))
        self.past_us = np.zeros((self.initial_history, self.action_dim))

        self.gamma = 1.0

    def setting_parameter(self):
        np.random.seed(0)
        A, B = [], []

        for i in range(self.state_order - 1):
            A.append(np.random.rand(self.state_dim, self.state_dim) * 0.05 - 0.025)
        A0 = np.ones((self.state_dim, self.state_dim)) * 0.025
        np.fill_diagonal(A0, 0.9)
        A.append(A0)

        for j in range(self.action_order):
            B.append(np.random.rand(self.action_dim, self.state_dim) * 0.05 - 0.025)

        self.A = np.stack(A)
        self.B = np.stack(B)
        self.C = np.random.rand(self.state_dim) * 0.05 - 0.025

    def step(self, u):  # u has dimension with (1, self.action_dim)
        # check dimension of u
        if u.shape != (1, self.action_dim):
            raise Exception('action dimension is WRONG')

        self.past_us = np.concatenate((self.past_us, u), axis=0)[1:, ]
        # new_x = self.forward(x=self.current_xs, u=self.current_us)
        ar_terms = np.einsum('ox, oxy -> y', self.previous_State(), self.A)
        exo_terms = np.einsum('ox, oxy -> y', self.previous_Action(), self.B)
        new_x = (ar_terms + exo_terms + self.C).reshape(1, -1)
        self.past_xs = np.concatenate((self.past_xs, new_x), axis=0)[1:]
        return new_x

    def reset(self):
        self.past_xs = np.zeros((self.initial_history, self.state_dim))
        self.past_us = np.zeros((self.initial_history - 1, self.action_dim))

        return self.past_xs, self.past_us

    def get_obs(self):  # get state at timestep t
        return self.past_xs[-1, :].reshape(1, -1)

    def get_action(self):
        return self.past_us[-1, :].reshape(1, -1)

    def previous_Action(self):  # extract previous action
        return self.past_us[-self.action_order:, :]

    def previous_State(self):  # extract previous state
        return self.past_xs[-self.state_order:, :]


if __name__ == '__main__':
    env = MiniFurnace()
    for i in range(10):
        state = env.step(np.ones((1, 3)))
        print(state)
    b = 1
