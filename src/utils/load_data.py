import re
import pickle
from typing import Union, List

import pandas as pd
import torch


def load_data(path: str,
              state_order,
              action_order,
              initial_history: int = 30,
              device: str = 'cpu'):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    num_trajectory = len(data['traj_obs'])
    states = []
    actions = []

    for i in range(num_trajectory):
        state = torch.tensor(data['traj_obs'][i]).float().to(device)
        action = torch.tensor(data['traj_action'][i]).float().to(device)

        state = state[initial_history - state_order:, :]
        action = action[initial_history - action_order:, :]

        states.append(state)
        actions.append(action)

    return states, actions
