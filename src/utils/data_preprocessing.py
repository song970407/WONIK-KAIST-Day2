from typing import Union, List

import torch


def get_data(states: Union[torch.Tensor, List[torch.Tensor]],
             actions: Union[torch.Tensor, List[torch.Tensor]],
             rollout_window: int,
             history_x_window: int,
             history_u_window: int,
             device: str = 'cpu'):
    if isinstance(states, torch.Tensor):
        states = [states]
        actions = [actions]

    history_xs = []
    history_us = []
    us = []
    ys = []
    for state, action in zip(states, actions):
        num_obs = action.shape[0]
        for i in range(num_obs - rollout_window - max(history_x_window, history_u_window) + 2):
            history_xs.append(state[i:i + history_x_window, :])
            history_us.append(action[i:i + history_u_window - 1, :])
            us.append(action[i + history_u_window - 1:i + history_u_window + rollout_window - 1, :])
            ys.append(state[i + history_x_window:i + history_x_window + rollout_window, :])

    history_xs = torch.stack(history_xs).transpose(1, 2).to(device)  # B * state_dim * state_order
    history_us = torch.stack(history_us).transpose(1, 2).to(device)  # B * action_dim * (action_order - 1)
    us = torch.stack(us).transpose(1, 2).to(device)  # B * action_dim * rollout_window
    ys = torch.stack(ys).transpose(1, 2).to(device)  # B * state_dim * rollout_window

    return history_xs, history_us, us, ys
