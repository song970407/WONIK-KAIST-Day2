import numpy as np
import torch
import torch.nn as nn
import torch.optim as th_opt


def get_discount_factor(horizon_length, gamma):
    g = 1.0
    gs = [g]
    for i in range(1, horizon_length):
        g *= gamma
        gs.append(g)
    return torch.tensor(gs)


def get_default_opt_config():
    opt_config = {'lr': 1e-2}
    return opt_config


class MPC(nn.Module):
    def __init__(self,
                 model: nn.Module,
                 action_dim: int,
                 receding_horizon: int,
                 action_min: float = -1.0,
                 action_max: float = 1.0,
                 gamma: float = 1.0,
                 max_iter: int = 200,
                 is_logging: bool = True,
                 device: str = 'cpu',
                 opt_config: dict = None):
        """
        :param model: (nn.Module) nn-parametrized dynamic model
        :param action_dim: (int) dimension of each action, usually 1
        :param receding_horizon: (int) receding horizon
        :param action_min: (float) the lower bound of action values
        :param action_max: (float) the upper bound of action values
        :param gamma: (float) time discount factor
        :param max_iter: (int) the maximum iteration for MPC optimization problem
        :param is_logging:
        :param device: (str) The computation device that is used for the MPC optimization
        :param opt_config: (dict)

        """
        super(MPC, self).__init__()

        self.model = model.to(device)
        self.action_dim = action_dim
        self.receding_horizon = receding_horizon
        self.action_min = action_min
        self.action_max = action_max
        self.gamma = gamma
        self.max_iter = max_iter
        self.is_logging = is_logging

        opt_config = get_default_opt_config() if opt_config is None else opt_config
        self.opt_config = opt_config

        self.criteria = torch.nn.SmoothL1Loss()

        if device is None:
            print("Running device is not given. Infer the running device from the system configuration ...")
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print("Initializing solver with {}".format(device))
        self.device = device
        # Initialize Control Variable
        self.actions = torch.empty(size=(self.receding_horizon, self.action_dim)).to(self.device)
        nn.init.uniform_(self.actions, a=action_min, b=action_max)

    def compute_loss(self, x0, u0, us, y):
        y_predicted = self.model.multi_step_prediction(x0=x0, u0=u0, us=us)
        return y_predicted, self.criteria(y_predicted, y)

    @staticmethod
    def record_log(observation, loss, actions, ys, best_idx, best_loss, best_action):
        log = {
            'observation': observation,
            'loss': loss,
            'actions': actions,
            'ys': ys,
            'best_idx': best_idx,
            'best_loss': best_loss,
            'best_action': best_action
        }
        return log

    def solve(self, x0, u0, y):
        """
        :param observation: obs_dim
        :return:
        """
        x0 = torch.from_numpy(x0).float().to(self.device)
        u0 = torch.from_numpy(u0).float().to(self.device)
        y = torch.from_numpy(y).float().to(self.device)

        loss_trajectory = []  # the loss over the optimization steps
        actions_trajectory = []  # the optimized actions over the optimization steps
        ys_trajectory = []

        actions = torch.nn.Parameter(self.actions).to(self.device)
        actions.data = actions.data.clamp(min=self.action_min, max=self.action_max)

        opt = th_opt.Adam([actions], **self.opt_config)
        for i in range(self.max_iter):
            ys, loss = self.compute_loss(x0=x0, u0=u0, us=actions, y=y)
            loss_trajectory.append(loss.item())
            actions_trajectory.append(actions.data.cpu().detach().numpy())
            ys_trajectory.append(ys.cpu().detach().numpy())
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                actions.data = actions.data.clamp(min=self.action_min, max=self.action_max)
        with torch.no_grad():
            ys, loss = self.compute_loss(x0=x0, u0=u0, us=actions, y=y)
            loss_trajectory.append(loss.item())
            actions_trajectory.append(actions.data.cpu().detach().numpy())
            ys_trajectory.append(ys.cpu().detach().numpy())
        # Return the best us
        idx = np.argmin(loss_trajectory)
        optimal_action = actions_trajectory[idx]
        if self.is_logging:
            log = self.record_log(observation=x0.cpu().detach().numpy(),
                                  loss=np.array(loss_trajectory),
                                  actions=np.array(actions_trajectory),
                                  ys=np.array(ys_trajectory),
                                  best_idx=idx,
                                  best_loss=loss_trajectory[idx],
                                  best_action=actions_trajectory[idx]
                                  )
        else:
            log = None
        return optimal_action, log
