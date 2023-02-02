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


class MPCFTC(nn.Module):
    def __init__(self,
                 model: nn.Module,
                 state_dim: int = 140,
                 action_dim: int = 40,
                 receding_horizon: int = 100,
                 smoothness_lambda: float = 0.0,
                 decaying_gamma: float = 1.0,
                 overshoot_weight: float = 1.0,
                 from_target: bool = False,
                 u_min: float = 0.0,
                 u_max: float = 1.0,
                 max_iter: int = 200,
                 timeout: float = 300,  # sections
                 is_logging: bool = True,
                 device: str = 'cpu',
                 opt_config: dict = {},
                 scheduler_config: dict = {}):
        """
        :param model: (nn.Module) nn-parametrized dynamic model
        :param state_dim: (int) dimension of states
        :param action_dim: (int) dimension of actions
        :param receding_horizon: (int) receding horizon
        :param smoothness_lambda: (float) coefficient for smoothness regularization
        :param from_target: (bool) default = True
            if True, action bound will be set to the +/- from the target temperatures
            if False, will be ignored
        :param u_min: (float) the lower bound of action values
        :param u_max: (float) the upper bound of action values
        :param max_iter: (int) the maximum iteration for MPC optimization problem
        :param timeout: (float) MPC optimization time budget (unit: seconds)
        :param is_logging:
        :param device: (str) The computation device that is used for the MPC optimization
        :param opt_config: (dict)
        :param scheduler_config: (dict)
        """
        super(MPCFTC, self).__init__()

        self.model = model.to(device)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.receding_horizon = receding_horizon
        self.smoothness_lambda = smoothness_lambda
        self.decaying_gamma = decaying_gamma
        self.overshoot_weight = overshoot_weight
        self.from_target = from_target
        self.u_min = u_min
        self.u_max = u_max
        self.max_iter = max_iter
        self.timeout = timeout
        self.is_logging = is_logging

        opt_config = get_default_opt_config() if opt_config is None else opt_config
        self.optimizer = opt_config.pop('name')
        self.opt_config = opt_config

        scheduler_config = get_default_scheduler_config() if scheduler_config is None else scheduler_config
        self.scheduler = scheduler_config.pop('name')
        self.scheduler_config = scheduler_config

        if device is None:
            print("Running device is not given. Infer the running device from the system configuration ...")
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            print("Initializing solver with {}".format(device))
        self.device = device
        # Initialize Control Variable

        us = torch.empty(size=(self.receding_horizon, self.action_dim), device=device)

        # nn.init.uniform_(us, u_min, u_max)
        # nn.init.constant_(us, (self.u_max - self.u_min)/2)
        nn.init.constant_(us, self.u_min)

        self.us = torch.nn.Parameter(us)
        self.clamp_us()
        self.criteria_regularizer = torch.nn.MSELoss(reduction='mean')

    def criteria_objective(self, pred_xs, target_xs):
        receding_horizon = target_xs.shape[0]
        discount_factor = get_discount_factor(receding_horizon, self.decaying_gamma).to(self.device)
        overshoot_loss = torch.dot(discount_factor,
                                   torch.square(nn.functional.relu(pred_xs - target_xs)).mean(dim=1)) / receding_horizon
        undershoot_loss = torch.dot(discount_factor, torch.square(nn.functional.relu(target_xs - pred_xs)).mean(
            dim=1)) / receding_horizon
        return self.overshoot_weight * overshoot_loss + undershoot_loss

    def clamp_us(self, u_min=None, u_max=None):
        if u_min is None:
            self.us.data = self.us.data.clamp(min=self.u_min)
        else:
            for i in range(len(u_min)):
                self.us.data[i] = self.us.data[i].clamp(min=u_min[i])
        if u_max is None:
            self.us.data = self.us.data.clamp(max=self.u_max)
        else:
            for i in range(len(u_min)):
                self.us.data[i] = self.us.data[i].clamp(max=u_max[i])

    def compute_loss(self, hist_xs, hist_us, target_xs, us=None):
        if us == None:
            us = self.us
        if self.from_target:
            """
            setting action min/max bound dynamically
            The rationale is "setting an appropriate bounds for the optimization variables, (i.e., actions),
            are necessary, in practice, to derive the solution of optimization become favorable.

            The trick made on here is to set the action min/max based depending on the current targets.                          
            """
            us = us + target_xs[:, :self.action_dim]
        pred_xs = self.predict_future(hist_xs, hist_us, us)
        loss_objective = self.criteria_objective(pred_xs, target_xs)
        prev_us = torch.cat([hist_us[-1:], us[:-1]], dim=0)
        loss_regularizer = self.smoothness_lambda * self.criteria_regularizer(us, prev_us)
        loss = loss_objective + loss_regularizer
        return loss_objective, loss_regularizer, loss, us.detach()

    @staticmethod
    def record_log(hist_xs, hist_us, target_xs,
                   loss_objective, loss_regularizer, loss, us, lr, time, total_runtime,
                   best_idx, best_loss, best_us):
        log = {
            'hist_xs': hist_xs,
            'hist_us': hist_us,
            'target_xs': target_xs,
            'loss_objective': loss_objective,
            'loss_regularizer': loss_regularizer,
            'loss': loss,
            'us': us,
            'lr': lr,
            'time': time,
            'total_runtime': total_runtime,
            'best_idx': best_idx,
            'best_loss': best_loss,
            'best_us': best_us
        }
        return log

    def solve(self, hist_xs, hist_us, target_xs, u_min=None, u_max=None):
        """
        :param hist_xs: [state_order x state dim]
        :param hist_us: [action_order-1 x action dim]
        :param target_xs: [receding_horizon x state dim]
        :return:
        """
        loop_start = perf_counter()
        opt = getattr(th_opt, self.optimizer)([self.us], **self.opt_config)
        scheduler = getattr(th_opt.lr_scheduler, self.scheduler)(opt, **self.scheduler_config)

        loss_objective_trajectory = []  # the loss_objective over the optimization steps
        loss_regularizer_trajectory = []  # the loss_regularizer over the optimization steps
        loss_trajectory = []  # the loss over the optimization steps
        us_trajectory = []  # the optimized actions over the optimization steps
        time_trajectory = []  # the optimization runtime over the optimization steps
        lr_trajectory = []

        with stopit.ThreadingTimeout(self.timeout) as to_ctx_mgr:
            is_break = False
            prev_loss = float('inf')
            for i in range(self.max_iter):
                iter_start = perf_counter()
                loss_objective, loss_regularizer, loss, us = self.compute_loss(hist_xs, hist_us, target_xs)
                loss_objective_trajectory.append(loss_objective.item())
                loss_regularizer_trajectory.append(loss_regularizer.item())
                loss_trajectory.append(loss.item())
                us_trajectory.append(us.cpu())
                lr_trajectory.append(opt.param_groups[0]['lr'])
                # Update us
                opt.zero_grad()
                loss.backward()
                opt.step()
                scheduler.step(loss)
                self.clamp_us(u_min=u_min, u_max=u_max)
                # Save the loss and us
                time_trajectory.append(perf_counter() - iter_start)
            with torch.no_grad():
                iter_start = perf_counter()
                loss_objective, loss_regularizer, loss, us = self.compute_loss(hist_xs, hist_us, target_xs)
                lr_trajectory.append(opt.param_groups[0]['lr'])
                loss_objective_trajectory.append(loss_objective.item())
                loss_regularizer_trajectory.append(loss_regularizer.item())
                loss_trajectory.append(loss.item())
                us_trajectory.append(us.cpu())
                time_trajectory.append(perf_counter() - iter_start)
        self.opt_config['lr'] = opt.param_groups[0]['lr']
        if to_ctx_mgr.state == to_ctx_mgr.TIMED_OUT:
            self.clamp_us(u_min=u_min,
                          u_max=u_max)  # To assure the optimized actions are always satisfying the constraints.

        # Return the best us
        idx = np.argmin(loss_trajectory)
        optimal_u = us_trajectory[idx].to(self.device)
        if self.is_logging:
            log = self.record_log(hist_xs=hist_xs.cpu().numpy(),
                                  hist_us=hist_us.cpu().numpy(),
                                  target_xs=target_xs.cpu().numpy(),
                                  loss_objective=np.array(loss_objective_trajectory),
                                  loss_regularizer=np.array(loss_regularizer_trajectory),
                                  loss=np.array(loss_trajectory),
                                  us=torch.stack(us_trajectory).cpu().numpy(),
                                  lr=np.array(lr_trajectory),
                                  time=np.array(time_trajectory),
                                  total_runtime=perf_counter() - loop_start,
                                  best_idx=idx,
                                  best_loss=loss_trajectory[idx],
                                  best_us=us_trajectory[np.argmin(loss_trajectory)].cpu().numpy()
                                  )
        else:
            log = None
        return optimal_u, log

    def predict_future(self, hist_xs, hist_us, us):
        prediction = self.model.multi_step_prediction(hist_xs, hist_us, us)
        return prediction