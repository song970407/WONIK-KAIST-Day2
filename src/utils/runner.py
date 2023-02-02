from src.control.mpc import MPCFTC


class Runner:
    def __init__(self,
                 model,
                 state_dim,
                 action_dim,
                 receding_horizon,
                 smoothness_lambda,
                 decaying_gamma,
                 overshoot_weight,
                 from_target,
                 u_min,
                 u_max,
                 max_iter,
                 timeout,
                 is_logging,
                 device,
                 opt_config,
                 scheduler_config,
                 state_scaler,
                 action_scaler):
        self.MPC_solver = MPCFTC(model=model,
                                 state_dim=state_dim,
                                 action_dim=action_dim,
                                 receding_horizon=receding_horizon,
                                 smoothness_lambda=smoothness_lambda,
                                 decaying_gamma=decaying_gamma,
                                 overshoot_weight=overshoot_weight,
                                 from_target=from_target,
                                 u_min=u_min,
                                 u_max=u_max,
                                 max_iter=max_iter,
                                 timeout=timeout,
                                 is_logging=is_logging,
                                 device=device,
                                 opt_config=opt_config,
                                 scheduler_config=scheduler_config).to(device)
        self.state_scaler = state_scaler
        self.action_scaler = action_scaler

    def solve(self, hist_xs, hist_us, target_xs, u_min=None, u_max=None):
        """
        :param hist_xs: state_order x num_state
        :param hist_ys: (action_order-1) x num_action
        :param target_xs: H x num_state
        :return:
        """
        action, log = self.MPC_solver.solve(hist_xs, hist_us, target_xs, u_min, u_max)
        action = action[0:1] * (self.action_scaler[1] - self.action_scaler[0]) + self.action_scaler[0]
        return action, log
