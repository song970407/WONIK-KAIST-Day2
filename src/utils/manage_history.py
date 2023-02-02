from typing import List

import torch


class HistoryManager:

    def __init__(self,
                 x_order: int,
                 u_order: int,
                 x_scaler: List[float],
                 u_scaler: List[float],
                 init_hist_xs: List[torch.tensor] = [],  # from the oldest to newest
                 init_hist_us: List[torch.tensor] = []  # from the oldest to newest
                 ):
        assert len(init_hist_xs) == x_order
        assert len(init_hist_us) == u_order
        assert isinstance(init_hist_xs, list)
        assert isinstance(init_hist_us, list)

        self.x_order = x_order
        self.hist_xs = init_hist_xs
        self.x_scaler = x_scaler

        self.u_order = u_order
        self.hist_us = init_hist_us
        self.u_scaler = u_scaler

    def append(self, xs, us):
        self.hist_xs.append(xs)
        if len(self.hist_xs) > self.x_order:
            self.hist_xs.pop(0)  # remove the oldest states

        self.hist_us.append(us)
        if len(self.hist_us) > self.u_order:
            self.hist_us.pop(0)

    def __call__(self, device):
        hist_xs = torch.stack(self.hist_xs).to(device)
        scaled_hist_xs = (hist_xs - self.x_scaler[0]) / (self.x_scaler[1] - self.x_scaler[0])
        hist_us = torch.stack(self.hist_us).to(device)
        scaled_hist_us = (hist_us - self.u_scaler[0]) / (self.u_scaler[1] - self.u_scaler[0])
        return scaled_hist_xs, scaled_hist_us
