import random

import matplotlib.pyplot as plt
import numpy as np
import torch


def set_seed(seed: int,
             use_cuda: bool):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def inspect_log(logs,
                n_xs: int = 140,
                n_us: int = 40,
                n_xcols: int = 8,
                n_xrows: int = 8):
    best_losses = [log['best_loss'] for log in logs]
    xs = np.stack([log['hist_xs'][-1, :] for log in logs])
    target_xs = np.stack([log['target_xs'][0, :] for log in logs])
    exerted_us = np.stack([log['best_us'][0, :] for log in logs])

    plt.plot(best_losses)

    n_cols = n_xcols
    n_rows = n_xrows
    n_figs = n_cols * n_rows
    viz_idx = sorted(np.random.choice(n_xs, n_figs, replace=False))

    fig, axes = plt.subplots(n_rows, n_cols, sharex=True,
                             dpi=500,
                             figsize=(n_cols * 1.25, n_rows))
    axes = axes.flatten()
    for ax_i, viz_i in enumerate(viz_idx):
        ax = axes[ax_i]
        ax.plot(xs[:, viz_i], label='xs')
        ax.plot(target_xs[:, viz_i], label='xs')
        #     ax.legend()
        ax.set_title('State {}'.format(viz_i))
    fig.tight_layout()

    n_cols = 8
    n_rows = 5
    n_figs = n_cols * n_rows
    viz_idx = sorted(np.random.choice(n_us, n_figs, replace=False))

    fig, axes = plt.subplots(n_rows, n_cols, sharex=True,
                             dpi=500,
                             figsize=(n_cols * 1.25, n_rows))
    axes = axes.flatten()
    for ax_i, viz_i in enumerate(viz_idx):
        ax = axes[ax_i]
        ax.plot(exerted_us[:, viz_i], label='xs')
        ax.set_title('Action {}'.format(viz_i))
    fig.tight_layout()
