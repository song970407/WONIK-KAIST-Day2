import os
import pickle

import torch

from torch.utils.data import DataLoader, TensorDataset
from src.utils.load_data import load_data
from src.model.get_model import get_multistep_linear_model
from src.utils.data_preprocessing import get_data

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def main(train_config):
    model_name = train_config['model_name']
    state_dim = train_config['state_dim']
    action_dim = train_config['action_dim']
    state_order = train_config['state_order']
    action_order = train_config['action_order']
    H = train_config['H']
    alpha = train_config['alpha']
    EPOCHS = train_config['EPOCHS']
    BS = train_config['BS']
    lr = train_config['lr']
    train_data_path = train_config['train_data_path']
    val_data_path = train_config['val_data_path']
    val_EVERY = 25
    SAVE_EVERY = 25

    # Prepare Model and Dataset
    m = get_multistep_linear_model(model_name, state_dim, action_dim, state_order, action_order).to(DEVICE)

    train_states, train_actions = load_data(path=train_data_path,
                                            state_order=state_order,
                                            action_order=action_order)

    # Set te minimum and maximum temperature as 20 and 420

    val_states, val_actions = load_data(path=val_data_path,
                                        state_order=state_order,
                                        action_order=action_order)

    train_history_xs, train_history_us, train_us, train_ys = get_data(states=train_states,
                                                                      actions=train_actions,
                                                                      rollout_window=H,
                                                                      history_x_window=state_order,
                                                                      history_u_window=action_order,
                                                                      device=DEVICE)

    train_history_xs = train_history_xs.transpose(1, 2)  # B * state_order * state_dim
    train_history_us = train_history_us.transpose(1, 2)  # B * (action_order - 1) * action_dim
    train_us = train_us.transpose(1, 2)  # B * rollout_window * action_dim
    train_ys = train_ys.transpose(1, 2)  # B * rollout_window * state_dim

    val_history_xs, val_history_us, val_us, val_ys = get_data(states=val_states,
                                                              actions=val_actions,
                                                              rollout_window=H,
                                                              history_x_window=state_order,
                                                              history_u_window=action_order,
                                                              device=DEVICE)
    val_history_xs = val_history_xs.transpose(1, 2)  # B * state_order * state_dim
    val_history_us = val_history_us.transpose(1, 2)  # B * (action_order - 1) * action_dim
    val_us = val_us.transpose(1, 2)  # B * rollout_window * action_dim
    val_ys = val_ys.transpose(1, 2)  # B * rollout_window * state_dim

    # Training Route Setting
    criteria = torch.nn.SmoothL1Loss()
    train_ds = TensorDataset(train_history_xs, train_history_us, train_us, train_ys)
    train_loader = DataLoader(train_ds, batch_size=BS, shuffle=True)
    val_criteria = torch.nn.SmoothL1Loss()

    opt = torch.optim.Adam(m.parameters(), lr=lr)
    iters = len(train_loader)
    num_params = sum(p.numel() for p in m.parameters())

    num_updates = 0
    best_val_loss = float('inf')

    model_save_path = 'saved_model/{}'.format(model_name)
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)
    with open('{}/train_config_{}_{}.txt'.format(model_save_path, state_order, action_order), 'wb') as f:
        pickle.dump(train_config, f)
    # Start Training
    print('Model: {} Training Start, State Order: {}, Action Order: {}'.format(model_name, state_order, action_order))
    for ep in range(EPOCHS):
        # if ep % 100 == 0:
        print("Epoch [{}] / [{}]".format(ep + 1, EPOCHS))
        for i, (x0, u0, u, y) in enumerate(train_loader):
            opt.zero_grad()
            y_predicted = m.rollout(x0, u0, u)
            loss_prediction = criteria(y_predicted, y)
            loss_regularizer = alpha * (sum(torch.norm(param, p=1) for param in m.parameters())) / num_params
            # loss_regularizer = alpha * (torch.mean(torch.abs(m.A.data)) + torch.mean(torch.abs(m.B.data)))
            loss = loss_prediction + loss_regularizer
            loss.backward()
            opt.step()
            num_updates += 1
            log_dict = {}
            log_dict['train_loss_prediction'] = loss_prediction.item()
            log_dict['train_loss_regularizer'] = loss_regularizer.item()
            log_dict['train_loss'] = loss.item()
            log_dict['lr'] = opt.param_groups[0]['lr']
            if num_updates % val_EVERY == 0:
                with torch.no_grad():
                    val_predicted_y = m.rollout(val_history_xs, val_history_us, val_us)
                    val_loss_prediction = val_criteria(val_predicted_y, val_ys)
                    val_loss_regularizer = alpha * (
                        sum(torch.norm(param, p=1) for param in m.parameters())) / num_params
                    val_loss = val_loss_prediction + val_loss_regularizer
                    log_dict['val_loss_prediction'] = val_loss_prediction.item()
                    log_dict['val_loss_regularizer'] = val_loss_regularizer.item()
                    log_dict['val_loss'] = val_loss.item()
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(m.state_dict(),
                               '{}/best_model_{}_{}.pt'.format(model_save_path, state_order, action_order))
                    print('Best model saved at iteration {}, loss: {}'.format(num_updates, best_val_loss))
                log_dict['best_val_loss'] = best_val_loss
            if num_updates % SAVE_EVERY == 0:
                torch.save(m.state_dict(), '{}/curr_model_{}_{}.pt'.format(model_save_path, state_order, action_order))
    print('train finished')
    print(num_updates)


if __name__ == '__main__':
    state_orders = [1, 5, 10]
    action_orders = [5, 10, 20]
    for state_order in state_orders:
        for action_order in action_orders:
            train_config = {
                'model_name': 'multistep_linear_res2',
                'state_dim': 5,
                'action_dim': 3,
                'state_order': state_order,
                'action_order': action_order,
                'H': 7,
                'alpha': 0.0,
                'EPOCHS': 500,
                'BS': 128,
                'lr': 1e-4,
                'train_data_path': 'data/train_data.pkl',
                'val_data_path': 'data/val_data.pkl'
            }
            main(train_config)
