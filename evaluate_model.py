import numpy as np
import torch
import matplotlib.pyplot as plt

from src.model.get_model import get_multistep_linear_model
from src.utils.load_data import load_data
from src.utils.data_preprocessing import get_data

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)


def main(evaluate_config):
    model_name = evaluate_config['model_name']
    state_dim = evaluate_config['state_dim']
    action_dim = evaluate_config['action_dim']
    state_order = evaluate_config['state_order']
    action_order = evaluate_config['action_order']
    test_data_paths = evaluate_config['test_data_paths']
    rollout_window = evaluate_config['rollout_window']
    saved_model_path = 'saved_model/{}/best_model_{}_{}.pt'.format(model_name, state_order, action_order)
    m = get_multistep_linear_model(model_name, state_dim, action_dim, state_order, action_order, saved_model_path)
    m.eval().to(device)
    test_states, test_actions = load_data(path=test_data_paths,
                                          state_order=state_order,
                                          action_order=action_order)

    history_xs, history_us, us, ys = get_data(states=test_states,
                                              actions=test_actions,
                                              rollout_window=rollout_window,
                                              history_x_window=state_order,
                                              history_u_window=action_order,
                                              device=device)
    history_xs = history_xs.transpose(1, 2)
    ys = ys.transpose(1, 2)
    history_us = history_us.transpose(1, 2)
    us = us.transpose(1, 2)

    def loss_fn(y_predicted, y):
        criteria_ = torch.nn.SmoothL1Loss(reduction='none')
        loss = criteria_(y_predicted, y)
        return loss.mean()

    with torch.no_grad():
        predicted_ys = m.rollout(history_xs, history_us, us)
        loss = loss_fn(ys, predicted_ys)
        print('({}, {}), Loss: {}'.format(state_order, action_order, loss))
        ys = ys.cpu().detach().numpy()
        predicted_ys = predicted_ys.cpu().detach().numpy()
    return ys, predicted_ys, loss.item()


if __name__ == '__main__':
    # Specify which model you want to evaluate
    model_name = 'multistep_linear_res2'
    # Test several combinations of (state_order, action_order) and find the best model
    state_orders = [1]
    action_orders = [1]
    results = []
    loss_lists = []
    for state_order in state_orders:
        for action_order in action_orders:
            evaluate_config = {
                'model_name': model_name,
                'state_dim': 5,
                'action_dim': 3,
                'state_order': state_order,
                'action_order': action_order,
                'test_data_paths': 'data/test_data.pkl',
                'rollout_window': 90
            }
            res = main(evaluate_config)

            # plot figure
            fig, axes = plt.subplots(1, 2, figsize=(10, 7))
            axes_flatten = axes.flatten()
            axes_flatten[0].plot(res[0][0])
            axes_flatten[0].set_title('True')
            axes_flatten[0].set_ylim([-1, 1])
            axes_flatten[1].plot(res[1][0])
            axes_flatten[1].set_title('Predicted')
            axes_flatten[1].set_ylim([-1, 1])
            fig.suptitle('Model: ({}, {}), Loss: {}'.format(state_order, action_order, res[2]))

            fig.show()
            fig.savefig('evaluate_result/{}_{}.png'.format(state_order, action_order))
            # results.append(res[1])
            loss_lists.append(res[2])
    ys = res[0]
    nrows = len(state_orders)
    ncols = len(action_orders)
    loss_lists = np.array(loss_lists)
    arg_loss_lists = np.argsort(loss_lists)
    num_plots = 5  # have to change value
    for j in range(num_plots):
        print('{}-th Model: ({}, {})'.format(j + 1, state_orders[arg_loss_lists[j] // len(action_orders)],
                                             action_orders[arg_loss_lists[j] % len(action_orders)]))
    loss_lists = np.reshape(loss_lists, (len(state_orders), len(action_orders)))
    plt.imshow(loss_lists)
    plt.colorbar()
    plt.xlabel('Action Order')
    plt.ylabel('State Order')
    plt.show()
