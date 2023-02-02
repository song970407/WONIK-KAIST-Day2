import copy
import os
import pickle
from time import perf_counter

import numpy as np
import torch
import yaml

from src.model.get_model import get_multistep_linear_model
from src.utils.runner import Runner
from src.utils.load_data import load_data
from src.utils.manage_history import HistoryManager
from src.utils.reference_generator import generate_reference
from src.utils.test_utils import set_seed

from ips.ControlPlc import ControlPlc
from ips.TraceLog import TraceLog, setFilename
import time
from datetime import datetime

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
set_seed(0, torch.cuda.is_available())


def saveParameter(control_dict):
    dir_path = 'parameter_log'
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    with open('{}//{}.yaml'.format(dir_path, datetime.now().strftime("%Y%m%d_%H%M%S")), 'w') as f:
        yaml.dump(control_dict, f)


def main(control_dict, env_dict=None):
    # IPS Code
    plc = ControlPlc()
    plc.connect_plc()

    step_log = TraceLog()

    running = False
    stop = False
    bStart = False

    """
        Step 0. Prepare MPC
    """
    # Call the variable from control_dict
    target_temps = control_dict['target_temps']
    target_times = control_dict['target_times']

    model_name = control_dict['model_name']
    state_dim = control_dict['state_dim']
    action_dim = control_dict['action_dim']
    state_order = control_dict['state_order']
    action_order = control_dict['action_order']
    training_H = control_dict['training_H']
    training_alpha = control_dict['training_alpha']
    state_scaler = control_dict['state_scaler']
    action_scaler = control_dict['action_scaler']

    receding_horizon = control_dict['receding_horizon']
    smoothness_lambda = control_dict['smoothness_lambda']
    decaying_gamma = control_dict['decaying_gamma']
    overshoot_weight = control_dict['overshoot_weight']
    from_target = control_dict['from_target']
    u_min = control_dict['u_min']
    u_max = control_dict['u_max']
    anneal_min = control_dict['anneal_min']
    anneal_max = control_dict['anneal_max']
    max_iter = control_dict['max_iter']
    timeout = control_dict['timeout']
    is_logging = control_dict['is_logging']
    opt_config = control_dict['opt_config']
    scheduler_config = control_dict['scheduler_config']

    saved_model_path = 'saved_model/{}/model_{}_{}_{}_{}.pt'.format(model_name, state_order, action_order, training_H,
                                                                    training_alpha)
    model = get_multistep_linear_model(model_name, state_dim, action_dim, state_order, action_order, saved_model_path)

    # Real Experiment Case
    hist_manager = HistoryManager(x_order=state_order,
                                  u_order=action_order - 1,
                                  x_scaler=state_scaler,
                                  u_scaler=action_scaler)

    u_min_list = generate_reference(target_temps=[u_min, u_min, anneal_min, anneal_min], target_times=target_times)
    u_max_list = generate_reference(target_temps=[u_max, u_max, anneal_max, anneal_max], target_times=target_times)
    target_xs = generate_reference(target_temps=target_temps, target_times=target_times)
    target_xs = torch.reshape(torch.tensor(target_xs).to(device), shape=(-1, 1)).repeat(repeats=(1, state_dim))
    scaled_target_xs = (target_xs - state_scaler[0]) / (state_scaler[1] - state_scaler[0])

    step = 0
    start_elapsed = 0
    iter_start = 0
    before_step = ''
    heatup_step = '375H'  # 375
    stable_step = '375S'  # 375
    anneal_step = '375A'  # 375

    trajectory_log = []
    trajectory_xs = []
    trajectory_us = []

    step_log.write('Model {}'.format(saved_model_path))
    step_log.write('Start FTC')

    while True:
        if not bStart and plc.isProcessing():
            iter_start = 0
            bStart = True
        global runner_state_
        runner_state_ = running
        stepName = plc.get_step_name()

        # Enter if not processing or stop
        if not plc.isProcessing() or stop:
            # Enter if running is True, save log file
            if running:
                step = 0
                start_elapsed = 0
                iter_start = 0
                running = False
                bStart = False
                before_step = ''
                step_log.write('runner End')

                """
                    Step 4. Save the Data
                """
                # Save whole value when runner finished
                cur_time = setFilename()
                save_dir = 'log/{}'.format(cur_time)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                with open(save_dir + '/mpc.log', 'wb') as f:
                    pickle.dump(trajectory_log, f)
                with open(save_dir + '/mpc_setting.dict', 'wb') as f:
                    pickle.dump(control_dict, f)
                trajectory_tc = np.stack(trajectory_xs, axis=0)
                trajectory_ws = np.stack(trajectory_us, axis=0)
                np.save(save_dir + '/trajectory_tc.npy', trajectory_tc)  # xs
                np.save(save_dir + '/trajectory_ws.npy', trajectory_ws)  # us

                saveParameter(control_dict)

                trajectory_log = []
                trajectory_xs = []
                trajectory_us = []

                step_log.write('logging End')
            stop = False
            continue
        # Enter if the running starts
        elif not running and stepName == heatup_step:
            runner = Runner(model=model,
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
                             opt_config=copy.copy(opt_config),
                             scheduler_config=copy.copy(scheduler_config),
                             state_scaler=state_scaler,
                             action_scaler=action_scaler)

            step_log.write('runner Start {}'.format(plc.get_recipe_name()))
            running = True
            iter_mid = perf_counter()
            start_elapsed = iter_mid - iter_start
            iter_start = perf_counter()
            continue
        elif running:
            step_log.write('RUNNING IN')
            iter_Running = perf_counter()
            print(iter_Running - iter_start)
            if stepName != heatup_step and stepName != stable_step and stepName != anneal_step:
                plc.reset_heater()
                stop = True
                step_log.write('Anneal End')
                continue

            stepTime = plc.get_step_time()
            stepName = plc.get_step_name()
            if stepName == stable_step:
                step_time = stepTime + int(heatup150 * 5)
            elif stepName == anneal_step:
                step_time = stepTime + int(heatup150 * 5) + int(stable150 * 5)
            else:
                step_time = stepTime  # steptime calculation

            quot = int(step_time // 5)
            step = quot

            if abs(step - before_step) > 10:
                step_log.write('SKIP {}'.format(step))
                before_step = step
                iter_start = perf_counter()
                continue

            scaled_hist_xs, scaled_hist_us = hist_manager(device)
            action, log = runner.solve(scaled_hist_xs, scaled_hist_us,
                                       scaled_target_xs[step:step + receding_horizon, :],
                                       u_min_list[step:step + receding_horizon],
                                       u_max_list[step:step + receding_horizon])

            trajectory_log.append(log)
            print(len(trajectory_log))

            workset = action.cpu().detach().numpy()  # [1 x 40] numpy.array
            wsvalue = workset.reshape(40, 1)

            bCheck = np.isnan(wsvalue).any()
            if bCheck:
                continue
            else:
                plc.set_heater(wsvalue)

                step_log.write(
                    'PC TIME : {}, CX TIME : {}, StepTime : {} PLC StepTime : {}, Step : {}, StepName : {}'.format(
                        plc.get_pc_time(),
                        float(plc.get_cx_time()), step_time, stepTime, step, stepName))


        """
        Step 3. Append the data to the HistoryManger instance
        """
        np_tc = np.array(plc.get_glass_tc(), dtype='float32')
        np_workset = np.array(plc.get_heater_sp(), dtype='float32')[:action_dim]
        observed_tc = torch.from_numpy(np_tc).view(-1)  # from_numpy = change numpy / torch.tensor = not change numpy
        workset = torch.from_numpy(np_workset).view(-1)

        hist_manager.append(observed_tc, workset)
        trajectory_xs.append(np_tc)
        trajectory_us.append(np_workset)

        before_step = step

        iter_end = perf_counter()
        elapsed = iter_end - iter_start + start_elapsed
        print('iter_end {}'.format(iter_end))
        print('iter_start {}'.format(iter_start))
        start_elapsed = 0
        wait_for = max(5.0000000000000000000000 - elapsed, 0)
        time.sleep(wait_for)
        iter_start = perf_counter()


if __name__ == '__main__':
    # Model Type
    model_name = 'multistep_linear_res2'
    state_order = 5
    # state_order = 10
    action_order = 50
    training_H = 200
    training_alpha = 0.0
    # training_alpha = 10.0
    train_dict_path = 'saved_model/{}/train_dict_{}_{}_{}_{}.txt'.format(model_name, state_order, action_order,
                                                                         training_H, training_alpha)

    with open(train_dict_path, 'rb') as f:
        train_dict = pickle.load(f)

    # Hyper-parameters for MPC optimizer
    receding_horizon = 100  # Receding horizon
    smoothness_lambda = 1000.0  # 10 ok, 100 ok, 1000
    # smoothness_lambda = 1.0
    decaying_gamma = 0.993  # time-decaying, originally: 1.0
    overshoot_weight = 1.0  # objective weighting, originally: 1.0
    from_target = True  # Check whether the action is residual between the workset and target or not
    max_iter = 50  # Maximum number of optimizer iterations
    u_range = 0.03  # will be ignored if smooth_u_type == penalty, cannot be list
    #anneal_range = u_range / 2
    anneal_range = u_range / 3
    if from_target:
        u_min = - u_range
        u_max = anneal_range
        #u_max = u_range
        anneal_min = - 0.75*anneal_range
        #anneal_min = - anneal_range
        anneal_max = anneal_range
    else:
        u_min = 0.0
        u_max = 1.0
    timeout = 5.0000000000000000000000
    is_logging = True

    # Default opt_config and scheduler_config
    opt_config = {
        'name': 'Adam',
        'lr': 5e-3
        # 'lr': 1e-1
    }
    scheduler_config = {
        'name': 'ReduceLROnPlateau',
        'patience': 2,
        'factor': 0.5,
        'min_lr': 1e-4
        # 'min_lr': 5e-4
    }

    # Generate Target Trajectory
    #heatup150 = 240 20M
    heatup150 = 540
    stable150 = 180
    anneal150 = 180

    target_temps = [159.0, 375.0, 375.0, 375.0]
    target_times = [heatup150, stable150, anneal150 + receding_horizon]

    control_dict = {
        'target_temps': target_temps,
        'target_times': target_times,
        'model_name': model_name,
        'state_dim': train_dict['state_dim'],
        'action_dim': train_dict['action_dim'],
        'state_order': state_order,
        'action_order': action_order,
        'training_H': training_H,
        'training_alpha': training_alpha,
        'state_scaler': train_dict['state_scaler'],
        'action_scaler': train_dict['action_scaler'],
        'receding_horizon': receding_horizon,
        'smoothness_lambda': smoothness_lambda,
        'decaying_gamma': decaying_gamma,
        'overshoot_weight': overshoot_weight,
        'from_target': from_target,
        'u_min': u_min,
        'u_max': u_max,
        'anneal_min': anneal_min,
        'anneal_max': anneal_max,
        'max_iter': max_iter,
        'timeout': timeout,
        'is_logging': is_logging,
        'opt_config': opt_config,
        'scheduler_config': scheduler_config,
        'u_range': u_range,
        'anneal_range': anneal_range
    }

    #saveParameter(control_dict)
    main(control_dict)
