import os.path as osp
upper_path = osp.abspath(osp.join(osp.abspath('.'),'..'))
import sys
sys.path.insert(1, upper_path)

from cem.cem import CEMPolicy
from experiments.planet.train import update_env_kwargs
from cem.visualize_cem import cem_make_gif
from planet.utils import transform_info
from envs.env import Env
from chester import logger
import torch
import pickle
import os
import os.path as osp
import copy
import multiprocessing as mp
import json
import numpy as np
from softgym.registered_env import env_arg_dict
import time
import matplotlib.pyplot as plt

import pyflex
from cem.utility import *
from PIL import Image

current_path = osp.dirname(__file__)
    
def run_task(vv, log_dir, exp_name, prev_sol):
    print(vv['test_num'])
    mp.set_start_method('spawn')
    env_name = vv['env_name']
    vv['algorithm'] = 'CEM'
    vv['env_kwargs'] = env_arg_dict[env_name]  # Default env parameters
    vv['plan_horizon'] = vv['env_kwargs_horizon'] #cem_plan_horizon[env_name]  # Planning horizon

    vv['population_size'] = vv['timestep_per_decision'] // vv['max_iters']
    if vv['use_mpc']:
        vv['population_size'] = vv['population_size'] // vv['plan_horizon']
    vv['num_elites'] = vv['population_size'] // 10 #10
    vv = update_env_kwargs(vv)

    # Configure logger
    logger.configure(dir=log_dir, exp_name=exp_name)
    logdir = logger.get_dir()
    assert logdir is not None
    os.makedirs(logdir, exist_ok=True)

    # Dump parameters
    # with open(osp.join(logger.get_dir(), 'variant.json'), 'w') as f:
    #     json.dump(vv, f, indent=2, sort_keys=True)

    env_symbolic = vv['env_kwargs']['observation_mode'] != 'cam_rgb'

    env_class = Env
    env_kwargs = {'env': vv['env_name'],
                  'symbolic': env_symbolic,
                  'seed': vv['seed'],
                  'max_episode_length': 200,
                  'action_repeat': 1,  # Action repeat for env wrapper is 1 as it is already inside the env
                  'bit_depth': 8,
                  'image_dim': None,
                  'env_kwargs': vv['env_kwargs']}
    
    env = env_class(**env_kwargs)

    policy = CEMPolicy(env, env_class, env_kwargs, vv['use_mpc'], plan_horizon=vv['plan_horizon'], max_iters=vv['max_iters'],
                       population_size=vv['population_size'], num_elites=vv['num_elites'])
    # Run policy
    initial_states, action_trajs, configs, all_infos = [], [], [], []
    for i in range(vv['test_episodes']):
        logger.log('episode ' + str(i))
        obs = env.reset()
        policy.reset(prev_sol)
        initial_state = env.get_state()
        action_traj = []
        infos = []
        time_cost_his = []
        
        time_start=time.time()
        
        j = 0
        while j < env.horizon:
            logger.log('episode {}, step {}'.format(i, j))
            time_start = time.time()
            action, cost_his= policy.get_action(obs)
            time_end = time.time()
            time_cost_his.append(time_end - time_start)
            print('time cost for one optimization: ', time_cost_his[-1])
            
            is_best_reached = False
            reward_his = []
            for k in range(vv['plan_horizon'] ):
                
                if not is_best_reached:
                    current_action = action[k]

                    action_traj.append(copy.copy(current_action))
                    obs, reward, _, info = env.step(current_action)
                    infos.append(info)
                    reward_his.append(reward)
                else:
                    infos.append(info)
                    reward_his.append(reward)
                
                if abs(reward * 100 + min(cost_his)) < 0.0001:
                    is_best_reached = True
        
            j = j + vv['plan_horizon']    
            
        ## plotting
        # plotting(data_x = np.linspace(1,len(cost_his),len(cost_his)), data_y = cost_his, save_path = './data/cem/cloth_folding/figs/cost_history', \
        #          x_label = 'number of iterations', y_label = 'minimum cost per iteration', line_color = 'red', line_width = 2,  line_marker = 'o', \
        #              line_label = 'cost history')
            
        # plotting(data_x = np.linspace(1,len(reward_his),len(reward_his)), data_y = reward_his, save_path = './data/cem/cloth_folding/figs/reward_history', \
        #          x_label = 'number of motion steps', y_label = 'reward per step', line_color = 'red', line_width = 2,  line_marker = 'o', \
        #              line_label = 'reward curve')

        all_infos.append(infos)
        
        ts = time.gmtime()
        ts = time.strftime("%Y_%m_%d_%H_%M_%S", ts)
        
        save_dir = osp.join(logdir, 'data/')
        if osp.exists(save_dir):     
            np.save(save_dir + 'all_infos_' + str(vv['test_num']) + '.npy', all_infos)
            np.save(save_dir + 'cost_his_' + str(vv['test_num'])  + '.npy', cost_his)
            np.save(save_dir + 'time_cost_his_' + str(vv['test_num'])  + '.npy', time_cost_his)
            np.save(save_dir + 'reward_his_' + str(vv['test_num'])  + '.npy', reward_his)
        else:
            os.makedirs(save_dir, exist_ok=True)
            np.save(save_dir + 'all_infos_' + str(vv['test_num'])  + '.npy', all_infos)
            np.save(save_dir + 'cost_his_' + str(vv['test_num'])  + '.npy', cost_his)
            np.save(save_dir + 'time_cost_his_' + str(vv['test_num'])  + '.npy', time_cost_his)
            np.save(save_dir + 'reward_his_' + str(vv['test_num'])  + '.npy', reward_his)

        save_dir = osp.join(logdir, 'traj/')
        if osp.exists(save_dir):  
            np.save(save_dir + 'action_traj_' + str(vv['test_num'])  + '.npy', action_traj)
        else:
            os.makedirs(save_dir, exist_ok=True)
            np.save(save_dir + 'action_traj_' + str(vv['test_num'])  + '.npy', action_traj)
        print(min(cost_his))
        print(max(reward_his))
        
        initial_states.append(initial_state.copy())
        action_trajs.append(action_traj.copy())
        configs.append(env.get_current_config().copy())

        # Log for each episode
        transformed_info = transform_info([infos])
        for info_name in transformed_info:
            logger.record_tabular('info_' + 'final_' + info_name, transformed_info[info_name][0, -1])
            logger.record_tabular('info_' + 'avarage_' + info_name, np.mean(transformed_info[info_name][0, :]))
            logger.record_tabular('info_' + 'sum_' + info_name, np.sum(transformed_info[info_name][0, :], axis=-1))
        logger.dump_tabular()

    # Dump trajectories
    traj_dict = {
        'initial_states': initial_states,
        'action_trajs': action_trajs,
        'configs': configs
    }
    with open(osp.join(log_dir, 'cem_traj.pkl'), 'wb') as f:
        pickle.dump(traj_dict, f)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    # Experiment
    parser.add_argument('--exp_name', default='cem', type=str)
    parser.add_argument('--env_name', default='ClothSideFold') #RopeConfiguration RopeFlatten ClothFold
    parser.add_argument('--log_dir', default='./data/simp_action/cem/')
    parser.add_argument('--test_episodes', default=1, type=int)
    parser.add_argument('--seed', default=100, type=int)
    parser.add_argument('--test_num', default=0, type=int)

    # CEM
    parser.add_argument('--max_iters', default=30, type=int) #10
    parser.add_argument('--timestep_per_decision', default=1000, type=int) #default=21000 120000
    parser.add_argument('--use_mpc', default=False, type=bool)

    # Override environment arguments
    parser.add_argument('--env_kwargs_render', default=False, type=bool)
    parser.add_argument('--env_kwargs_camera_name', default='default_camera', type=str)
    parser.add_argument('--env_kwargs_observation_mode', default='key_point', type=str)
    parser.add_argument('--env_kwargs_num_variations', default=1, type=int)
    parser.add_argument('--env_kwargs_num_picker', default=1, type=int)
    
    # 4 24  # 8 12  # 4 32  # 10 8
    parser.add_argument('--env_kwargs_horizon', default=15, type=int) #15
    parser.add_argument('--env_kwargs_action_repeat', default=8, type=int) # 8
    # parser.add_argument('--env_kwargs_headless', default=0, type=int)
    parser.add_argument('--env_kwargs_use_cached_states', default=True, type=bool)
    parser.add_argument('--env_kwargs_save_cached_states', default=False, type=bool)
    
    # key_point_idx = None
    key_point_idx = get_cloth_key_point_idx(parser)
    # key_point_idx = get_rand_int(low=0, high=10000, size=4)
    # key_point_idx = get_rand_edge_idx(width= 100, height = 100, size = 6)
    print(key_point_idx)
    parser.add_argument('--env_kwargs_key_point_idx', default=key_point_idx, type=np.array)
    
    # args = parser.parse_args()
    # run_task(args.__dict__, args.log_dir, args.exp_name, prev_sol = None)

if __name__ == '__main__':
    main()
