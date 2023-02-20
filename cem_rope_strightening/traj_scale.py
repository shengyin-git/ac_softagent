"""
Created on Tue Aug 16 10:29:07 2022

@author: shengyin
"""
import sys
sys.path.insert(1, '/home/shengyin/OneDrive/Lab_Ubun_Projects/softagent_rope_straightening_cem/')

from enum import Flag
from experiments.planet.train import update_env_kwargs
from visualize_cem import cem_make_gif
from planet.utils import transform_info
from envs.env import Env
from chester import logger
import torch
import pickle
import os
import os.path as osp
import copy
import json
import numpy as np
from softgym.registered_env import env_arg_dict
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import interpolate
import math
import pyflex
from softgym.utils.visualization import save_numpy_as_gif
import torchvision

from PIL import Image

current_path = osp.dirname(__file__)
img_size = 720

def get_picked_point(action):
    # key_point_index = (np.sort(np.load('cem/data/key_point/key_point_index.npy')))
    key_point_index = np.array([0, 40]) 
    # key_point_index = np.linspace(0,40,41) 
    num_key_points = len(key_point_index)
    index = np.linspace(0, num_key_points + 1, num_key_points+2)
    action_range = np.linspace(0, 1, num_key_points + 2)
    f_interp = interpolate.interp1d(action_range, index)
    none_threshold = num_key_points

    lb, ub = 0, 1
    scaled_action = lb + (action + 1.) * 0.5 * (ub - lb)
    scaled_action = np.clip(scaled_action, lb, ub)
    idx = np.floor(f_interp(scaled_action))

    if idx >= none_threshold:
        picking_idx = None
    else:
        picking_idx = np.int32(key_point_index[np.int32(idx)])
        
    return picking_idx

def get_scaled_action(action):
    lb, ub = np.array([-0.01, -0.01, -0.01]), np.array([0.01, 0.01, 0.01])
    scaled_action = lb + (action + 1.) * 0.5 * (ub - lb)
    scaled_action = np.clip(scaled_action, lb, ub)
       
    return scaled_action

def show_picture(row, col, r, g, b, save_path):
    image = Image.new("RGB", (row, col))
    
    counter = 0
    
    for i in range(0, row):
        for j in range(0, col):
            image.putpixel((j, i), (r[counter], g[counter], b[counter]))
            counter = counter + 1
    # image.show()

    # save_dir =  osp.join(save_path, 'images/')
    # if os.path.exists(save_dir):
    #     image.save(save_dir+ str(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())) + '.png')
    # else:
    #     os.makedirs(save_dir, exist_ok=True)
    #     image.save(save_dir+ str(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())) + '.png')

def plotting(data_x, data_y, save_path, x_label, y_label, line_color = 'red', line_width = 2,  line_marker = 'o', line_label = 'cost history'):
    
    fig = plt.figure(num=1, figsize = (4,4))
    plt.plot(data_x, data_y, c = line_color, linewidth = line_width, marker = line_marker, label = line_label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
        
    ts = time.gmtime()
    ts = time.strftime("%Y_%m_%d_%H_%M_%S", ts)
    save_dir =  osp.join(save_path, 'figs/')
    if os.path.exists(save_dir):
        plt.savefig(save_dir + ts + '.png')
    else:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_dir + ts + '.png')
        
    # plt.show()

def vv_to_args(vv):
    class VArgs(object):
        def __init__(self, vv):
            for key, val in vv.items():
                setattr(self, key, val)

    args = VArgs(vv)

    return args
    
def plotting_3d(particle_pos = None, picker_start = None, picker_end = None, saving_path = None):
    fig = plt.figure(figsize=(8,8))
    ax = Axes3D(fig)
    
    xs = particle_pos[:,2]
    ys = particle_pos[:,0]
    zs = particle_pos[:,1]
    
    ax.scatter(xs, ys, zs, c = 'y', marker = 'o', alpha = 0.05)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    ax.quiver(picker_start[2], picker_start[0], picker_start[1], picker_end[2]-picker_start[2], picker_end[0]-picker_start[0], picker_end[1]-picker_start[1], arrow_length_ratio = 0.3)
    
    ax.set_xlim(-0.4,0.4)
    ax.set_ylim(-0.4,0.4)
    ax.set_zlim(-0.4,0.4)
    
    # current_path = osp.dirname(__file__)
    ts = time.gmtime()
    ts = time.strftime("%Y_%m_%d_%H_%M_%S", ts)

    save_dir =  osp.join(saving_path, 'cloth_with_action_images/')
    if os.path.exists(save_dir):
        plt.savefig(save_dir+ ts + '.png')
    else:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_dir+ ts + '.png')
    
    # plt.show() 

def plotting_actions(particle_pos_his, picker_pos_his_new, action_sequence, save_path):
    particle_pos_his = np.array(particle_pos_his)
    picker_pos_his_new = np.array(picker_pos_his_new)
    action_sequence = np.array(action_sequence)
    num_actions = np.int32(len(picker_pos_his_new[:,0]) / 2)

    ### plotting the actions on cloth
    for i in range(num_actions):
        plotting_3d(particle_pos = particle_pos_his[i], picker_start = picker_pos_his_new[2*i], picker_end = picker_pos_his_new[2*i+1], saving_path = save_path)
        time.sleep(1)

    ## get action data for the real robot
    dia_sim = np.linalg.norm(particle_pos_his[0,0,:] - particle_pos_his[0,40,:])/40
    len_sim = dia_sim * 141

    len_real = 0.3
    len_ratio = len_real / len_sim

    action_for_robot = []
    pre_grasped_idx = None

    for i in range(num_actions):
        grasped_idx = get_picked_point(action_sequence[i,3])
        
        if i > 0:
            pre_picker_start = picker_pos_his_new[2*i-1]
        
        picker_start = picker_pos_his_new[2*i]
        picker_end = picker_pos_his_new[2*i+1]
        
        picker_motion = (picker_end - picker_start) * len_ratio
        
        if i == 0:
            # this cannot be none
            action_temp = []
            action_temp.append(picker_motion)
            pre_grasped_idx = grasped_idx
        elif i == num_actions-1:
            if grasped_idx is not None and pre_grasped_idx != grasped_idx:
                action_for_robot.append(action_temp)
                start_pos = action_temp[-1] + (picker_start - pre_picker_start)*len_ratio
                action_temp = []
                action_temp.append(start_pos)
                action_temp.append(start_pos + picker_motion)
                action_for_robot.append(action_temp)    
            elif grasped_idx is not None:
                start_pos = action_temp[-1]
                action_temp.append(start_pos + picker_motion)
                action_for_robot.append(action_temp)
            else:
                break
        elif grasped_idx is None:
            if pre_grasped_idx is None:
                break
                # continue
            else:
                action_for_robot.append(action_temp)
                action_temp = []
                pre_grasped_idx = None
        elif pre_grasped_idx is None:
            start_pos = action_for_robot[-1][-1]
            action_temp.append(start_pos + picker_motion)
            pre_grasped_idx = grasped_idx
        elif pre_grasped_idx != grasped_idx:
            action_for_robot.append(action_temp)
            start_pos = action_temp[-1] + (picker_start - pre_picker_start)*len_ratio
            action_temp = []
            action_temp.append(start_pos)
            action_temp.append(start_pos + picker_motion)
            pre_grasped_idx = grasped_idx
        else:
            start_pos = action_temp[-1]
            action_temp.append(start_pos + picker_motion)
    ts = time.gmtime()
    ts = time.strftime("%Y_%m_%d_%H_%M_%S", ts)

    save_dir =  osp.join(save_path, 'robot_action/')
    if os.path.exists(save_dir):
        np.save(save_dir + ts +'.npy', action_for_robot)
    else:
        os.makedirs(save_dir, exist_ok=True)
        np.save(save_dir + ts +'.npy', action_for_robot)
            
    num_pick_place = len(action_for_robot)
    
    fig = plt.figure(figsize = (8, 6))
    ax = Axes3D(fig)
    
    action_for_robot = np.array(action_for_robot)
    for i in range(num_pick_place):
        if i == 0:
            temp_action = np.array(action_for_robot[i])
            temp_action_ini = np.array([0,0,0])

            temp_action = np.vstack((temp_action_ini, temp_action))
            ax.plot(temp_action[:,2],temp_action[:,0],temp_action[:,1], color = "green")
        else:
            temp_action = np.array(action_for_robot[i])
            temp_action_ini = np.array(action_for_robot[i-1])[-1,:]

            temp_action = np.vstack((temp_action_ini, temp_action))
            ax.plot(temp_action[:,2],temp_action[:,0],temp_action[:,1], color = "green")

    ax.plot(picker_pos_his_new[:,2],picker_pos_his_new[:,0],picker_pos_his_new[:,1], color = "red")    
    ax.set(xlabel="X", ylabel = "Y", zlabel = "Z")

    ax.set_xlim(-0.4,0.4)
    ax.set_ylim(-0.4,0.4)
    ax.set_zlim(-0.4,0.4)

    ts = time.gmtime()
    ts = time.strftime("%Y_%m_%d_%H_%M_%S", ts)
    
    save_dir =  osp.join(save_path, 'scaled trjectory/')
    if os.path.exists(save_dir):
        plt.savefig(save_dir+ ts + '.png')
    else:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_dir+ ts + '.png')

def run_task(vv, log_dir, exp_name):
    env_name = vv['env_name']
    vv['algorithm'] = 'CEM'
    vv['env_kwargs'] = env_arg_dict[env_name]  # Default env parameters
    vv['plan_horizon'] = vv['env_kwargs_horizon'] #cem_plan_horizon[env_name]  # Planning horizon

    vv = update_env_kwargs(vv)

    logger.configure(dir=log_dir, exp_name=exp_name)
    logdir = logger.get_dir()
    assert logdir is not None
    os.makedirs(logdir, exist_ok=True)

    # Dump parameters
    with open(osp.join(logger.get_dir(), 'variant.json'), 'w') as f:
        json.dump(vv, f, indent=2, sort_keys=True)

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

    env_kwargs_render = copy.deepcopy(env_kwargs)
    env_kwargs_render['env_kwargs']['render'] = True

    if vv['saved_action_path'] is None:
        print('invalid action path!')
        return
    else:
        action_seq = np.load(vv['saved_action_path'])
        ori_action_seq = copy.deepcopy(action_seq)
        num_actions = len(action_seq[:,0])

    obs = env.reset()

    get_initial_image = vv['get_initial_image']
    if get_initial_image:
        temp_image = env.get_image(img_size, img_size)
        show_picture(img_size, img_size, temp_image[:,:,0].flatten(), temp_image[:,:,1].flatten(), temp_image[:,:,2].flatten(), save_path = logdir)
        particle_pos = np.array(pyflex.get_positions()).reshape(-1, 4)[:, :3]
        save_dir = osp.join(logdir, 'ini_pos/')
        if osp.exists(save_dir):
            np.save(save_dir + 'side_folded_pos.npy', particle_pos)
        else:
            os.makedirs(save_dir, exist_ok=True)
            np.save(save_dir + 'side_folded_pos.npy', particle_pos)
        # return
    
    initial_state = []
    initial_state.append(env.get_state())
    configs = []
    configs.append(env.get_current_config().copy())
    reward_his = []
    particle_pos_his = []
    picker_pos_his = []    
    picker_pos_his_new = []

    ## this is for the final folded state
    get_folded_data = vv['get_folded_data']
    if get_folded_data:
        env._set_to_final_folded()
        temp_image = env.get_image(img_size, img_size)
        show_picture(img_size, img_size, temp_image[:,:,0].flatten(), temp_image[:,:,1].flatten(), temp_image[:,:,2].flatten(), save_path = logdir)
        particle_pos = np.array(pyflex.get_positions()).reshape(-1, 4)[:, :3]
        save_dir = osp.join(logdir, 'final_pos/')
        if osp.exists(save_dir):
            np.save(save_dir + 'side_folded_pos.npy', particle_pos)
        else:
            os.makedirs(save_dir, exist_ok=True)
            np.save(save_dir + 'side_folded_pos.npy', particle_pos)
        return

    all_frames = []
    frames = []   
    frames.append(env.get_image(img_size, img_size))

    for i in range(num_actions):
        particle_pos = np.array(pyflex.get_positions()).reshape(-1, 4)[:, :3]
        particle_pos_his.append(particle_pos)
        
        picker_pos = np.array(pyflex.get_shape_states()).reshape(-1, 14)[0,:3]
        picker_pos_his.append(picker_pos)
                
        current_action = action_seq[i]
        # print('action', current_action)
        obs, reward, _, info = env.step(current_action, record_continuous_video=True, img_size=img_size)
        # print('reward', reward)
        reward_his.append(reward)
        frames.extend(info['flex_env_recorded_frames'])
        
        picker_pos = np.array(pyflex.get_shape_states()).reshape(-1, 14)[0,:3]
        picker_pos_his.append(picker_pos)
        
        action_seq[i,3] = get_picked_point(action_seq[i,3])
        action_seq[i,:3] = get_scaled_action(action_seq[i,:3])
        if math.isnan(action_seq[i,3]):
            continue
        else:
            picked_pos = particle_pos[np.int32(action_seq[i,3])]
        
        picker_pos_his_new.append(picked_pos)
        picker_pos_his_new.append(picked_pos + vv['env_kwargs']['action_repeat'] * action_seq[i,:3])
        
        time.sleep(1)
        temp_image = env.get_image(img_size, img_size)
        show_picture(img_size, img_size, temp_image[:,:,0].flatten(), temp_image[:,:,1].flatten(), temp_image[:,:,2].flatten(), save_path = logdir)

    # all_frames.append(frames)
    # all_frames = np.array(all_frames).transpose([1, 0, 4, 2, 3])
    # grid_imgs = [torchvision.utils.make_grid(torch.from_numpy(frame), nrow=5).permute(1, 2, 0).data.cpu().numpy() for frame in all_frames]
    # ts = time.gmtime()
    # ts = time.strftime("%Y_%m_%d_%H_%M_%S", ts)
    # save_path = osp.join(logdir, 'gifs/')
    # if osp.exists(save_path):
    #     save_numpy_as_gif(np.array(grid_imgs), osp.join(logdir, 'gifs/' + ts + '.gif'))
    # else:
    #     os.makedirs(save_path, exist_ok=True)
    #     save_numpy_as_gif(np.array(grid_imgs), osp.join(logdir, 'gifs/' + ts + '.gif'))
        
    # particle_pos = np.array(pyflex.get_positions()).reshape(-1, 4)[:, :3]
    # particle_pos_his.append(particle_pos)
        
    # picker_pos = np.array(pyflex.get_shape_states()).reshape(-1, 14)[0,:3]
    # picker_pos_his.append(picker_pos)     
                  
    plotting(data_x = np.linspace(1,len(reward_his),len(reward_his)), data_y = reward_his, save_path = logdir + 'reward_history/', \
                 x_label = 'number of motion steps', y_label = 'reward per step', line_color = 'red', line_width = 2,  line_marker = 'o', \
                     line_label = 'reward curve')
    print('maximum reward is:', max(reward_his))

    # Dump data and video
    # ts = time.gmtime()
    # ts = time.strftime("%Y_%m_%d_%H_%M_%S", ts)
    # save_dir = osp.join(logdir, 'traj/')
    # if osp.exists(save_dir):           
    #     np.save(save_dir + 'picker_pos_his_' + ts +'.npy', picker_pos_his)
    #     np.save(save_dir + 'picker_pos_his_new_' + ts +'.npy', picker_pos_his_new)
    #     np.save(save_dir + 'particle_pos_his_' + ts +'.npy', particle_pos_his)
    # else:
    #     os.makedirs(save_dir, exist_ok=True)
    #     np.save(save_dir + 'picker_pos_his_' + ts +'.npy', picker_pos_his)
    #     np.save(save_dir + 'picker_pos_his_new_' + ts +'.npy', picker_pos_his_new)
    #     np.save(save_dir + 'particle_pos_his_' + ts +'.npy', particle_pos_his)

    # save_dir = osp.join(logdir, 'gifs/')
    # if osp.exists(save_dir):           
    #     cem_make_gif(env_render, initial_state, [ori_action_seq], configs, logdir, 'gifs/' + vv['env_name'] + '_' + ts + '.gif')
    # else:
    #     os.makedirs(save_dir, exist_ok=True)
    #     cem_make_gif(env_render, initial_state, [ori_action_seq], configs, logdir, 'gifs/' + vv['env_name'] + '_' + ts + '.gif')      

    ## plotting the actions on cloth
    plotting_actions(particle_pos_his, picker_pos_his_new, ori_action_seq, save_path = logdir)

cem_plan_horizon = {
    'PassWater': 7,
    'PourWater': 40,
    'PourWaterAmount': 40,
    'ClothFold': 4, # 15
    'ClothFoldCrumpled': 30,
    'ClothFoldDrop': 30,
    'ClothFlatten': 15,
    'ClothDrop': 15,
    'RopeFlatten': 15,
    'RopeConfiguration': 20, #15,
}

def main():
    import argparse
    parser = argparse.ArgumentParser()
    # Experiment
    parser.add_argument('--exp_name', default='cem', type=str)
    parser.add_argument('--env_name', default='RopeFlatten') #RopeConfiguration RopeFlatten ClothFold
    parser.add_argument('--log_dir', default='./data/post_process/hand_crafted/')
    parser.add_argument('--test_episodes', default=1, type=int)
    parser.add_argument('--seed', default=100, type=int)

    parser.add_argument('--get_initial_image', default=False, type=bool)
    parser.add_argument('--get_folded_data', default=False, type=bool)

    # simple 'data/cem/simple/20220902145755/traj/action_traj_1.npy' 20 2
    # original
    # hand_crafted 80 8 'data/cem/hand_crafted/20220902155539/traj/action_traj_1.npy'
    path_to_action = 'data/cem/simple/20220902145755/traj/action_traj_1.npy'
    parser.add_argument('--saved_action_path', default=path_to_action, type=str)

    # Override environment arguments
    parser.add_argument('--env_kwargs_render', default=True, type=bool)
    parser.add_argument('--env_kwargs_camera_name', default='default_camera', type=str)
    parser.add_argument('--env_kwargs_observation_mode', default='key_point', type=str)
    parser.add_argument('--env_kwargs_num_variations', default=1, type=int)
    parser.add_argument('--env_kwargs_num_picker', default=1, type=int) #2
    
    parser.add_argument('--env_kwargs_horizon', default=80, type=int) #15
    parser.add_argument('--env_kwargs_action_repeat', default=8, type=int) # 8
    parser.add_argument('--env_kwargs_headless', default=1, type=int)
    parser.add_argument('--env_kwargs_use_cached_states', default=True, type=bool)
    parser.add_argument('--env_kwargs_save_cached_states', default=False, type=bool)
    
    # parser.add_argument('--env_kwargs_use_simplified_key_point', default=Ture, type=bool)

    args = parser.parse_args()
    run_task(args.__dict__, args.log_dir, args.exp_name)
    
if __name__ == '__main__':
    main()