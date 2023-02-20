from PIL import Image
import matplotlib.pyplot as plt
import os.path as osp
import numpy as np
import time
import os
from mesh_simp.cloth_smp import *
from softgym.registered_env import env_arg_dict
from experiments.planet.train import update_env_kwargs
from envs.env import Env
from line_fitting.line_fitting_interpolation import *

current_path = osp.dirname(__file__)

cem_plan_horizon = {
    'PassWater': 7,
    'PourWater': 40,
    'PourWaterAmount': 40,
    'ClothFold': 15,
    'ClothFoldCrumpled': 30,
    'ClothFoldDrop': 30,
    'ClothFlatten': 15,
    'ClothDrop': 15,
    'RopeFlatten': 15,
    'RopeConfiguration': 20
}

def get_rope_key_point_idx(parser):
    args = parser.parse_args()
    vv = args.__dict__
    log_dir = args.log_dir
    exp_name = args.exp_name

    env_name = vv['env_name']
    vv['env_kwargs'] = env_arg_dict[env_name]  # Default env parameters
    vv = update_env_kwargs(vv)
    
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
    env_kwargs['env_kwargs']['render'] = True
    
    env = env_class(**env_kwargs)
    env.reset()
    # pos = env.goal_position
    goal_img, pos = env._get_goal_state(camera_height=720, camera_width=720)
    # env.close()

    lf = line_fitting()
    lf.load_data(pos)
    key_points = lf.get_key_point()

    return key_points

def get_cloth_key_point_idx(parser):
    args = parser.parse_args()
    vv = args.__dict__

    env_name = vv['env_name']
    vv['env_kwargs'] = env_arg_dict[env_name]  # Default env parameters
    vv = update_env_kwargs(vv)
    
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
    env_kwargs['env_kwargs']['render'] = True
    
    env = env_class(**env_kwargs)
    env.reset()
    goal_img, flat_pos = env._get_flat_state(camera_height=720, camera_width=720)
    goal_img, goal_pos = env._get_goal_state(camera_height=720, camera_width=720)

    # ini_pos_path = osp.join(current_path,'data/simp_model/particle_pos_ini.npy')
    # flat_pos = np.load(ini_pos_path)
    # goal_pos_path = osp.join(current_path,'data/simp_model/particle_pos_final.npy')
    # goal_pos = np.load(goal_pos_path)

    # env.close() # the goal pos may change for different runs

    cs = mesh_smp()
    cs.import_flat_points(flat_pos)
    cs.generate_initial_mesh()
    cs.generate_final_mesh(goal_pos)
    cs.smp_the_mesh()
    key_points = cs.get_key_point()
    # cs.plot_simplified_mesh()

    return key_points

def get_picked_point(action):
    # key_point_index = (np.sort(np.load('./cem/data/key_point/key_point_index.npy')))
    key_point_index = np.linspace(0,8099,8100)
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
        picking_idx = key_point_index[np.int32(idx)]
        
    return picking_idx

def get_scaled_action(action):
    lb, ub = np.array([-0.01, -0.01, -0.01]), np.array([0.01, 0.01, 0.01])
    scaled_action = lb + (action + 1.) * 0.5 * (ub - lb)
    scaled_action = np.clip(scaled_action, lb, ub)
       
    return scaled_action
    
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
    dia_sim = np.linalg.norm(particle_pos_his[0,0,:] - particle_pos_his[0,89,:])/89
    len_sim = dia_sim * 90

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

def plotting(data_x, data_y, save_dir, x_label, y_label, line_color = 'red', line_width = 2,  line_marker = 'o', line_label = 'cost history'):
    
    fig = plt.figure(num=1, figsize = (8,8))
    plt.plot(data_x, data_y, c = line_color, linewidth = line_width, marker = line_marker, label = line_label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
        
    ts = time.gmtime()
    ts = time.strftime("%Y_%m_%d_%H_%M_%S", ts)

    # save_dir =  osp.join(save_dir, 'images/')
    if os.path.exists(save_dir):
        plt.savefig(save_dir + ts + '.png')
    else:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_dir + ts + '.png')
        
    plt.show()

def vv_to_args(vv):
    class VArgs(object):
        def __init__(self, vv):
            for key, val in vv.items():
                setattr(self, key, val)

    args = VArgs(vv)

    return args
    
def get_rand_int(low, high, size):
    rand_int = []
    i = 0
    while i < size:
        rand_temp = np.random.randint(low=low, high=high)
        if rand_temp not in rand_int:
            rand_int.append(rand_temp)
            i = i + 1
    return np.array(rand_int)

def get_rand_edge_idx(width, height, size):
    edge_idx = []
    
    for i in range(height):
        if i == 0:
            for j in range(width):
                edge_idx.append(j)
        elif i == height - 1:
            temp = width*(height-1)
            for j in range(width):
                edge_idx.append(temp+j)
        else:
            edge_idx.append(i*width)
            edge_idx.append((i+1)*width-1)
    
    size_edge_idx = len(edge_idx)

    rand_int = []
    i = 0
    while i < size:
        rand_temp = edge_idx[np.random.randint(low=0, high=size_edge_idx)]
        if rand_temp not in rand_int:
            rand_int.append(rand_temp)
            i = i + 1
    return np.array(rand_int)
