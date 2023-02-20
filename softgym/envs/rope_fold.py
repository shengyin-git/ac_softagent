import re
import numpy as np
import random
import pickle
import os.path as osp
import pyflex
from gym.spaces import Box
from softgym.envs.rope_flatten import RopeFlattenEnv
import scipy
import copy
from copy import deepcopy
import scipy.optimize as opt
import cv2
from scipy import interpolate

class RopeFoldEnv(RopeFlattenEnv):
    def __init__(self, cached_states_path='rope_folding_init_states.pkl', reward_type = 'index', use_simplified_cost = False, **kwargs):
        """
        :param cached_states_path:
        :param num_picker: Number of pickers if the aciton_mode is picker
        :param kwargs:

        manipulate the rope into a given character shape.
        """
        super().__init__(cached_states_path=cached_states_path, **kwargs)
        
        self.goal_position = None
        
        self.reward_type = reward_type

        # self.key_point_idx = np.array([0,16,24,40])

        self.use_simplified_cost = use_simplified_cost
        
    def _reset(self):
        obs = super()._reset()
        self.get_goal_config()
        
        return obs
        
    def get_goal_config(self):
        config = self.current_config
        num_segment = config['segment']
        config_radius = config['radius']
        
        num_point = num_segment + 1
        
        ## get goal shape interpolation function
        lenth_rope = num_segment * config_radius * 0.5 
        ratio = 30/lenth_rope
        
        goal_shape_x = np.array([-3, 0, 3, 0, -3, 0, 3]) / ratio
        goal_shape_y = np.array([0, 3*np.sqrt(3), 6*np.sqrt(3), 6*np.sqrt(3), 6*np.sqrt(3), 3*np.sqrt(3), 0]) / ratio
        goal_shape_length = np.array([0, 6, 12, 15, 18, 24, 30]) / ratio
        
        f_interp_x = interpolate.interp1d(goal_shape_length, goal_shape_x, kind = 'linear')
        f_interp_y = interpolate.interp1d(goal_shape_length, goal_shape_y, kind = 'linear')
        
        rope_interp = np.arange(num_point) * config_radius * 0.5
        goal_pos_x = f_interp_x(rope_interp)
        goal_pos_y = f_interp_y(rope_interp)
        
        ## get goal configuration            
        particle_pos = np.array(pyflex.get_positions()).reshape((-1, 4))[:, :3]
            
        goal_pos = deepcopy(particle_pos)
        
        goal_pos[:,0] = goal_pos_x
        goal_pos[:,2] = goal_pos_y
                
        self.goal_position = goal_pos        

    def _get_goal_state(self, camera_height, camera_width):
        
        all_positions = pyflex.get_positions().reshape([-1, 4])
        goal_pos =  self.goal_position
        all_positions[:,0:3] = goal_pos.copy()
        pyflex.set_positions(all_positions)
        default_config = self.get_default_config()
        self.update_camera('default_camera', default_config['camera_params']['default_camera']) 
        self.action_tool.reset([0., -1., 0.]) # hide picker
        goal_img = self.get_image(camera_height, camera_width)
        return goal_img, goal_pos

    def get_initial_image(self, camera_height, camera_width):
        
        all_positions = pyflex.get_positions().reshape([-1, 4])
        initial_pos =  self.cached_init_states[0]['particle_pos'].reshape([-1,4])[:,:3]
        all_positions[:,0:3] = initial_pos.copy()
        pyflex.set_positions(all_positions)
        default_config = self.get_default_config()
        self.update_camera('default_camera', default_config['camera_params']['default_camera']) 
        self.action_tool.reset([0., -1., 0.]) # hide picker
        goal_img = self.get_image(camera_height, camera_width)
        return goal_img                   

    def compute_reward(self, action=None, obs=None, **kwargs):
        """ Reward is the matching degree to the goal character"""
        goal_c_pos = self.goal_position
        current_pos = pyflex.get_positions().reshape((-1, 4))[:, :3]
        
        # way1: index matching
        # if self.reward_type == 'index':
        #     dist = np.linalg.norm(current_pos - goal_c_pos, axis=1)
        #     # reward = -np.sum(dist)
        #     reward = -np.mean(dist)
        
        # return reward

        if False: #self.use_simplified_cost:
            goal_c_pos = self.goal_position
            current_pos = pyflex.get_positions().reshape((-1, 4))[:, :3]
            if self.reward_type == 'index':
                dist = np.linalg.norm(current_pos[self.key_point_idx,:] - goal_c_pos[self.key_point_idx,:], axis=1)
                dist_all = np.linalg.norm(current_pos - goal_c_pos, axis=1)
                reward = -np.mean(dist)
                reward_all = -np.mean(dist_all)                

            return [reward, reward_all]

        else:
            if self.reward_type == 'index':
                dist = np.linalg.norm(current_pos - goal_c_pos, axis=1)
                reward = -np.mean(dist)
            return reward


            

            

