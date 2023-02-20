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

class RopeConfigurationEnv(RopeFlattenEnv):
    def __init__(self, cached_states_path='rope_configuration_init_states.pkl', reward_type = 'index', **kwargs):
        """
        :param cached_states_path:
        :param num_picker: Number of pickers if the aciton_mode is picker
        :param kwargs:

        manipulate the rope into a given character shape.
        """
        super().__init__(cached_states_path=cached_states_path, **kwargs)
        
        self.goal_position = None
        
        self.reward_type = reward_type
        
    def _reset(self):
        obs = super()._reset()
        self.get_goal_config()
        
        return obs
        
    def get_goal_config(self):
        
        config = self.current_config
        num_segment = config['segment']
        config_radius = config['radius']
        
        # rope_length_original = config['segment'] * config['radius'] * 0.5        
        # # rope_length = 0.5
        # print('length = %f'%rope_length_original)
        
        num_point = num_segment + 1
        
        particle_pos = np.array(pyflex.get_positions()).reshape((-1, 4))[:, :3]
        
        goal_pos = copy.deepcopy(particle_pos)
        
        num_inter = np.round(num_point / 2)
        
        for i in range(num_point):
            if i == 0:
                goal_pos[i,2] = -0.5 * np.sqrt(2) / 4
                goal_pos[i,0] = 0
            elif i <= num_inter:
                goal_pos[i,2] = goal_pos[i-1,2] + config_radius / 2 * np.sqrt(2) / 2
                goal_pos[i,0] = goal_pos[i-1,0] + config_radius / 2 * np.sqrt(2) / 2
            else:
                goal_pos[i,2] = goal_pos[i-1,2] + config_radius / 2 * np.sqrt(2) / 2
                goal_pos[i,0] = goal_pos[i-1,0] - config_radius / 2 * np.sqrt(2) / 2
                
        self.goal_position = goal_pos

    def get_goal_image(self, camera_height, camera_width):
        
        all_positions = pyflex.get_positions().reshape([-1, 4])
        goal_pos =  self.goal_position
        all_positions[:,0:3] = goal_pos.copy()
        pyflex.set_positions(all_positions)
        default_config = self.get_default_config()
        self.update_camera('default_camera', default_config['camera_params']['default_camera']) # why we need to do this?
        self.action_tool.reset([0., -1., 0.]) # hide picker
        goal_img = self.get_image(camera_height, camera_width)
        return goal_img                    

    def compute_reward(self, action=None, obs=None, **kwargs):
        """ Reward is the matching degree to the goal character"""
        goal_c_pos = self.goal_position
        current_pos = pyflex.get_positions().reshape((-1, 4))[:, :3]
        
        # way1: index matching
        if self.reward_type == 'index':
            dist = np.linalg.norm(current_pos - goal_c_pos, axis=1)
            reward = -np.mean(dist)
        
        return reward
            

