from scipy.optimize import minimize
from scipy import interpolate
import numpy as np
from math import *
import time
import matplotlib.pyplot as plt
import os.path as osp

current_path = osp.dirname(__file__)

def funccc(x):
        
    global x_real, y_real, num_real, len_real
    
    sorted_positions = np.array(x).reshape([-1,2])
    num_points = len(sorted_positions[:,0])
    
    len_fit = np.zeros(num_points)
    for i in range(num_points):
        if i == 0:
            len_fit[i] = 0
        else:
            len_fit[i] = len_fit[i-1] + np.linalg.norm(sorted_positions[i,:] - sorted_positions[i-1,:])

    num_interpolation = 40

    fit_interpolation = np.linspace(0, len_fit[-1], num_interpolation)
    
    f_fit_x = interpolate.interp1d(len_fit, sorted_positions[:,0])      
    f_fit_y = interpolate.interp1d(len_fit, sorted_positions[:,1])

    fitted_interpolate_x = f_fit_x(fit_interpolation)
    fitted_interpolate_y = f_fit_y(fit_interpolation)                                 
 
    real_interpolation = np.linspace(0, len_real[-1], num_interpolation)
    
    f_real_x = interpolate.interp1d(len_real, x_real)      
    f_real_y = interpolate.interp1d(len_real, y_real)

    real_interpolate_x = f_real_x(real_interpolation)
    real_interpolate_y = f_real_y(real_interpolation)
    
    output = np.sum((real_interpolate_x - fitted_interpolate_x)**2 + (real_interpolate_y - fitted_interpolate_y)**2)

    return output    

def get_corresponding_point(real_xy, fitted_xy):
    num_real = len(real_xy[:,0])
    len_real = np.zeros(num_real)
    
    for i in range(num_real):
        if i == 0:
            len_real[i] = 0
        else:
            len_real[i] = len_real[i-1] + np.linalg.norm(real_xy[i,:] - real_xy[i-1,:])

    num_fit = len(fitted_xy[:,0])
    len_fit = np.zeros(num_fit)
    
    for i in range(num_fit):
        if i == 0:
            len_fit[i] = 0
        else:
            len_fit[i] = len_fit[i-1] + np.linalg.norm(fitted_xy[i,:] - fitted_xy[i-1,:])

    num_interpolation = 50
    fit_interpolation = np.linspace(0, len_fit[-1], num_interpolation)
    
    f_fit_x = interpolate.interp1d(len_fit, fitted_xy[:,0])      
    f_fit_y = interpolate.interp1d(len_fit, fitted_xy[:,1])

    fitted_interpolate_x = f_fit_x(fit_interpolation)
    fitted_interpolate_y = f_fit_y(fit_interpolation)      
    fitted_interpolate_xy = np.hstack((fitted_interpolate_x[:,np.newaxis], fitted_interpolate_y[:,np.newaxis]))                        
 
    real_interpolation = np.linspace(0, len_real[-1], num_interpolation)
    
    f_real_x = interpolate.interp1d(len_real, x_real)      
    f_real_y = interpolate.interp1d(len_real, y_real)
    
    real_interpolate_x = f_real_x(real_interpolation)
    real_interpolate_y = f_real_y(real_interpolation)
    real_interpolate_xy = np.hstack((real_interpolate_x[:,np.newaxis], real_interpolate_y[:,np.newaxis])) 

    key_points = np.zeros([num_fit, 2])
    key_point_index = np.zeros(num_fit)
    index = np.linspace(0, num_interpolation-1, num_interpolation)
    index_real = np.linspace(0, num_real-1, num_real)
    for i in range(num_fit):
        distance = np.sqrt(np.sum(np.asarray(fitted_xy[i,:] - fitted_interpolate_xy)**2, axis = 1))
        index_distance = np.transpose(np.vstack((index, distance))) 
        index_distance = index_distance[np.argsort(index_distance[:,1])]            

        matched_real_interp_xy = real_interpolate_xy[np.int32(index_distance[0,0]),:]
        distance = np.sqrt(np.sum(np.asarray(matched_real_interp_xy - real_xy)**2, axis = 1))
        index_distance = np.transpose(np.vstack((index_real, distance))) 
        index_distance = index_distance[np.argsort(index_distance[:,1])] 
        key_point_index[i] = index_distance[0,0]
        key_points[i,:] = (real_xy[np.int32(index_distance[0,0]),:])

    return np.array(key_point_index,dtype=np.int64), key_points, fitted_interpolate_xy, real_interpolate_xy  

class line_fitting(object):
    def __init__(self):
        self.pos = None
        np.random.seed(100)

    def load_data(self, pos=None):
        if pos is None:
            self.pos = np.load(osp.join(current_path,'data/random.npy' ))
            pass
        else:
            self.pos = pos

    def get_key_point(self, threshold=5):
        global x_real, y_real, num_real, len_real
        real_xy = self.pos[:,[0,2]]    
        x_real = real_xy[:,0]
        y_real = real_xy[:,1]
        self.bnds = ((-1, 1), (-1, 1))
    
        num_real = len(x_real)
        len_real = np.zeros(num_real)
    
        for i in range(num_real):
            if i == 0:
                len_real[i] = 0
            else:
                len_real[i] = len_real[i-1] + np.linalg.norm(np.array([x_real[i], y_real[i]]) - np.array([x_real[i-1], y_real[i-1]]))

        ## optimization & plotting
        not_improved = 0
        min_cost = np.inf
        best_fit = None
        start_time = time.time()
        num_simp_points = 2
        while not_improved <= threshold:            
            bnds = self.bnds * num_simp_points
            x0 = np.random.rand(2*num_simp_points)
            res = minimize(funccc, x0, method = 'SLSQP', bounds=bnds, options={'ftol': 1e-10, 'disp': False, 'maxiter': 10000})

            num_simp_points += 1
            if res.fun > min_cost or abs(res.fun - min_cost) < 0.001:
                not_improved += 1
            else:
                min_cost = res.fun
                not_improved = 0
                best_fit = res.x

        elapsed = (time.time() - start_time)
        print('time used for line fitting:', elapsed)

        fitted_xy = np.array(best_fit).reshape([-1,2])
        key_point_index, key_points, fitted_interpolate_xy, real_interpolate_xy = get_corresponding_point(real_xy, fitted_xy)

        print(key_point_index)

        return key_point_index

def main():
    lf = line_fitting()
    lf.load_data()
    lf.get_key_point()

if __name__ == '__main__':
    main()
    













