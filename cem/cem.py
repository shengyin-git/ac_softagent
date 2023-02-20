import copy
import pickle
import numpy as np
import scipy.stats as stats
from tqdm import tqdm
from cem.parallel_worker import ParallelRolloutWorker


class CEMOptimizer(object):
    def __init__(self, cost_function, solution_dim, max_iters, population_size, num_elites,
                 upper_bound=None, lower_bound=None, epsilon=0.05):
        """
        :param cost_function: Takes input one or multiple data points in R^{sol_dim}\
        :param solution_dim: The dimensionality of the problem space
        :param max_iters: The maximum number of iterations to perform during optimization
        :param population_size: The number of candidate solutions to be sampled at every iteration
        :param num_elites: The number of top solutions that will be used to obtain the distribution
                            at the next iteration.
        :param upper_bound: An array of upper bounds for the sampled data points
        :param lower_bound: An array of lower bounds for the sampled data points
        :param epsilon: A minimum variance. If the maximum variance drops below epsilon, optimization is stopped.
        """
        super().__init__()
        self.solution_dim, self.max_iters, self.population_size, self.num_elites = \
            solution_dim, max_iters, population_size, num_elites

        self.ub, self.lb = upper_bound.reshape([1, solution_dim]), lower_bound.reshape([1, solution_dim])
        self.epsilon = epsilon

        if num_elites > population_size:
            raise ValueError("Number of elites must be at most the population size.")

        self.cost_function = cost_function
    ## original solution
    # def obtain_solution(self, cur_state, init_mean=None, init_var=None):
    #     """ Optimizes the cost function using the provided initial candidate distribution
    #     :param cur_state: Full state of the current environment such that the environment can always be reset to this state
    #     :param init_mean: (np.ndarray) The mean of the initial candidate distribution.
    #     :param init_var: (np.ndarray) The variance of the initial candidate distribution.
    #     :return:
    #     """
    #     mean = (self.ub + self.lb) / 2. if init_mean is None else init_mean
    #     var = (self.ub - self.lb) / 4. if init_var is None else init_var
    #     t = 0
    #     X = stats.norm(loc=np.zeros_like(mean), scale=np.ones_like(mean))
        
    #     cost_his = np.zeros(self.max_iters)

    #     while (t < self.max_iters):  # and np.max(var) > self.epsilon:
    #         print("inside CEM, iteration {}".format(t))
    #         samples = X.rvs(size=[self.population_size, self.solution_dim]) * np.sqrt(var) + mean
    #         samples = np.clip(samples, self.lb, self.ub)
    #         costs = self.cost_function(cur_state, samples)
    #         sort_costs = np.argsort(costs)

    #         elites = samples[sort_costs][:self.num_elites]
    #         mean = np.mean(elites, axis=0)
    #         var = np.var(elites, axis=0)
            
    #         cost_his[t] = min(costs)
            
    #         t += 1
    #     sol, solvar = mean, var
    #     return sol, cost_his
    
    # modified solution
    def obtain_solution(self, cur_state, mean_coeff, var_coeff, init_mean=None, init_var=None):
        mean = (self.ub + self.lb) / mean_coeff if init_mean is None else init_mean
        var = (self.ub - self.lb) / var_coeff if init_var is None else init_var
        t = 0
        X = stats.norm(loc=np.zeros_like(mean), scale=np.ones_like(mean))
        
        cost_his = np.zeros(self.max_iters)
        
        best_sol = mean #None
        best_var = None
        min_cost = np.inf

        while (t < self.max_iters):  # and np.max(var) > self.epsilon:
            print("inside CEM, iteration {}".format(t))
            samples = X.rvs(size=[self.population_size, self.solution_dim]) * np.sqrt(var) + mean
            samples = np.clip(samples, self.lb, self.ub)
            samples = np.vstack((samples, best_sol))
            
            if t == 0:
                costs = self.cost_function(cur_state, samples)
            else:
                costs = self.cost_function(cur_state, samples[0:-1])
                costs.append(min_cost)
            sort_costs = np.argsort(costs)            

            elites = samples[sort_costs][:self.num_elites]
            current_best = samples[0]
            mean = np.mean(elites, axis=0)
            var = np.var(elites, axis=0)
            
            if costs[sort_costs[0]] < min_cost:
                min_cost = costs[sort_costs[0]]
                best_sol = elites[0]
                best_var = var
                cost_his[t] = min_cost
            else:
                cost_his[t] = cost_his[t-1]

            print(cost_his[t])
                
            # if np.mean(sort_costs[:self.num_elites]) < min_cost:
            #     min_cost = np.mean(sort_costs[:self.num_elites]) 
            #     best_sol = mean
            #     cost_his[t] = min_cost
            # else:
            #     cost_his[t] = cost_his[t-1]
            
            t += 1
        sol, solvar = best_sol, best_var
        return sol, solvar, cost_his

class CEMPolicy(object):
    """ Use the ground truth dynamics to optimize a trajectory of actions. """

    def __init__(self, env, env_class, env_kwargs, use_mpc, plan_horizon, max_iters, population_size, num_elites):
        self.env, self.env_class, self.env_kwargs = env, env_class, env_kwargs
        self.use_mpc = use_mpc
        self.plan_horizon, self.action_dim = plan_horizon, len(env.action_space.sample())
        self.action_buffer = []
        self.prev_sol = None
        self.rollout_worker = ParallelRolloutWorker(env_class, env_kwargs, plan_horizon, self.action_dim)

        lower_bound = np.tile(env.action_space.low[None], [self.plan_horizon, 1]).flatten()
        upper_bound = np.tile(env.action_space.high[None], [self.plan_horizon, 1]).flatten()
        self.optimizer = CEMOptimizer(self.rollout_worker.cost_function,
                                      self.plan_horizon * self.action_dim,
                                      max_iters=max_iters,
                                      population_size=population_size,
                                      num_elites=num_elites,
                                      lower_bound=lower_bound,
                                      upper_bound=upper_bound)

    # def cost_function(self, cur_state, action_trajs):
    #     env = self.env
    #     env.reset()
    #     action_trajs = action_trajs.reshape([-1, self.plan_horizon, self.action_dim])
    #     n = action_trajs.shape[0]
    #     costs = []
    #     print('evalute trajectories...')
    #     for i in tqdm(range(n)):
    #         env.set_state(cur_state)
    #         ret = 0
    #         for j in range(self.plan_horizon):
    #             _, reward, _, _ = env.step(action_trajs[i, j, :])
    #             ret += reward
    #         costs.append(-ret)
    #     return costs

    def reset(self, prev_sol = None, mean_coeff = 2, prev_var=None, var_coeff = 4):
        self.prev_sol = prev_sol
        self.mean_coeff = mean_coeff
        self.prev_var = prev_var
        self.var_coeff = var_coeff
        
    # def get_action(self, state):
    #     if len(self.action_buffer) > 0 and self.use_mpc:
    #         action, self.action_buffer = self.action_buffer[0], self.action_buffer[1:]
    #         return action
    #     self.env.debug = False
    #     env_state = self.env.get_state()

    #     soln, cost_his= self.optimizer.obtain_solution(env_state, self.prev_sol)
    #     soln = soln.reshape([-1, self.action_dim])
    #     if self.use_mpc:
    #         self.prev_sol = np.vstack([np.copy(soln)[1:, :], np.zeros([1, self.action_dim])]).flatten()
    #     else:
    #         self.prev_sol = None
    #         self.action_buffer = soln[1:]  # self.action_buffer is only needed for the non-mpc case.
    #     self.env.set_state(env_state)  # Recover the environment
    #     print("cem finished planning!")
    #     return soln[0], cost_his

    def get_action(self, state=None, ):
        if len(self.action_buffer) > 0 and self.use_mpc:
            action, self.action_buffer = self.action_buffer[0], self.action_buffer[1:]
            return action
        self.env.debug = False
        env_state = self.env.get_state()

        soln, solvar, cost_his= self.optimizer.obtain_solution(cur_state = env_state, mean_coeff = self.mean_coeff, var_coeff = self.var_coeff, init_mean = self.prev_sol, init_var = self.prev_var)
        soln = soln.reshape([-1, self.action_dim])
        solvar = solvar.reshape([-1, self.action_dim])
        if self.use_mpc:
            self.prev_sol = soln.flatten() #np.vstack([np.copy(soln)[1:, :], np.zeros([1, self.action_dim])]).flatten()
        else:
            self.prev_sol = None
            self.action_buffer = soln[1:]  # self.action_buffer is only needed for the non-mpc case.
        self.env.set_state(env_state)  # Recover the environment
        print("cem finished planning!")
        return soln, solvar, cost_his


if __name__ == '__main__':
    import gym
    import softgym
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--replay", action="store_true")
    parser.add_argument("-mpc", "--use_mpc", action="store_true")
    parser.add_argument("--traj_path", default="./data/folding_traj/traj.pkl")
    args = parser.parse_args()
    traj_path = args.traj_path

    softgym.register_flex_envs()
    # env = gym.make('ClothFlattenPointControl-v0')
    env = gym.make('ClothFoldSphereControl-v0')

    if not args.replay:
        policy = CEMPolicy(env,
                           args.use_mpc,
                           plan_horizon=20,
                           max_iters=5,
                           population_size=50,
                           num_elites=5)
        # Run policy
        obs = env.reset()
        initial_state = env.get_state()
        action_traj = []
        for _ in range(env.horizon):
            action = policy.get_action(obs)
            action_traj.append(copy.copy(action))
            obs, reward, _, _ = env.step(action)
            print('reward:', reward)

        traj_dict = {
            'initial_state': initial_state,
            'action_traj': action_traj
        }

        with open(traj_path, 'wb') as f:
            pickle.dump(traj_dict, f)
    else:
        with open(traj_path, 'rb') as f:
            traj_dict = pickle.load(f)
        initial_state, action_traj = traj_dict['initial_state'], traj_dict['action_traj']
        env.start_record(video_path='./data/videos/', video_name='cem_folding.gif')
        env.reset()
        env.set_state(initial_state)
        for action in action_traj:
            env.step(action)
        env.end_record()
    # Save the trajectories and replay
