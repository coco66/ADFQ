import numpy as np
from numpy import linalg as LA
import os, pdb

from envs.maps import map_utils
import envs.env_utils as util

from envs.agent_models import Agent
from envs.target_tracking.infoplanner.infoplanner_binding import Configure, Policy
from envs.target_tracking.metadata import METADATA
from envs.target_tracking.target_tracking import TargetTrackingEnv1

class BeliefWrapper(object):
    def __init__(self, num_targets=1, dim=4):
        self.num_targets = num_targets
        self.dim = dim
        self.state = None
        self.cov = None

    def update(self, state, cov):
        self.state = np.reshape(state, (self.num_targets, self.dim))
        self.cov = [cov[n*self.dim: (n+1)*self.dim,n*self.dim: (n+1)*self.dim ] for n in range(self.num_targets)]

class TargetWrapper(object):
    def __init__(self, num_targets=1, dim=4):
        self.state = None
        self.num_targets = num_targets
        self.dim = dim

    def reset(self, target):
        self.target = target
        self.state = np.reshape(self.target.getTargetState(), (self.num_targets, self.dim))

    def update(self):
        self.target.forwardSimulate(1)
        self.state = np.reshape(self.target.getTargetState(), (self.num_targets, self.dim))

class TargetTrackingInfoPlanner1(TargetTrackingEnv1):
    """
    Double Integrator
    """
    def __init__(self, num_targets=1, map_name='empty', is_training=True, known_noise=True):
        TargetTrackingEnv1.__init__(self, num_targets=num_targets,
            map_name=map_name, is_training=is_training, known_noise=known_noise)
        self.id = 'TargetTracking-info1'

        map_dir_path = '/'.join(map_utils.__file__.split('/')[:-1])
        # Setup Ground Truth Target Simulation
        map_nd = IGL.map_nd(self.MAP.mapmin, self.MAP.mapmax, self.MAP.mapres)
        if self.MAP.map is None:
            cmap_data = list(map(str, [0] * map_nd.size()[0] * map_nd.size()[1]))
        else:
            cmap_data = list(map(str, np.squeeze(self.MAP.map.astype(np.int8).reshape(-1, 1)).tolist()))
        se2_env = IGL.SE2Environment(map_nd, cmap_data, os.path.join(map_dir_path,'mprim_SE2_RL.yaml'))

        self.cfg = Configure(map_nd, cmap_data)
        sensor = IGL.RangeBearingSensor(self.sensor_r, self.fov, self.sensor_r_sd, self.sensor_b_sd, map_nd, cmap_data)
        self.agent = Agent_InfoPlanner(dim=3, sampling_period=self.sampling_period, limit=self.limit['agent'],
                            collision_func=lambda x: map_utils.is_collision(self.MAP, x),
                            se2_env=se2_env, sensor_obj=sensor)
        self.belief_targets = BeliefWrapper(num_targets)
        self.targets = TargetWrapper(num_targets)
        self.reset_num = 0

    def get_init_pose(self, init_random=True):
        if init_random:
            init_pose = {}
            if self.MAP.map is None:
                a_init = self.agent_init_pos[:2]
            else:
                isvalid = False
                while(not isvalid):
                    a_init = np.random.random((2,))*(self.MAP.mapmax-self.MAP.mapmin) + self.MAP.mapmin
                    isvalid = not(map_utils.is_collision(self.MAP, a_init))

            init_pose['agent'] = [a_init[0], a_init[1], np.random.random()*2*np.pi-np.pi]
            init_pose['targets'] = []
            init_pose['belief_targets'] = []
            for i in range(self.num_targets):
                isvalid = False
                while(not isvalid):
                    rand_ang = np.random.rand()*2*np.pi - np.pi
                    t_r = METADATA['init_distance']
                    t_init = np.array([t_r*np.cos(rand_ang), t_r*np.sin(rand_ang)]) + a_init
                    if (np.sqrt(np.sum((t_init - a_init)**2)) < METADATA['margin']):
                        isvalid = False
                    else:
                        isvalid = not(map_utils.is_collision(self.MAP, t_init))

                init_pose['targets'].append([t_init[0], t_init[1], rand_ang])
                init_pose['belief_targets'].append(np.concatenate((t_init + 10*(np.random.rand(2)-0.5), [0.0])))
        else:
            import pickle
            given_file = pickle.load(open('results/target_tracking/test_init_log.pkl','rb'))
            init_pose = given_file[self.reset_num]
            self.reset_num += 1
        return init_pose

    def reset(self, init_random=True):
        self.state = []
        t_init_sets = []
        t_init_b_sets = []
        init_pose = self.get_init_pose(init_random=init_random)
        a_init_igl = IGL.SE3Pose(init_pose['agent'], np.array([0, 0, 0, 1]))

        for i in range(self.num_targets):
            t_init_b_sets.append(init_pose['belief_targets'][i][:2])
            t_init_sets.append(init_pose['targets'][i][:2])
            r, alpha, _ = util.relative_measure(t_init_b_sets[-1][:2], init_pose['agent'])
            logdetcov = np.log(LA.det(self.target_init_cov*np.eye(self.target_dim)))
            self.state.extend([r, alpha, 0.0, 0.0, logdetcov, 0.0])

        self.state.extend([self.sensor_r, np.pi])
        self.state = np.array(self.state)
        # Build a target
        target = self.cfg.setup_integrator_targets(n_targets=self.num_targets, init_pos=t_init_sets,
                                                init_vel=self.target_init_vel, q=METADATA['const_q_true'], max_vel=METADATA['target_vel_limit'])  # Integrator Ground truth Model
        belief_target = self.cfg.setup_integrator_belief(n_targets=self.num_targets, q=METADATA['const_q'],
                                                init_pos=t_init_b_sets,
                                                cov_pos=self.target_init_cov, cov_vel=self.target_init_cov,
                                                init_vel=(0.0, 0.0))
         # Build a robot
        self.agent.reset(a_init_igl, belief_target)
        self.targets.reset(target)
        self.belief_targets.update(self.agent.get_belief_state(), self.agent.get_belief_cov())
        return np.array(self.state)

    def get_reward(self, obstacles_pt, observed, is_training=True):
        if obstacles_pt is None:
            penalty = 0.0
        else:
            penalty = METADATA['margin2wall']**2 * \
                        1./max(METADATA['margin2wall']**2, obstacles_pt[0]**2)

        if sum(observed) == 0:
            reward = - penalty
        else:
            cov = self.agent.get_belief_cov()
            detcov = [LA.det(cov[self.target_dim*n: self.target_dim*(n+1), self.target_dim*n: self.target_dim*(n+1)]) for n in range(self.num_targets)]
            reward = - 0.1 * np.log(np.mean(detcov) + np.std(detcov)) - penalty
            reward = max(0.0, reward) + np.mean(observed)
        test_reward = None

        if not(is_training):
            cov = self.agent.get_belief_cov()
            logdetcov = [np.log(LA.det(cov[self.target_dim*n: self.target_dim*(n+1), self.target_dim*n: self.target_dim*(n+1)])) for n in range(self.num_targets)]
            test_reward = -np.mean(logdetcov)

        return reward, False, test_reward

    def step(self, action):
        self.agent.update(action, self.targets.state)

        # Update the true target state
        self.targets.update()
        # Observe
        measurements = self.agent.observation(self.targets.target)
        obstacles_pt = map_utils.get_cloest_obstacle(self.MAP, self.agent.state)
        # Update the belief of the agent on the target using KF
        GaussianBelief = IGL.MultiTargetFilter(measurements, self.agent.agent, debug=False)
        self.agent.update_belief(GaussianBelief)
        self.belief_targets.update(self.agent.get_belief_state(), self.agent.get_belief_cov())

        observed = [m.validity for m in measurements]
        reward, done, test_reward = self.get_reward(obstacles_pt, observed, self.is_training)
        if obstacles_pt is None:
            obstacles_pt = (self.sensor_r, np.pi)

        self.state = []
        target_b_state = self.agent.get_belief_state()
        target_b_cov = self.agent.get_belief_cov()
        for n in range(self.num_targets):
            r_b, alpha_b, _ = util.relative_measure(target_b_state[self.target_dim*n: self.target_dim*n+2],
                                                self.agent.state)
            rel_target_vel = util.coord_change2b(target_b_state[self.target_dim*n: self.target_dim*n+2],
                                                        alpha_b + self.agent.state[-1])
            self.state.extend([r_b, alpha_b,
                                rel_target_vel[0], rel_target_vel[1],
                                    np.log(LA.det(target_b_cov[self.target_dim*n: self.target_dim*(n+1), self.target_dim*n: self.target_dim*(n+1)])),
                                        float(observed[n])])

        self.state.extend([obstacles_pt[0], obstacles_pt[1]])
        self.state = np.array(self.state)
        return self.state, reward, done, {'test_reward': test_reward}

class Agent_InfoPlanner(Agent):
    def __init__(self,  dim, sampling_period, limit, collision_func,
                    se2_env, sensor_obj, margin=METADATA['margin']):
        Agent.__init__(self, dim, sampling_period, limit, collision_func, margin=margin)
        self.se2_env = se2_env
        self.sensor = sensor_obj
        self.sampling_period = sampling_period
        self.action_map = {}
        for (i,v) in enumerate([3,2,1,0]):
            for (j,w) in enumerate([np.pi/2, 0, -np.pi/2]):
                self.action_map[3*i+j] = (v,w)

    def reset(self, init_state, belief_target):
        self.agent = IGL.Robot(init_state, self.se2_env, belief_target, self.sensor)
        self.state = self.get_state()
        return self.state

    def update(self, action, target_state):
        action =  self.update_filter(action, target_state)
        self.agent.applyControl([int(action)], 1)
        self.state = self.get_state()

    def get_state(self):
        return np.concatenate((self.agent.getState().position[:2], [self.agent.getState().getYaw()]))

    def get_state_object(self):
        return self.agent.getState()

    def observation(self, target_obj):
        return self.agent.sensor.senseMultiple(self.get_state_object(), target_obj)

    def get_belief_state(self):
        return self.agent.tmm.getTargetState()

    def get_belief_cov(self):
        return self.agent.tmm.getCovarianceMatrix()

    def update_belief(self, GaussianBelief):
        self.agent.tmm.updateBelief(GaussianBelief.mean, GaussianBelief.cov)

    def update_filter(self, action, target_state):
        state = self.get_state()
        control_input = self.action_map[action]
        tw = self.sampling_period*control_input[1]
        # Update the agent state
        if abs(tw) < 0.001:
            diff = np.array([self.sampling_period*control_input[0]*np.cos(state[2]+tw/2),
                    self.sampling_period*control_input[0]*np.sin(state[2]+tw/2),
                    tw])
        else:
            diff = np.array([control_input[0]/control_input[1]*(np.sin(state[2]+tw) - np.sin(state[2])),
                    control_input[0]/control_input[1]*(np.cos(state[2]) - np.cos(state[2]+tw)),
                    tw])
        new_state = state + diff
        if len(target_state.shape)==1:
            target_state = [target_state]
        target_col = False
        for n in range(target_state.shape[0]): # For each target
            target_col = np.sqrt(np.sum((new_state[:2] - target_state[n][:2])**2)) < METADATA['margin']
            if target_col:
                break

        if self.collision_check(new_state) or target_col: # no update
            new_action = 9 + action%3
        else:
            new_action = action
        return new_action
