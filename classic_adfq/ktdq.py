from adfq import Base
import numpy as np
from scipy.linalg import cholesky
import seeding
import copy
import util

class KTD_Q(Base):
	def __init__(self,env_name, discount, init_mean=None, init_var = 10.0, TH=None):
		"""KTD-Q
		Geist, Matthieu, and Olivier Pietquin. "Kalman temporal differences." Journal of artificial intelligence research 39 (2010): 483-532.
		https://www.jair.org/index.php/jair/article/view/10675/25513

		Parameters
		----------
		env_name : experimental domain name in models.py
    	discount : the discount factor in MDP
    	init_mean : initial mean for the mean parameters. Scalar - initialize with the same value
    	init_var : initial variance for the variance parameters. Scalar - initialize with the same value
    	TH : finite-time horizon (maximum learning steps)
	    """
		Base.__init__(self, env_name, discount, TH, memory_size=None)

		self.phi_func = self.env.phi[0]
		self.dim = self.env.phi[1]
		if init_mean is None:
			self.init_params()
		else:
			self.means = init_mean*np.ones(self.dim,dtype=np.float) # row vector
		self.cov = init_var*np.eye(self.dim)
		self.step = 0
		self.t_history = []

	def update(self, state, action, state_n, reward, done, epsilon):
		# Prediction Step
		pre_mean = self.means
		pre_cov = self.cov + self.eta*np.eye(self.dim)

		"""Sigma Point Computation:
		"""
		sig_th, W = sigma_points(pre_mean,pre_cov,self.kappa)
		#sig_R = np.matmul(sig_th, self.phi_func(state,action)) \
		#		 - int(not done)*self.discount*np.max([np.matmul(sig_th, self.phi_func(state_n, b)) for b in range(self.env.anum)], axis=0)
		sig_R = np.matmul(sig_th, self.phi_func(state,action)) \
				 - self.discount*np.max([np.matmul(sig_th, self.phi_func(state_n, b)) for b in range(self.env.anum)], axis=0)
		r_est = np.dot(W, sig_R)
		cov_th_r = np.matmul(W*(sig_R-r_est),(sig_th-pre_mean))
		cov_r = self.obs_noise + np.dot(W, (sig_R-r_est)**2)

		"""Correction Step:
		"""
		K = cov_th_r/cov_r
		self.means = pre_mean + K*(reward-r_est)
		self.cov = pre_cov - cov_r*np.outer(K,K)
		self.cov = 0.5*self.cov +0.5*np.transpose(self.cov) + epsilon*np.eye(self.dim)

	def learning(self, actionPolicy, actionParam=None, kappa=1.0, eta=0.0, obs_noise=1.0, epsilon = 1e-05, eval_greedy=False, draw = False):
		"""training KTD-Q
		Parameters
		----------
		actionPolicy : action policy
		actionParam : a hyperparameter for the chosen action policy if necessary
		kappa : the hyperparameter determining the number of sigma points.
		eta : evolution noise
		obs_noise : observation noise
		epsilon : for sigma point stability
		eval_greedy : True to evaluate the current policy during learning
		draw : True to print out simulation (grid and maze domains)
		"""
		if len(self.rewards)==self.env.timeH:
			print("The object has already learned")
			return None
		self.Q_target = np.array(self.env.optQ(self.discount))
		self.kappa = float(kappa)
		self.eta = eta
		self.obs_noise = obs_noise
		state = self.env.reset(self.np_random)
		t = 0 # This is "step" in Inv_pendulum and self.step is episode.
		while( self.step < self.env.timeH):
			if self.step%(int(self.env.timeH/util.EVAL_NUM)) == 0:
				self.Q_err.append(self.err())

			if actionPolicy == "active":
				action = self.active_learning(state,kappa)
			elif actionPolicy == "egreedy":
				action = self.get_action_eps(state, kappa, actionParam)
			elif actionPolicy == "offline":
				action = actionParam[self.step]
			elif actionPolicy == "random":
				action = self.np_random.choice(range(self.env.anum))
			else:
				print("You must choose between egreedy, active, or offline for the action selection.")
				break
			reward, state_n, done = self.env.observe(state,action,self.np_random)
			self.update(state, action, state_n, reward, done, epsilon = epsilon)

			self.states.append(state)
			self.actions.append(action)
			self.rewards.append(reward)
			if draw:
				self.draw(state,action,t,r)

			self.visits[state][action] += 1

			if eval_greedy and ((self.step+1)%(int(self.env.timeH/util.EVAL_NUM)) == 0):
				count, rew, _, _= self.greedy_policy(lambda x : self.get_action_eps(x, kappa, util.EVAL_EPS))
				self.test_counts.append(count)
				self.test_rewards.append(rew)
			state = self.env.reset(self.np_random) if done else state_n
			self.step += 1

	def learning_cartpole(self,kappa, eta=0.0, obs_noise=1.0, epsilon = 1e-05):
		"""training KTD-Q in cartpole
		Parameters
		----------
		kappa : the hyperparameter determining the number of sigma points.
		eta : evolution noise
		obs_noise : observation noise
		epsilon : for sigma point stability
		"""
		assert(self.env.name == 'inv_pendulum')

		state = self.env.reset(self.np_random)
		self.kappa = float(kappa)
		self.eta = eta
		self.obs_noise = obs_noise
		step = 0
		episode = 0
		while(episode<self.env.timeH):
			action = np.random.choice(self.env.anum,)
			reward, state_n, done = self.env.observe(state,action,self.np_random)
			self.update(state, action, state_n, reward, done, epsilon = epsilon)

			self.states.append(state)
			self.actions.append(action)
			self.rewards.append(reward)
			state = state_n
			step += 1
			if done or (step > self.env.step_bound):
				self.t_history.append(step)
				state = self.env.reset(self.np_random)
				if episode%50 == 0:
					count, rew, count_sd, _ = self.greedy_policy(lambda x : self.get_action_eps(x, kappa, 0.0),
																	step_bound = self.env.step_bound, num_itr=100)
					self.test_counts.append(count)
					self.test_rewards.append(rew)
					print("After %d steps, Episode %d : %.2f, SD: %.2f"%(step, episode, count, count_sd))
				episode += 1
				step = 0

	def learning_cartpole_gym(self,kappa, eta=0.0, obs_noise=1.0):
		env = gym.make('CartPole-v0')
		state = env.reset()
		self.kappa = float(kappa)
		self.eta = eta
		self.obs_noise = obs_noise
		step = 0
		episode = 0
		num_itr = 100
		while(episode<self.env.timeH):
			action = np.random.choice(self.env.anum,)
			env.render()
			state_n, reward, done, _ = env.step(action)
			self.update(state[-2:], action, state[-2:], reward, done)

			self.states.append(state)
			self.actions.append(action)
			self.rewards.append(reward)
			state = state_n
			step += 1
			if done or (step > self.env.step_bound):
				self.t_history.append(step)
				state = env.reset()
				if episode%50 == 0:
					test_env = gym.make('CartPole-v0')
					step_bound = self.env.step_bound
					t_total, reward_total, it = 0,0,0
					while(it<num_itr):
						t = 0
						s_test = test_env.reset() #np_random_local.choice(range(self.env.anum))
						r_test = 0.0
						done = False
						while((not done) and (t<step_bound)):
							a_test = np.argmax([np.dot(self.means, self.phi_func(s_test[-2:], a)) for a in range(self.env.anum)])
							sn_test, r, done, _ = test_env.step(a_test)
							s_test = sn_test
							r_test += r
							t +=1
						reward_total += r_test
						t_total += t
						it += 1
					self.test_counts.append(t_total/float(num_itr))
					self.test_rewards.append(reward_total/float(num_itr))
					print("After %d steps, Episode %d : %d"%(step, episode, self.test_counts[-1]))
				episode += 1
				step = 0

	def get_action_eps(self,state,kappa,eps):
		if self.np_random.rand() < eps:
			return self.np_random.choice(range(self.env.anum))
		else:
			Q = [np.dot(self.means, self.phi_func(state, a)) for a in range(self.env.anum)]
			return np.argmax(Q)

	def active_learning(self, state, kappa):
		"""Active Learning Scheme (Section 6 in the main paper)
		"""
		sig_th, W = sigma_points(self.means, self.cov, kappa)
		if sig_th is None:
			return None
		Q_mean=[np.dot(W,np.matmul(sig_th, self.phi_func(state,a))) for a in range(self.env.anum)]
		Q_var =[np.dot(W,(np.matmul(sig_th, self.phi_func(state,a)) - Q_mean[a])**2) for a in range(self.env.anum)]
		rand_num = np.random.rand(1)[0]
		prob = np.sqrt(Q_var)
		prob = prob / sum(prob)
		cumsum = 0
		for (i,v) in enumerate(prob):
			cumsum += v
			if rand_num <= cumsum:
				action = i
				break
		return action

	def get_total_reward(self):
		return sum(self.rewards)

	def get_visits(self):
		return self.visits

def sample_sigma_points(mean, variance, kappa):
    n = len(mean)
    X = np.empty((2 * n + 1, n))
    X[:, :] = mean[None, :]
    C = np.linalg.cholesky((kappa + n) * variance)
    for j in range(n):
        X[j + 1, :] += C[:, j]
        X[j + n + 1, :] -= C[:, j]
    W = np.ones(2 * n + 1) * (1. / 2 / (kappa + n))
    W[0] = (kappa / (kappa + n))
    return X, W

def sigma_points(mean, cov_in, k):
	cov = copy.deepcopy(cov_in)
	n = np.prod(mean.shape)
	count = 0

	chol_t = (cholesky((n+k)*cov)).T # array form cov
	m = np.reshape(mean, (n,1))
	sigs = np.concatenate((m, m+chol_t),axis=1)
	sigs = np.concatenate((sigs, m-chol_t),axis=1)

	W = 0.5/(k+n)*np.ones(n*2+1)
	W[0] = k / float(k + n)

	return sigs.T, W

def isPostiveDefinite(x):
	return np.all(np.linalg.eigvals(x) > 0)
