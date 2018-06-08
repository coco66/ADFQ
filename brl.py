""""""""""""""""""""" 
ADFQ and KTD-Q
"""""""""""""""""""""

import models
import numpy as np
from scipy.stats import norm
from scipy.linalg import cholesky
import time
import sys
import random
import seeding
import copy

import brl_util as util
import pdb

class BRL(object):
	def __init__(self, scene, discount, TH, useGym=False, memory_size=None):
		"""BRL base class.
    	Parameters
    	----------
    	scene : experimental domain name in models.py
    	discount : the discount factor in MDP
    	TH : finite-time horizon (maximum learning steps) 
    	useGym : True if you are experimenting with OpenAI gym
    	memory_size : Experience Replay memory size
        """
		self.scene = scene
		if useGym:
			import gym
			self.env = gym.make(scene)
		else:
			self.env = models.model_assign(scene)
		self.discount = discount
		self.states = []
		self.actions = []
		self.rewards = []
		self.np_random,_  = seeding.np_random(None)
		self.test_counts = []
		self.test_rewards = []
		self.Q_err = []
		self.visits = np.zeros((self.env.snum,self.env.anum))
		self.useGym = useGym
		self.memory_size = memory_size
		self.replayMem ={(i,j):[] for i in range(self.env.snum) for j in range(self.env.anum)}

		if not(TH==None):
			self.env.set_time(TH)

	def get_visits(self):
		return self.visits

	def get_total_reward(self):
		return sum(self.rewards)

	def err(self):
		"""Computing RMSE of Q 
    	"""
		mean_eval = np.reshape(self.means, (self.env.snum, self.env.anum) )
		return np.sqrt(np.mean((self.Q_target[self.env.eff_states,:] - mean_eval[self.env.eff_states,:])**2))

	def draw(self,s,a,t,r):
		"""Print out simulation.
    	"""	
		print "s:",s,"t:",t,"Reward:",r,"Total Reward:",sum(self.rewards)+r
		self.env.plot(s,a)
		print "====="
		time.sleep(0.5)	

	def greedy_policy(self, get_action_func, step_bound = None, num_itr = util.EVAL_RUNS):
		"""Evaluation during learning
		Parameters
    	----------
    		get_action_func : a function for an evaluation action policy
    		step_bound : the maximum number of steps for each evaluation
    		num_itr : the number of iterations
		"""
		if step_bound is None:
			step_bound = self.env.timeH/util.EVAL_STEPS
		counts = [] 
		rewards = []
		itr = 0 
		while(itr<num_itr):
			t = 0
			state = self.env.reset(self.np_random) 
			reward = 0.0
			done = False
			while((not done) and (t<step_bound)):
				action = get_action_func(state)
				r, state_n, done = self.env.observe(state,action,self.np_random)
				state = state_n
				reward += r
				t +=1
			rewards.append(reward)
			counts.append(t)
			itr += 1
		return np.mean(counts), np.mean(rewards), np.std(counts), np.std(rewards)

	def init_params(self):
		"""Initialize parameters corresponding to Q values according the first reward 
		that a learning agent sees by random exploration.
		"""
		s = self.env.reset(self.np_random)
		while(True):
			a = self.np_random.choice(range(self.env.anum))
			r, s_n, done = self.env.observe(s,a,self.np_random)
			if r > 0: # First nonzero reward
				if self.env.episodic:
					self.means = r*np.ones(self.dim,dtype=np.float)
				else:
					self.means = r/(1-self.discount)*np.ones(self.dim,dtype=np.float)
				break
			else:
				if done:
					self.means = np.zeros(self.dim,dtype=np.float)
					break
				s = s_n

	def store(self, causality):
		"""Experience Replay - Store in a memory
		Parameters
		----------
			causality : a dictionary for the causality tuple (s,a,s',r,done)
		"""
		sa_pair = (causality['state'], causality['action'])
		if (len(self.replayMem[sa_pair]) == self.memory_size):
			self.replayMem[sa_pair].pop(0)
			self.replayMem[sa_pair].append(causality)
		else:
			self.replayMem[sa_pair].append(causality)

	def get_batch(self, s, a, batch_size):
		"""Return a random batch
		Parameters
		----------
			s : the current state
			a : the current action
			batch_size : the size of the batch
		"""
		minibatch = {'state':[], 'action':[], 'reward':[], 'state_n':[], 'terminal':[]}
		for _ in range(batch_size):
			d = self.replayMem[(s,a)][random.randint(0,len(self.replayMem[(s,a)])-1)]
			for (k,v) in minibatch.items():
				v.append(d[k])
		return minibatch

class adfq(BRL):
	def __init__(self,scene, discount, init_mean = None, init_var = 100.0, TH=None, useGym=False, memory_size = 200):
		"""ADFQ class object
		Parameters
		----------
			scene : experimental domain name in models.py
    		discount : the discount factor in MDP
    		init_mean : initial mean for the mean parameters. Scalar - initialize with the same value
    		init_var : initial variance for the variance parameters. Scalar - initialize with the same value
    		TH : finite-time horizon (maximum learning steps) 
	    	memory_size : Experience Replay memory size
		"""
		BRL.__init__(self, scene, discount, TH, useGym=useGym, memory_size=memory_size)
		self.dim = (self.env.snum,self.env.anum)
		if init_mean is None:
			self.init_params()
		else:
			self.means = init_mean*np.ones(self.dim,dtype=np.float)
		self.vars = init_var*np.ones(self.dim,dtype=np.float)
		self.step = 0

	def learning(self, actionPolicy, actionParam, updatePolicy='adfq', eval_greedy = False, draw = False, 
					varTH = 1e-10, updateParam=None, asymptotic=False, asymptotic_trigger=1e-8, 
					useScale=False, noise=0.0, batch_size=0):
		"""train with ADFQ
		Parameters
		----------
			actionPolicy : action policy. See "action_selection" function below.
			actionParam : a hyperparameter for the chosen action policy if necessary.
			updatePolicy : 'adfq' for the ADFQ algorithm. 'numeric' for the ADFQ-Numeric update. 'adfq-v2' for the ADFQ V2 update (appendix).
			eval_greedy : True to evaluate the current policy during learning.
			draw : True to print out the simulation (for grid and maze domains)
			varTH : variance thereshold 
			asymptotic : True to use the asymptotic update 
			asymptotic_trigger : a value to decide when to start the asymptotic update if "asymptotic==True"
			useScale : use the scaling trick.
			noise : for stochastic case, you can add a small noise to the variance[s,a]
			batch_size : batch size. 0 if you don't use experience replay.
		"""
		if len(self.rewards)==self.env.timeH:
			print("The object has already learned")
			return None

		if (actionPolicy == 'offline') and (len(actionParam) != self.env.timeH):
			print(len(actionParam), self.env.timeH)
			raise ValueError('The given action trajectory does not match with the number of learning steps.')

		np.random.seed()
		self.Q_target = np.array(self.env.optQ(self.discount))
		self.varTH = varTH
		
		if batch_size > 0:
			s = self.env.reset(self.np_random)
			while(len(self.replayMem[(0,0)]) < self.memory_size):
				a = np.random.choice(self.env.anum)
				r, s_n, done = self.env.observe(s,a,self.np_random)
				self.store({'state':s, 'action':a, 'reward':r, 'state_n':s_n, 'terminal':done})
		
		s = self.env.reset(self.np_random)
		self.log_scale = 0.0
		while(self.step < self.env.timeH):
			if self.step%(self.env.timeH/util.EVAL_NUM) == 0:
				self.Q_err.append(self.err())

			a = self.action_selection(s, actionPolicy, actionParam)

			# Observation
			r, s_n, done = self.env.observe(s,a,self.np_random)
			self.rewards.append(r)
			self.visits[s][a] += 1
			if batch_size > 0:
				self.store({'state':s, 'action':a, 'reward':r, 'state_n':s_n, 'terminal':done})
				batch = self.get_batch(s, a, batch_size)
				n_means = self.means[batch['state_n'],:]
				n_vars = self.vars[batch['state_n'],:]
				c_mean = self.means[batch['state'], batch['action']]
				c_var = self.vars[batch['state'], batch['action']]
				reward = batch['reward']
				terminal = batch['terminal']
			else:
				# Record
				self.states.append(s)
				self.actions.append(a)
				n_means = self.means[s_n]
				n_vars = self.vars[s_n]
				c_mean = self.means[s][a]
				c_var = self.vars[s][a]
				reward = r
				terminal = done
			# Update
			self.varTH = varTH/np.exp(self.log_scale, dtype=util.DTYPE)
			if (updatePolicy == 'adfq'):
				new_mean, new_var, _ = util.posterior_adfq(n_means, n_vars, c_mean, c_var, reward, self.discount,
					terminal, scale_factor = np.exp(self.log_scale, dtype=util.DTYPE), varTH =self.varTH, asymptotic=asymptotic, 
					asymptotic_trigger=asymptotic_trigger, noise=noise, batch = (batch_size>0))
			
			elif updatePolicy == 'numeric' :
				new_mean, new_var, _ = util.posterior_numeric( n_means, n_vars, c_mean, c_var, reward, self.discount,
					terminal, scale_factor = np.exp(self.log_scale, dtype=util.DTYPE), varTH = self.varTH, noise=noise, batch = (batch_size>0))	
			
			elif (updatePolicy == 'adfq-v2'):
				new_mean,new_var, _ = util.posterior_adfq_v2(n_means, n_vars, c_mean, c_var, reward, self.discount,
					terminal, scale_factor = np.exp(self.log_scale, dtype=util.DTYPE), varTH = self.varTH, asymptotic=asymptotic, 
					asymptotic_trigger=asymptotic_trigger, noise=noise, batch = (batch_size>0))	
				
			elif updatePolicy == 'hybrid':
				new_mean,new_var, _ = util.posterior_hybrid(n_means, n_vars, c_mean, c_var, reward, self.discount,
					terminal, scale_factor = np.exp(self.log_scale, dtype=util.DTYPE), varTH = self.varTH, noise=noise, batch = (batch_size>0))	
	
			else:
				raise ValueError("No such update policy")

			self.means[s][a] = np.mean(new_mean)
			self.vars[s][a] = np.mean(new_var) #np.maximum(self.varTH, new_var)

			if useScale:
				delta =  np.log(np.mean(self.vars[self.env.eff_states,:]))
				self.vars[self.env.eff_states,:] = np.exp(np.log(self.vars[self.env.eff_states,:]) - delta, dtype = np.float64)
				self.log_scale = np.maximum( -100.0, self.log_scale + delta)

			if draw:
				self.draw(s,a,self.step,r)

			if eval_greedy and ((self.step+1)%(self.env.timeH/util.EVAL_NUM) == 0):
				count, rew , _, _= self.greedy_policy(lambda x : self.get_action_egreedy(x, util.EVAL_EPS))
				self.test_counts.append(count)
				self.test_rewards.append(rew)
			s = self.env.reset(self.np_random) if done else s_n
			self.step += 1

	def action_selection(self, state, action_policy, param):
		"""Action Policies
			'egreedy': epsilon greedy. param = epsilon 
			'semi-Bayes': BS with (1-epsilon) probability and random with epsilon probability. param = epsilon
			'Bayes' : Bayesian (posterior) sampling. No parameter is required
			'uniform' : uniform random. No parameter is required
			'offline' : action trajectory is given before training. param = a set of actions (array)
		"""		
		if action_policy == 'egreedy':
			action = self.get_action_egreedy(state,param)
		elif action_policy == 'semi-Bayes':
			if self.np_random.rand(1)[0] < param:
				action = int(self.np_random.choice(range(self.env.anum)))
			else:
				action = self.get_action_Bayesian(state, self.log_scale)	
		elif action_policy == 'Bayes': 
			action = self.get_action_Bayesian(state, self.log_scale)
		elif action_policy == 'random':
			action = self.np_random.choice(range(self.env.anum))
		elif action_policy == 'offline':
			action = param[self.step]
		elif action_policy == 'vpi':
			action = self.vpi(state)

		return action

	def get_action_Bayesian(self, state, log_scale):
		if len(set(self.means[state]))==1:
			return int(self.np_random.choice(range(self.env.anum)))
		else:
			tmp  = self.np_random.normal(self.means[state],np.sqrt(self.vars[state]))
			return np.argmax(tmp)

 	def get_action_egreedy(self,state,epsilon):
 		if self.np_random.rand(1)[0] < epsilon: 
			return int(self.np_random.choice(range(self.env.anum)))
		else:
			return np.argmax(self.means[state])

	def get_action_eB(self,state,epsilon):
		# epsilon-greedy inspired
		if self.np_random.rand(1)[0] > (1-epsilon): 
			return int(self.np_random.choice(range(self.env.anum)))
		else:
			if (self.vars[state] < self.varTH).any():
				return np.argmax(self.means[state])
			if len(set(self.means[state]))==1:
				return  int(self.np_random.choice(range(self.env.anum)))
			else:
				tmp  = self.np_random.normal(self.means[state],np.sqrt(self.vars[state]))
				return np.argmax(tmp)
	def vpi(self,state):
		#pdb.set_trace()
		vpi_vals = np.zeros((self.env.anum,),dtype=np.float32)
		id_sorted = np.argsort(self.means[state,:])
		if self.means[state,id_sorted[-1]] == self.means[state,id_sorted[-2]]:
			if np.random.rand(1)[0] < 0.5:
				tmp = id_sorted[-1]
				id_sorted[-1] = id_sorted[-2]
				id_sorted[-2] = tmp
		# a = a_1
		best_a = id_sorted[-1]
		mu = self.means[state, best_a]
		sig  = np.sqrt(self.vars[state, best_a])
		vpi_vals[best_a] = self.means[state,id_sorted[-2]]* norm.cdf(self.means[state,id_sorted[-2]], mu, sig) \
			- mu*norm.cdf(self.means[state,id_sorted[-2]],mu, sig) + sig*sig*norm.pdf(self.means[state,id_sorted[-2]], mu, sig)
					#- mu + sig*sig*norm.pdf(self.means[state,id_sorted[-2]], mu, sig)/max(0.0001,norm.cdf(self.means[state,id_sorted[-2]],mu, sig))
					
		for a_id in id_sorted[:-1]:
			mu = self.means[state, a_id]
			sig = np.sqrt(self.vars[state, a_id])
			vpi_vals[a_id] = mu*(1-norm.cdf(self.means[state,best_a], mu, sig)) + sig*sig*norm.pdf(self.means[state, best_a], mu, sig) \
				- self.means[state, best_a]*(1-norm.cdf(self.means[state,best_a], mu, sig))
			#mu + sig*sig*norm.pdf(self.means[state, best_a], mu, sig)/max(0.0001,(1-norm.cdf(self.means[state,best_a], mu, sig))) \
					
		a_orders = np.argsort(vpi_vals)
		if vpi_vals[a_orders[-1]] == vpi_vals[a_orders[-2]]:
			return np.random.choice(a_orders[-2:])
		else:
			return np.argmax(vpi_vals+self.means[state,:])


class ktd_Q(BRL): 
	def __init__(self,scene, discount, init_mean=None, init_var = 10.0, TH=None, useGym=False):
		"""KTD-Q
		Geist, Matthieu, and Olivier Pietquin. "Kalman temporal differences." Journal of artificial intelligence research 39 (2010): 483-532.
		https://www.jair.org/index.php/jair/article/view/10675/25513

		Parameters
		----------
		scene : experimental domain name in models.py
    	discount : the discount factor in MDP
    	init_mean : initial mean for the mean parameters. Scalar - initialize with the same value
    	init_var : initial variance for the variance parameters. Scalar - initialize with the same value
    	TH : finite-time horizon (maximum learning steps) 
	    useGym : True if you use OpenAI gym
	    """
		BRL.__init__(self, scene, discount, TH, useGym=useGym, memory_size=None)

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
		
	def learning(self, actionPolicy, actionParam, kappa, eta=0.0, obs_noise=1.0, epsilon = 1e-05, eval_greedy=False, draw = False):
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
			if self.step%(self.env.timeH/util.EVAL_NUM) == 0:
				self.Q_err.append(self.err())

			if actionPolicy == "active":
				action = self.active_learning(state,kappa)
			elif actionPolicy == "egreedy":
				action = self.get_action_eps(state, kappa, actionParam)
			elif actionPolicy == "offline":
				action = actionParam[self.step]
			elif actionPolicy == "uniform":
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
			
			if eval_greedy and ((self.step+1)%(self.env.timeH/util.EVAL_NUM) == 0):
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


