""" 
<Q-Learning and Monte Carlo method algorithm for finite state and action spaces>

Author: Heejin Chloe Jeong (chloe.hjeong@gmail.com)
Affiliation: University of Pennsylvania
"""

import models as mdl
import numpy as np
import time 
import seeding
import pdb
import brl_util as util
import copy
import random
from scipy.misc import logsumexp
class Tabular(object):
	def __init__(self,scene,discount, initQ, TH, memory_size):
		"""Tabular RL
		Parameters
		----------
		scene : A name of a task you want to test. (See models.py)
		alpha : learning rate of Q-learning
		discount : discount factor in MDP
		initQ : initial Q value (initialize all Q values with the same number)
		TH : finite-time horizon (maximum learning steps)
		memory_size : Experience Replay memory size
		"""
		self.scene = scene
		self.env = mdl.model_assign(scene)
		self.discount = discount
		self.states, self.actions, self.rewards = [],[],[]
		self.visits = np.zeros((self.env.snum, self.env.anum),dtype=np.int)
		self.Q = initQ*np.ones((self.env.snum, self.env.anum),dtype=float)
		self.np_random, _  = seeding.np_random(None) 
		self.test_counts = []
		self.test_rewards = []
		if initQ is None:
			self.init_params()
		
		self.Q_err = []
		self.memory_size = memory_size
		self.replayMem ={(i,j):[] for i in range(self.env.snum) for j in range(self.env.anum)}

		if TH is not None:
			self.env.set_time(TH)

	def get_total_reward(self):
		return sum(self.rewards)

	def get_visits(self):
		return self.visits


	def draw(self,s,a,t,r):
		"""Print out simulation.
    	"""
		self.env.plot(s,a)
		print "s:",s,"t:",t,"Reward:",r,"Total Reward:",sum(self.rewards)+r
		print "B",self.Q[s]
		time.sleep(0.5)

	def err(self):
		"""Computing RMSE of Q 
    	"""
		return np.sqrt(np.mean((self.Q_target[self.env.eff_states,:] - self.Q[self.env.eff_states,:])**2))

	def init_params(self):
		"""Initialize parameters corresponding to Q values according the first reward 
		that a learning agent sees by random exploration.
		"""
		s = self.env.reset(self.np_random)
		while(True):
			a = self.np_random.choice(range(self.env.anum))
			rew, s_n, done = self.env.observe(s,a,self.np_random)
			if rew > 0: # First nonzero reward
				if self.env.episodic:
					self.Q = rew*np.ones(self.Q.shape,dtype=np.float)
				else:
					self.Q = rew/(1-self.discount)*np.ones(self.Q.shape, dtype=np.float)
				break
			else:
				if done:
					self.Q = np.zeros(self.Q.shape,dtype=np.float)
					break
			s = s_n

	def action_selection(self, state, actionPolicy, actionParam):
		"""Action Policies
			'egreedy': epsilon greedy. param = epsilon 
			'uniform' : uniform random. No parameter is required
			'softmax' : softmax action selection with Boltzmann distribution
			'offline' : action trajectory is given before training. param = a set of actions (array)
		"""	
		if actionPolicy == 'uniform':
				action = int(self.np_random.choice(self.env.anum,1))

		elif actionPolicy == 'egreedy':
			if (len(set(self.Q[state]))==1) or (self.np_random.rand(1)[0] < actionParam): # epsilon probability 
				action = int(self.np_random.choice(self.env.anum,1))
			else:
				action = np.argmax(self.Q[state])

		elif actionPolicy == 'softmax':
			if len(set(self.Q[state]))==1:
				action = int(self.np_random.choice(self.env.anum,1))
			else:
				action = -1
				x = self.Q[state]/actionParam
				logexpQ = x - logsumexp(x)
				expQ = np.exp(logexpQ)
				rand_num = self.np_random.rand(1)[0]
				cum = 0.0
				for (i,v) in enumerate(expQ):
					cum+=v
					if rand_num <= cum:
						action = i
						break	
				if action < 0:
					pdb.set_trace()
		elif actionPolicy == 'offline':
			action = actionParam[self.step]

		else:
			ValueError("You must choose between egreedy or softmax for the action selection.")

		return action

	def greedy_policy(self, get_action_func, step_bound=None, num_itr = util.EVAL_RUNS):
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
		for _ in xrange(batch_size):
			d = self.replayMem[(s,a)][random.randint(0,len(self.replayMem[(s,a)])-1)]
			for (k,v) in minibatch.items():
				v.append(d[k])
		return minibatch

	def get_action_egreedy(self,state,epsilon):
 		if self.np_random.rand(1)[0] > (1-epsilon): 
			return int(self.np_random.choice(range(self.env.anum)))
		else:
			return np.argmax(self.Q[state])

class Qlearning(Tabular):
	def __init__(self,scene,alpha,discount,initQ=None, TH=None, memory_size=50):
		Tabular.__init__(self,scene,discount,initQ, TH, memory_size )
		self.alpha = alpha # Learning Rate

	def learning(self, actionPolicy, actionParam, eval_greedy = False, draw = False, rate_decay=True, batch_size=0):
		"""train with Q-learning
		Parameters
		----------
			actionPolicy: 'uniform', 'egreedy', 'softmax', or 'offline'
			actionParam: necessary hyperparameters for a chosen action policy.
			eval_greedy: True or 1, if you want to evaluate greedily during the learning process
			draw: True or 1, if you want visualization
			rate_decay: learning rate decay 
			batch_size: batch size			
		"""
		if len(self.rewards)==self.env.timeH:
			print("The object has already learned")
			return None

		self.step = 0

		if batch_size > 0:
			s = self.env.reset(self.np_random)
			while(len(self.replayMem[(0,0)]) < self.memory_size):
				a = np.random.choice(self.env.anum)
				r, s_n, done = self.env.observe(s,a,self.np_random)
				self.store({'state':s, 'action':a, 'reward':r, 'state_n':s_n, 'terminal':done})
		state = self.env.reset(self.np_random)
		self.Q_target = self.env.optQ(self.discount)
		n_0 = round(0.01 * self.env.timeH /self.alpha / (1-0.01/self.alpha))
		Q_history = []
		while (self.step < self.env.timeH) :
			
			if self.step%(self.env.timeH/util.EVAL_NUM) == 0:
				self.Q_err.append(self.err())
				Q_history.append(copy.deepcopy(self.Q))

			action = self.action_selection(state, actionPolicy, actionParam)
			reward, state_n, done = self.env.observe(state,action,self.np_random)

			if batch_size > 0:
				self.store({'state':state, 'action':action, 'reward':reward, 'state_n':state_n, 'terminal':done})
				batch = self.get_batch(state, action, batch_size)
				target = np.mean( np.array(batch['reward']) + self.discount* (1 - np.array(batch['terminal'], dtype=int)) * np.max(self.Q[batch['state_n'],:], axis=-1))			
			else:
				self.states.append(state)
				self.actions.append(action)
				self.visits[state][action] += 1
				self.rewards.append(reward)
				target = reward+self.discount*int(not done)*max(self.Q[state_n])

			if rate_decay:
				alpha_t = self.alpha*n_0/(n_0+self.visits[state][action] )
			else:
				alpha_t = self.alpha

			new_q = (1-alpha_t)*self.Q[state][action] + alpha_t*target
			self.Q[state][action] = new_q
			
			if draw:
				self.draw(state,action,t,reward)
				pdb.set_trace()

			if eval_greedy and ((self.step+1)%(self.env.timeH/util.EVAL_NUM) == 0):
				count, rew, _, _= self.greedy_policy(lambda x : self.get_action_egreedy(x, util.EVAL_EPS))
				self.test_counts.append(count)
				self.test_rewards.append(rew)

			state = self.env.reset(self.np_random) if done else state_n
			self.step += 1
		self.Q_history = np.array(Q_history)

class MC(Tabular):
	def __init__(self,scene,discount,initQ, TH=None):
		Tabular.__init__(self,scene,discount,initQ, TH)
		if not(self.env.episode):
			raise ValueError("Learning Environment must be epsisodic.")

	def learning(self, actionPolicy, actionParam, eval_greedy = False,draw = False, rate_decay=True):
		"""train with Monte Carlo method
		Parameters
		----------
			actionPolicy: 'uniform', 'egreedy', 'softmax', or 'offline'
			actionParam: necessary hyperparameters for a chosen action policy.
			eval_greedy: True or 1, if you want to evaluate greedily during the learning process
			draw: True or 1, if you want visualization
			rate_decay: learning rate decay 
		"""

		if len(self.rewards)==self.env.timeH:
			print("The object has already learned")
			return None

		if (actionPolicy=='offline') and (len(actionParam) != self.env.timeH):
			raise ValueError('The given action trajectory does not match with the number of learning steps.')

		self.step = 0
		
		while (self.step < self.env.timeH) :
			self.Q_err.append(self.err())
			epsiode = self.sample_episode(self.get_action_egreedy)
			epLen = len(epsiode['state'])
			gammas = np.power(gamma, range(0,epLen))

			for i in xrange(epLen):
				G = np.dot(epsiode['reward'][i:], gammas[i:]) if i > (epLen-200) else 0
				currQ = self.Q[epsiode['state'][i], epsiode['action'][i]]
				if self.visits[epsiode['state'][i], epsiode['action'][i]] == 0:
					pdb.set_trace()
				self.Q[epsiode['state'][i], epsiode['action'][i]] = currQ + 1.0*(G-currQ)/self.visits[epsiode['state'][i], epsiode['action'][i]]

			if eval_greedy and ((self.step+1)%(self.env.timeH/util.EVAL_NUM) == 0):
				count, rew, _, _= self.greedy_policy(lambda x : self.get_action_egreedy(x, util.EVAL_EPS))
				self.test_counts.append(count)
				self.test_rewards.append(rew)
			self.step += 1

	def sample_episode(self, action_policy):
		episode = {'state':[], 'action':[], 'reward':[]}
		done = False
		state = self.env.reset(self.np_random)
		while(not done):
			episode['state'].append(state)
			action = action_policy(state, self.n0/(self.n0+sum(self.visits[state])))
			self.visits[state][action] += 1
			reward, state_n, done = self.env.observe(state, action)
			episode['action'].append(action)
			episode['reward'].append(reward)
			state = state_n
		return episode








