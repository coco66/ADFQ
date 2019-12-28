from adfq import ADFQ
import numpy as np
from scipy.stats import norm
import time
import copy
import envs

import util
import adfq_fun

class ADFQ_dynamic(ADFQ):
	def __init__(self,env_name, discount, init_mean = None, init_var = 100.0, TH=None, memory_size = 200):
		"""ADFQ class object for changing environment
		Parameters
		----------
			env_name : experimental domain name in models.py
    		discount : the discount factor in MDP
    		init_mean : initial mean for the mean parameters. Scalar - initialize with the same value
    		init_var : initial variance for the variance parameters. Scalar - initialize with the same value
    		TH : finite-time horizon (maximum learning steps)
	    	memory_size : Experience Replay memory size
		"""
		adfq.__init__(self,env_name, discount, init_mean = init_mean, init_var =init_var, TH=TH, memory_size = memory_size)

	def learning(self, actionPolicy, actionParam, updatePolicy='adfq', eval_greedy = False, draw = False,
					varTH = 1e-10, updateParam=None, asymptotic=False, asymptotic_trigger=1e-8,
					useScale=False, noise=0.0, batch_size=0, change=True, beta = 0.0):
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

		records = {'t':[],'k':[], 'var':[], 'mean':[]}

		if batch_size > 0:
			s = self.env.reset(self.np_random)
			while(len(self.replayMem[(0,0)]) < self.memory_size):
				a = np.random.choice(self.env.anum)
				r, s_n, done = self.env.observe(s,a,self.np_random)
				self.store({'state':s, 'action':a, 'reward':r, 'state_n':s_n, 'terminal':done})

		s = self.env.reset(self.np_random)
		self.log_scale = 0.0

		temp = []
		while(self.step < self.env.timeH):
			if change and (self.step == self.env.changePt):# 0.5*self.env.timeH):
				self.env.change()
				self.Q_target = np.array(self.env.optQ(self.discount, changed=True))

			if self.step%(int(self.env.timeH/util.EVAL_NUM)) == 0:
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
				new_mean, new_var, stats = adfq_fun.posterior_adfq(n_means, n_vars, c_mean, c_var, reward, self.discount,
					terminal, scale_factor = np.exp(self.log_scale, dtype=util.DTYPE), varTH =self.varTH, asymptotic=asymptotic,
					asymptotic_trigger=asymptotic_trigger, noise=noise, batch = (batch_size>0))

			elif updatePolicy == 'numeric' :
				new_mean, new_var, _ = adfq_fun.posterior_numeric( n_means, n_vars, c_mean, c_var, reward, self.discount,
					terminal, scale_factor = np.exp(self.log_scale, dtype=util.DTYPE), varTH = self.varTH, noise=noise, batch = (batch_size>0))

			elif (updatePolicy == 'adfq-v2'):
				new_mean,new_var, stats = adfq_fun.posterior_adfq_v2(n_means, n_vars, c_mean, c_var, reward, self.discount,
					terminal, scale_factor = np.exp(self.log_scale, dtype=util.DTYPE), varTH = self.varTH, asymptotic=asymptotic,
					asymptotic_trigger=asymptotic_trigger, noise=noise, batch = (batch_size>0))

			elif updatePolicy == 'hybrid':
				new_mean,new_var, _ = adfq_fun.posterior_hybrid(n_means, n_vars, c_mean, c_var, reward, self.discount,
					terminal, scale_factor = np.exp(self.log_scale, dtype=util.DTYPE), varTH = self.varTH, noise=noise, batch = (batch_size>0))

			else:
				raise ValueError("No such update policy")

			td_err = reward + self.discount*n_means - c_mean#np.clip(np.abs(reward + self.discount*n_means - c_mean), 0.1, 10.0)
			add_vars = c_var + self.discount**2*n_vars
			#penalty = np.dot(stats[2], norm.cdf(td_err,0.0, 0.001*np.sqrt(add_vars)))-0.5
			#penalty = 50*(np.tanh(0.1*(np.dot(stats[2],td_err**2/add_vars)-50.0))+1.0)
			gate_bound = 1.0
			penalty = np.dot(stats[2],td_err**2/add_vars)
			gate_const = 1.0 if penalty > gate_bound else 0.0
			#penalty *= gate_const
			steepness = 0.01
			midpoint = 5.0
			penalty = gate_const*30.0/(1.+ np.exp(-steepness*(penalty-midpoint)))
			temp.append([np.dot(stats[2],td_err**2/add_vars), penalty])
			if s == 1 and a == 3:
				records['t'].append(self.step)
				records['k'].append(stats[2])
			records['mean'].append(copy.deepcopy(self.means))
			records['var'].append(copy.deepcopy(self.vars))
				#print("t:%d, var:%.4f, penalty:%.4f"%(self.step,new_var, penalty))

			self.means[s][a] = np.mean(new_mean)
			self.vars[s][a] = np.mean(new_var) + beta*penalty #np.maximum(self.varTH, new_var)

			if useScale:
				delta =  np.log(np.mean(self.vars[self.env.eff_states,:]))
				self.vars[self.env.eff_states,:] = np.exp(np.log(self.vars[self.env.eff_states,:]) - delta, dtype = np.float64)
				self.log_scale = np.maximum( -100.0, self.log_scale + delta)

			if draw:
				#self.var_plot()
				self.draw(s,a,self.step,r)

			if eval_greedy and ((self.step+1)%(int(self.env.timeH/util.EVAL_NUM)) == 0):
				count, rew , _, _= self.greedy_policy(lambda x : self.get_action_egreedy(x, util.EVAL_EPS))
				self.test_counts.append(count)
				self.test_rewards.append(rew)
			s = self.env.reset(self.np_random) if done else s_n
			self.step += 1
		return records, temp

	def var_plot(self):
		import matplotlib
		matplotlib.use('TkAgg')
		from matplotlib import pyplot as plt
		if self.env.name != 'movingmaze':
			return None
		y_mat = -15.0*np.ones(self.env.dim)
		for s in range(self.env.snum):
			y_mat[int(s/3), int(s%3)] = np.log(np.mean(self.vars[s]))
		plt.imshow(y_mat, cmap='gray', vmin = -15.0, vmax=5.0)
		plt.title(str(self.step))
		plt.show()
