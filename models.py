""""""""""""""""""""""""""""""""" 
Some MDP environments

Author: Heejin Chloe Jeong (chloe.hjeong@gmail.com)
Affiliation: University of Pennsylvania

Available domains:
	Chain (from Dearden et al., Bayesian Q-learning)
	Loop (from Dearden et al., Bayesian Q-learning)
	MiniMaze 
	Maze (from Dearden et al., Bayesian Q-learning)
	Grid5 
	Grid10 
	Inverted Pendulum (from Geist et al., Kalman Temporal Differences)
	Tsitsiklis 
"""""""""""""""""""""""""""""""""
import numpy as np
import random 
import brl_util as util
import pdb
import math
import policy_iter as pi

ACTMAP = {0:3, 1:2, 2:0, 3:1} # 0:UP, 1:DOWN, 2:LEFT, 3:RIGHT

def model_assign(scene):
	if scene == 'chain': 
		return Chain()
	elif scene == 'loop' : 
		return Loop()
	elif scene == 'maze' : 
		return Maze()
	elif scene == 'grid5' :
		return Grid5()
	elif scene == 'grid10' :
		return Grid10()
	elif scene == 'inv_pendulum':
		return Inv_Pendulum()
	elif scene == 'minimaze':
		return MiniMaze()
	elif scene == 'Tsitsiklis':
		return Tsitsiklis()
	else:
		raise ModelError('unknown scene')

class Models(object):
	def __init__(self):
		pass

	def set_time(self, time):
		self.timeH = time

	def set_slip(self, p):
		"""setting a stochasticity of an environment. With a given "slip" probability, 
		a leanring agent performs a different action from the one it chose.
		"""
		self.slip = p

class Chain(Models):
	# state number : 0,1,2,3,4
	# action number : 0,1
	def __init__(self):
		Models.__init__(self)

		self.name = 'chain'
		self.episodic = False
		self.snum = 5
		self.anum = 2
		self.timeH = util.T_chain
		self.slip = 0.1
		self.stochastic = True
		self.map = np.asarray(["01234"],dtype='c')
		self.goal = (5,5) # This is not a goal. default.
		self.eff_states = [i for i in range(self.snum) if not(i in [])]
		self.phi = (lambda x,y: util.discrete_phi(x, y, self.snum*self.anum, self.anum), self.snum*self.anum)

	def observe(self,state,action,np_rand):
		
		if np_rand.rand(1)[0] > (1-self.slip):
			a = -action+1;
		else:
			a = action

		if a == 1 :
			return 2.0, 0, False
		else:
			if state == 4:
				return 10.0, 4, False
			else:
				return 0.0, state+1, False

	def plot(self,state,action):
		desc = self.map.tolist()
		desc[0][state] = util.colorize(desc[0][state], "red", highlight=True)        
		print("action: ", ["1","2"][action]) if action is not None else None
		print("\n".join("".join(row) for row in desc))

	def reset(self,np_rand):
		return 0 

	def optQ(self,discount):
		try:
			Q = np.load("optQ/chain_"+str(int(discount*100))+"_"+str(int(self.slip*10))+".npy")
		except:		
			print("No Stored Data for optQ")
			_,Q,_ = pi.policy_iter(self, discount, 0.001)
			np.save("optQ/chain_"+str(int(discount*100))+"_"+str(int(self.slip*10))+".npy", Q)
		return Q

		# Q = np.zeros((self.snum,self.anum))
		# V = np.zeros((self.snum,1))
		# a = discount*(1-self.slip)
		# b = discount*self.slip
		# if self.slip == 0 :
		# 	const = 10*(discount**4)/(1-discount)
		# 	coeff = 0
		# else:
		# 	const = (1+a+a**2+a**3+(a**4)*(5/self.slip-4)/(1-a))*2*self.slip/(1-b)
		# 	coeff = a*b/(1-b)*(1+a+a**2+(a**3)/(1-a))

		# V[0][0] = const/(1-coeff)
		# V[4][0] = (10-8*self.slip + b*V[0][0])/(1-a)
		# Q[0][0] = V[0][0]
		# Q[4][0] = V[4][0]
		# Q[4][1] = 2*(1-self.slip)+a*V[0][0]+b*V[4][0] + 10*self.slip

		# for i in range(3,0,-1):
		# 	V[i][0] = a*V[i+1][0] + 2*self.slip + b*V[0][0]
		# 	Q[i][0] = V[i][0]
		# 	Q[i][1] = 2*(1-self.slip)+a*V[0][0]+b*V[i+1][0]
		# Q[0][1] = 2*(1-self.slip) + a*V[0][0] + b*V[1][0]
		# return Q

class Loop(Models):
	# state number : 0,1,...,8
	# action number : 0,1
	def __init__(self):
		Models.__init__(self)
		self.name = 'loop'
		self.episodic = False
		self.snum = 9
		self.anum = 2
		self.timeH = util.T_loop
		self.slip = 0.0
		self.stochastic = False
		self.goal = (9,9) # This is not a goal. default.
		self.eff_states = [i for i in range(self.snum) if not(i in [])]
		self.phi = (lambda x,y: util.discrete_phi(x, y, self.snum*self.anum, self.anum), self.snum*self.anum)

	def observe(self, state,action,np_rand=None):
		# state = 0,...,8
		if state == 4:
			return 1.0, 0, False
		elif state == 8:
			return 2.0, 0, False
		else:
			if np_rand.rand(1)[0] > (1-self.slip):
				a = -action+1;
			else:
				a = action

			if state == 0:
				if a == 0:
					return 0.0, 1, False
				else:
					return 0.0, 5, False
			elif (state>0) and (state<4):
				return 0.0, state+1, False
			else:
				return 0.0, a*(state+1), False
	def optQ(self,discount):
		try:
			Q = np.load("optQ/loop_"+str(int(discount*100))+"_"+str(int(self.slip*10))+".npy")
		except:		
			print("No Stored Data for optQ")
			_,Q,_ = pi.policy_iter(self, discount, 0.001)
			np.save("optQ/loop_"+str(int(discount*100))+"_"+str(int(self.slip*10))+".npy", Q)
		return Q
		# Q = np.zeros((self.snum, self.anum),dtype=float)
		# Q[0][1] = 2*(discount**4)/(1-discount**5)
		# Q[0][0] = (discount**4)*(1+discount*Q[0][1])
		# Q[4][0] = 1+discount*Q[0][1]
		# Q[4][1] = Q[4][0]
		# Q[8][0] = 1+Q[4][0]
		# Q[8][1] = Q[8][0]
		# for i in [3,2,1]:
		# 	Q[i][0] = discount*Q[i+1][0]
		# 	Q[i][1] = Q[i][0]
		# for i in [7,6,5]:
		# 	Q[i][1] = discount*Q[i+1][1]
		# 	Q[i][0] = discount*Q[0][1]
		
		# return np.array(Q)
	def reset(self,np_rand):
		return 0

class MiniMaze(Models):
	# state id : 0, ..., 111
	# action id : 0,1,2,3 (Up, Down, Left, Right)
	obstacles = [(0,1),(0,3),(2,0),(2,4),(3,2),(3,4)]
	def __init__(self):
		Models.__init__(self)
		self.name = 'minimaze'
		self.episodic = True
		self.stochastic = True
		self.snum = 112
		self.anum = 4
		self.timeH = util.T_minimaze
		self.slip = 0.1
		self.dim = (4,5)
		self.start_pos = (0,0)
		self.goal_pos = (0,4)
		self.goal = (96,104)
		redundants = [16, 17, 19, 21, 48, 50, 51, 54, 88, 89, 90, 92, 96, 97, 98, 99, 100, 101, 102, 103]
		self.eff_states = [i for i in range(self.snum) if not(i in redundants)]
		self.phi = (lambda x,y: util.discrete_phi(x, y, self.snum*self.anum, self.anum), self.snum*self.anum)
		self.map = np.asarray(["SWFWG","OOOOO","WOOOW","FOWFW"],dtype='c')
		self.img_map = np.ones(self.dim)
		for x in MiniMaze.obstacles:
			self.img_map[x[0]][x[1]] = 0
		self.idx2cell = {0: (0, 0), 1: (1, 0), 2: (3, 0), 3: (1, 1), 4: (2, 1), 5: (3, 1),
			6: (0, 2), 7: (1, 2), 8: (2, 2), 9: (1, 3), 10: (2, 3), 11: (3, 3), 12: (0, 4), 13: (1, 4)}
		self.cell2idx = {(1, 2): 7, (0, 0): 0, (3, 3): 11, (3, 0): 2, (3, 1): 5, (2, 1): 4, 
			(0, 2): 6, (1, 3): 9, (2, 3): 10, (1, 4): 13, (2, 2): 8, (0, 4): 12, (1, 0): 1, (1, 1): 3}
	
	def observe(self,state,action,np_rand):

		if np_rand.rand(1)[0] < self.slip:
			a = ACTMAP[action]
		else:
			a = action
		
		cell = self.idx2cell[int(state/8)]
		if a == 0:
			c_next = cell[1]
			r_next = max(0,cell[0]-1)
		elif a ==1:
			c_next = cell[1]
			r_next = min(self.dim[0]-1,cell[0]+1)
		elif a == 2:
			c_next = max(0,cell[1]-1)
			r_next = cell[0]
		elif a == 3:
			c_next = min(self.dim[1]-1,cell[1]+1)
			r_next = cell[0]
		else:
			print(action, a) 
			raise ValueError
		
		if (r_next == self.goal_pos[0]) and (c_next == self.goal_pos[1]): # Reach the exit
			v_flag = self.num2flag(state%8)
			return float(sum(v_flag)), 8*self.cell2idx[(r_next,c_next)] + state%8, True
		else:
			if (r_next,c_next) in MiniMaze.obstacles: # obstacle tuple list
				return 0.0, state, False
			else: # Flag locations
				v_flag = self.num2flag(state%8)
				if (r_next,c_next) == (0,2):
					v_flag[0] = 1
				elif (r_next,c_next)==(3,0):
					v_flag[1] = 1
				elif (r_next,c_next) == (3,3):
					v_flag[2] = 1
				return 0.0, 8*self.cell2idx[(r_next,c_next)] + self.flag2num(v_flag), False


	def num2flag(self,n):
		# n is a positive integer
		flaglist = [(0,0,0),(1,0,0),(0,1,0),(0,0,1),(1,1,0),(1,0,1),(0,1,1),(1,1,1)]
		return list(flaglist[n])

	def flag2num(self,v):
		# v: list
		if sum(v) < 2:
			return np.inner(v,[1,2,3])
		else:
			return np.inner(v,[1,2,3])+1

	def plot(self,state,action):
		cell = self.idx2cell[int(state/8)]
		desc = self.map.tolist()
		desc[cell[0]][cell[1]] = util.colorize(desc[cell[0]][cell[1]], "red", highlight=True)        
		print("action: ", ["UP","DOWN","LEFT","RIGHT"][action]) if action is not None else None
		print("\n".join("".join(row) for row in desc))

	def optQ(self, discount):
		try:
			Q = np.load("optQ/minimaze_"+str(int(discount*100))+"_"+str(int(self.slip*10))+".npy")
		except:		
			print("No Stored Data for optQ")
			_,Q,_ = pi.policy_iter(self, discount, 0.001)
			np.save("optQ/minimaze_"+str(int(discount*100))+"_"+str(int(self.slip*10))+".npy", Q)
		return Q

	def reset(self,np_rand):
		return 0

class Maze(Models):
	# state id : 0, ..., 263
	# action id : 0,1,2,3 (Up, Down, Left, Right)
	obstacles = [(0,1),(1,1),(0,4),(1,4),(3,0),(3,1),(3,5),(3,6),(5,6)]
	def __init__(self):
		Models.__init__(self)
		self.name = 'maze'
		self.episodic = True
		self.stochastic = True
		self.snum = 264
		self.anum = 4
		self.timeH = util.T_maze
		self.slip = 0.0
		self.dim = (6,7)
		self.start_pos = (0,0)
		self.goal_pos = (0,6)
		self.goal = (232,240)
		redundants = [32, 33, 35, 37, 64, 66, 67, 70, 232, 233, 234, 235, 236, 237, 238, 239, 256, 257, 258, 260]
		self.eff_states = [i for i in range(self.snum) if not(i in redundants)]
		self.phi = (lambda x,y: util.discrete_phi(x, y, self.snum*self.anum, self.anum), self.snum*self.anum)
		self.map = np.asarray(["SWFOWOG","OWOOWOO","OOOOOOO","WWOOOWW","OOOOOOF","FOOOOOW"],dtype='c')
		self.img_map = np.ones(self.dim)
		for x in Maze.obstacles:
			self.img_map[x[0]][x[1]] = 0
		self.idx2cell = {0: (0, 0), 1: (1, 0), 2: (2, 0), 3: (4, 0), 4: (5, 0), 5: (2, 1), 6: (4, 1), 
			7: (5, 1), 8: (0, 2), 9: (1, 2), 10: (2, 2), 11: (3, 2), 12: (4, 2), 13: (5, 2), 
			14: (0, 3), 15: (1, 3), 16: (2, 3), 17: (3, 3), 18: (4, 3), 19: (5, 3), 20: (2, 4), 
			21: (3, 4), 22: (4, 4), 23: (5, 4), 24: (0, 5), 25: (1, 5), 26: (2, 5), 27: (4, 5), 
			28: (5, 5), 29: (0, 6), 30: (1, 6), 31: (2, 6), 32: (4, 6)}
		self.cell2idx = {(1, 3): 15, (5, 4): 23, (2, 1): 5, (2, 6): 31, (1, 6): 30, (5, 1): 7, (0, 3): 14, 
			(2, 5): 26, (4, 0): 3, (1, 2): 9, (3, 3): 17, (0, 6): 29, (4, 4): 22, (1, 5): 25, (5, 0): 4, 
			(2, 2): 10, (4, 1): 6, (3, 2): 11, (0, 0): 0, (4, 5): 27, (5, 5): 28, (2, 3): 16, (4, 2): 12, 
			(1, 0): 1, (5, 3): 19, (4, 6): 32, (3, 4): 21, (0, 2): 8, (2, 0): 2, (4, 3): 18, (0, 5): 24, 
			(5, 2): 13, (2, 4): 20}

	def observe(self,state,action,np_rand):

		if np_rand.rand(1)[0] < self.slip:
			a = ACTMAP[action]
		else:
			a = action
		
		cell = self.idx2cell[int(state/8)]
	
		if a == 0:
			c_next = cell[1]
			r_next = max(0,cell[0]-1)
		elif a ==1:
			c_next = cell[1]
			r_next = min(self.dim[0]-1,cell[0]+1)
		elif a == 2:
			c_next = max(0,cell[1]-1)
			r_next = cell[0]
		elif a == 3:
			c_next = min(self.dim[1]-1,cell[1]+1)
			r_next = cell[0]
		else:
			print(action, a)
			raise ValueError		
		if (r_next == self.goal_pos[0]) and (c_next == self.goal_pos[1]): 
			v_flag = self.num2flag(state%8)
			return float(sum(v_flag)), 8*self.cell2idx[(r_next,c_next)] + state%8, True #self.flag2num(v_flag))
		else:
			if (r_next,c_next) in Maze.obstacles: # obstacle tuple list
				return 0.0, state, False
			else: # Flag locations
				v_flag = self.num2flag(state%8)
				if (r_next,c_next) == (0,2):
					v_flag[0] = 1
				elif (r_next,c_next)==(self.dim[0]-1,0):
					v_flag[1] = 1
				elif (r_next,c_next) == (self.dim[0]-2,self.dim[1]-1):
					v_flag[2] = 1
				return 0.0, 8*self.cell2idx[(r_next,c_next)] + self.flag2num(v_flag), False

	def num2flag(self,n):
		# n is a positive integer
		flaglist = [(0,0,0),(1,0,0),(0,1,0),(0,0,1),(1,1,0),(1,0,1),(0,1,1),(1,1,1)]
		return list(flaglist[n])

	def flag2num(self,v):
		# v: list
		if sum(v) < 2:
			return np.inner(v,[1,2,3])
		else:
			return np.inner(v,[1,2,3])+1

	def plot(self,state,action):
		cell = self.idx2cell[int(state/8)]
		desc = self.map.tolist()
		desc[cell[0]][cell[1]] = util.colorize(desc[cell[0]][cell[1]], "red", highlight=True)        
		print("action: ", ["UP","DOWN","LEFT","RIGHT"][action]) if action is not None else None
		print("\n".join("".join(row) for row in desc))

	def optQ(self, discount):
		try:
			Q = np.load("optQ/maze_"+str(int(discount*100))+"_"+str(int(self.slip*10))+".npy")
		except:		
			print("No Stored Data for optQ")
			_,Q,_ = pi.policy_iter(self, discount, 0.001)
			np.save("optQ/maze_"+str(int(discount*100))+"_"+str(int(self.slip*10))+".npy", Q)
		return Q

	def reset(self,np_rand):
		return 0

class Grid5(Models):
	def __init__(self, optQ_given=None):
		Models.__init__(self)
		self.name = 'grid5'
		self.episodic = True
		self.stochastic = True
		self.snum = 25 
		self.anum = 4
		self.timeH = util.T_grid5
		self.slip = 0.1
		self.dim = (5,5)
		self.start_pos = (0,0)
		self.goal_pos = (4,4)
		self.goal = (24,25)
		self.eff_states = range(self.snum-1)
		self.optQ_given = optQ_given
		self.map = np.asarray(["SOOOO","OOOOO","OOOOO","OOOOO","OOOOG"],dtype='c')
		self.img_map = np.ones((5,5))
		self.phi = (lambda x,y: util.discrete_phi(x, y, self.snum*self.anum, self.anum), self.snum*self.anum)

	def observe(self,state,action,np_rand):
		rand_num = np.random.rand(1)
		if rand_num >(1-self.slip):
			a = ACTMAP[action]
		else:
			a = action

		r = state%5
		c = int(state/5)
		# action :: 0:up 1:down 2:left 3:right
		(r,c) = self.move(a,(r,c))
		if r==self.goal_pos[0] and c==self.goal_pos[1]:
			return 1.0, self.dim[0]*c+r, True
		else:
			return 0.0, self.dim[0]*c+r, False

	def plot(self,state,action):
		r = state%5
		c = int(state/5)
		desc = self.map.tolist()
		desc[r][c] = util.colorize(desc[r][c], "red", highlight=True) 
		print("action: ", ["UP","DOWN","LEFT","RIGHT"][action]) if action is not None else None       
		print("\n".join("".join(row) for row in desc))

	def move(self,action,pos): 
		"""
		action: action number - 0:up 1:down 2:left 3:right
		pos: a tuple of (row,colunm)
		"""
		(r,c) = pos 
		if action == 0:
			r = max(0,pos[0]-1)
		elif action == 1:
			r = min(4,pos[0]+1)
		elif action ==2:
			c = max(0,pos[1]-1)
		elif action == 3:
			c = min(4,pos[1]+1)
		else:
			raise ValueError
		return (r,c)

	def optQ(self, discount):
		try:
			Q = np.load("optQ/grid5_"+str(int(discount*100))+"_"+str(int(self.slip*10))+".npy")
		except:		
			print("No Stored Data for optQ")
			_,Q,_ = pi.policy_iter(self, discount, 0.001)
			np.save("optQ/grid5_"+str(int(discount*100))+"_"+str(int(self.slip*10))+".npy", Q)
		return Q

	def reset(self,np_rand):
		return 0

class Grid10(Models):
	def __init__(self):
		Models.__init__(self)
		self.name = 'grid10'
		self.episodic = True
		self.stochastic = True
		self.snum = 100 
		self.anum = 4
		self.timeH = util.T_grid10
		self.slip = 0.1
		self.dim = (10,10)
		self.start_pos = (0,0)
		self.goal_pos = (9,9)
		self.goal = (99,100)
		self.eff_states = range(self.snum-1)
		self.map = np.asarray(["SOOOOOOOOO","OOOOOOOOOO","OOOOOOOOOO","OOOOOOOOOO","OOOOOOOOOO","OOOOOOOOOO","OOOOOOOOOO","OOOOOOOOOO","OOOOOOOOOO","OOOOOOOOOG"],dtype='c')
		self.phi = (lambda x,y: util.discrete_phi(x, y, self.snum*self.anum, self.anum), self.snum*self.anum)

	def observe(self,state,action,np_rand):
		rand_num = np.random.rand(1)
		if rand_num >(1-self.slip):
			a = ACTMAP[action]
		else:
			a = action

		r = state%10
		c = int(state/10)
		# action :: 0:up 1:down 2:left 3:right
		if a == 0:
			r = max(0,r-1)
		elif a == 1:
			r = min(9,r+1)
		elif a ==2:
			c = max(0,c-1)
		elif a == 3:
			c = min(9,c+1)
		else:
			raise ValueError

		if r==self.goal_pos[0] and c==self.goal_pos[1]:
			return 1.0, self.dim[0]*c+r, True
		else:
			return 0.0, self.dim[0]*c+r, False

	def plot(self,state,action):
		r = state%10
		c = int(state/10)
		desc = self.map.tolist()
		desc[r][c] = util.colorize(desc[r][c], "red", highlight=True)  
		print("action: ", ["UP","DOWN","LEFT","RIGHT"][action]) if action is not None else None      
		print("\n".join("".join(row) for row in desc))

	def optQ(self, discount):
		try:
			Q = np.load("optQ/grid10_"+str(int(discount*100))+"_"+str(int(self.slip*10))+".npy")
		except:		
			print("No Stored Data for optQ")
			_,Q,_ = pi.policy_iter(self, discount, 0.001)
			np.save("optQ/grid10_"+str(int(discount*100))+"_"+str(int(self.slip*10))+".npy", Q)
		return Q

	def reset(self,np_rand):
		return 0

class Inv_Pendulum(Models):
	def __init__(self):
		Models.__init__(self)
		self.name = 'inv_pendulum'
		self.snum = 2 # state dimension
		self.anum = 3 #-1, 0, +1
		self.timeH = 1000 # number of episodes
		self.step_bound = 3000
		self.force_mag = 50.0
		self.phi = (lambda x,y: util.rbf(x,y,10), 30)
		self.episodic = True

	def observe(self,state,action,np_rand=None):
		# mass and length (model parameters)
		m,M,l =2.0, 8.0, 0.5
		beta = 1/(m+M) 
		del_t = 0.1  # sec
		sintheta = math.sin(state[0])
		costheta = math.cos(state[0])
		sin2theta = math.sin(2*state[0])
		force = self.force_mag * (action -1)
		ang_acc = 9.8 * sintheta - beta*(m*l*state[1]*state[1]*sin2theta*0.5+costheta*force)
		ang_acc = ang_acc/(4*l/3.0-beta*m*l*(costheta**2))
		#next_state = [state[0]+ state[1]*del_t + 0.5*ang_acc*del_t*del_t,state[1] + ang_acc*del_t ]
		next_state = [state[0] + del_t*state[1], state[1]+del_t*ang_acc]
		if abs(next_state[0]) < (0.5*np.pi):
			return 0.0, next_state, False
		else:
			return -1.0, next_state, True
	def optQ(self, discount):
		# Null
		return 0.0 

	def reset(self, np_rand):
		return np_rand.uniform(low=-0.05, high=0.05, size=(2,))

class Tsitsiklis(Models):
	def __init__(self):
		Models.__init__(self)
		self.name = 'Tsitsiklis'
		self.action = [0,1]
		self.anum = len(self.action)
		self.timeH = 1000
		self.snum = 3
		self.episodic = False
		self.slip = 0.0
		self.stochastic = False

	def observe(self,state,action,np_rand):
		rand_num = np.random.rand(1)
		if rand_num > 0.5:
			return 0.0, state, False
		else:
			if state == 0:
				return 0.0, self.snum-1, False
			else:
				return 0.0, state-1, False

	def optQ(self, discount):
		_,Q,_ = pi.policy_iter(self, discount, 0.001)
		return Q
		
	def reset(self,np_rand):
		return 0
