"""""""""""""""""""""""""""""""""
Example script for running ADFQ, KTD-Q, Tabular Q-learning.
The default values are not necessarily the best values.
Please see the ADFQ paper for details on the evaluation methods and hyperparameters.
You can choose different aciton policies for the actionPolicy variable:
	- offline # give an action trajectory
	- egreedy # epsilon-greedy
	- softmax # softmax with the boltzmann distribution only for Q-learning
	- Bayes # Bayesian (posterior) sampling only for ADFQ
	- semi-Bayes # random with a small given probability and Bayesian sampling otherwise, only for ADFQ
	- active # active learning scheme for KTD-Q
	- random # uniform-random
You can also test the algorithms in different domains presented in envs/classic_mdp.py
	- chain
	- loop
	- grid5
	- grid10
	- minimaze
	- maze
	- inv_pendulum # KTD-Q only

"""""""""""""""""""""""""""""""""
from adfq import *
import util
import envs
from ktdq import KTD_Q
from tabularRL import *

import numpy as np
import argparse
import os
from tabulate import tabulate
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--num_trials', type=int, default=10, help='the number of trials to repeat')
parser.add_argument('--discount', type=float, default=0.95, help= 'discount factor')
parser.add_argument('--log_dir', type=str, default='./')
parser.add_argument('--env', type=str, default='loop', help='domain name')
parser.add_argument('--slip', type=float, default=0.0, help='stochasticity of the environment')
parser.add_argument('--q_learning_rate', type=float, default=0.5, help ='initial learning rate for Q-learning')
parser.add_argument('--eps', type=float, default=0.1, help='epsilon of epsilon greedy')
parser.add_argument('--init_mean', type=float, default=0.001)
args = parser.parse_args()
print(tabulate([(k,v) for (k,v) in vars(args).items()]))

f1, ax1 = plt.subplots()
ax1.set_title('Performance')
ax1.set_xlabel('Learning Steps')
ax1.set_ylabel('Average Episode Rewards')

f2, ax2 = plt.subplots()
ax2.set_title('Convergence') # RMSE error is always larger than 0.0 for 'minimaze' and 'maze' domain since there are some states that never get visited.
ax2.set_xlabel('Learning Steps')
ax2.set_ylabel('Average Episode Rewards')

# Random Action trajectory
env = envs.make(args.env, 'classic_mdp')
actions = np.random.choice(env.anum, env.timeH)

x_vals = range(0, env.timeH, int(env.timeH/util.EVAL_NUM))
# ADFQ
print("Running ADFQ ... ")
noise = 0.0 if args.slip == 0.0 else 0.001
batch_size = 0 if args.slip == 0.0 else 20
adfq = ADFQ(args.env, args.discount, init_mean = args.init_mean)
adfq.env.set_slip(args.slip)
adfq.learning(actionPolicy='offline', actionParam=actions, eval_greedy = True, noise=noise, batch_size=batch_size)
ax1.plot(x_vals, adfq.test_rewards)
ax2.plot(x_vals, adfq.Q_err)

# Q-learning
print("Running Q-learning ... ")
qlearning = Qlearning(args.env, args.q_learning_rate, args.discount, initQ = args.init_mean)
qlearning.env.set_slip(args.slip)
qlearning.learning(actionPolicy='offline', actionParam=actions, eval_greedy=True)
ax1.plot(x_vals, qlearning.test_rewards)
ax2.plot(x_vals, qlearning.Q_err)

# KTD-Q
print("Running KTD-Q ... ")
ktd = KTD_Q(args.env, args.discount, init_mean = args.init_mean)
ktd.env.set_slip(args.slip)
ktd.learning(actionPolicy='offline', actionParam=actions, kappa=1, eval_greedy = True)
ax1.plot(x_vals, ktd.test_rewards)
ax2.plot(x_vals, ktd.Q_err)

ax1.legend(['ADFQ', 'Q-learning', 'KTD-Q'])
ax2.legend(['ADFQ', 'Q-learning', 'KTD-Q'])

plt.show()
