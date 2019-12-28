from adfq import ADFQ
from ktdq import KTD_Q
import numpy as np
import tabularRL

env_name = 'minimaze'
TH = 15000
action_random = np.random.choice(4,TH)
print("Testing the algorithm in the Mini-Maze environment...")

x = ADFQ(env_name,0.95, init_mean = 0.001, TH=TH)
x.env.set_slip(0.0)
x.learning('offline', action_random, eval_greedy=True)
avg_rew = np.mean(x.test_rewards[-10:])
if avg_rew > 2.9:
	print("ADFQ Deterministic: Reached an optimal policy... Passed the test!")
else:
	print(avg_rew)
	raise ValueError("It was unable to reach an optimal policy")

y = ADFQ(env_name, 0.95, init_mean = 0.001, TH=TH)
y.learning('offline', action_random, noise=0.001, eval_greedy=True)
avg_rew = np.mean(y.test_rewards[-10:])
if avg_rew > 2.0:
	print("ADFQ Stochastic: Reached an optimal policy... Passed the test!")
else:
	print(avg_rew)
	raise ValueError("It was unable to reach an optimal policy")

q = tabularRL.Qlearning(env_name,0.5,0.95, initQ=0.001, TH=TH)
q.env.set_slip(0.0)
q.learning('offline', action_random, eval_greedy=True)
print("Q-learning, %.2f"%(np.mean(q.test_rewards[-10:])))

ktd = KTD_Q('loop',0.95, TH=10000)
ktd.learning('egreedy', 0.1, kappa=1, eval_greedy=True)
avg_rew = np.mean(ktd.test_rewards[-10:])
if avg_rew >= 70.0:
	print("KTD-Q eps greedy: Reached a near-optimal policy... Passed the test!")
else:
	print(avg_rew)
	raise ValueError("KTD-Q eps was unable to reach an optimal policy")

ktd = KTD_Q('loop',0.95, TH=10000)
ktd.learning('active', 0.1, kappa=1, eval_greedy=True)
avg_rew = np.mean(ktd.test_rewards[-10:])
if avg_rew >= 70.0:
	print("KTD-Q active learning: Reached a near-optimal policy... Passed the test!")
else:
	print(avg_rew)
	raise ValueError("It was unable to reach an optimal policy")
