""" 
<Policy Iteration>

Author: Heejin Chloe Jeong (chloe.hjeong@gmail.com)
Affiliation: University of Pennsylvania
"""
import numpy as np
import seeding

def policy_iter(env, discount, threshold, T=5000):

	V = np.zeros(env.snum)
	policy = np.random.choice(env.anum, env.snum)
	np_random, _  = seeding.np_random(None)
	p_stable = False
	trans_dict = {}
	rew_dict = {}
	slip_prob = env.slip
	if env.stochastic_reward:
		slip_prob_r = env.slip_r
	for state in env.eff_states:
		for action in range(env.anum):
			transM = np.zeros(env.snum)
			rewM = np.zeros(env.snum)

			if env.stochastic:
				env.slip = 0.0
				if env.stochastic_reward:
					env.slip_r = 0.0
					r0, s_n0, _ = env.observe(state,action,np_random)
					transM[s_n0] = 1.0-slip_prob
					rewM[s_n0] = (1.0-slip_prob_r)*r0

					env.slip_r = 1.0
					r1, s_n1, _ = env.observe(state,action,np_random)
					rewM[s_n1] += slip_prob_r*r1
					assert(s_n0 == s_n1)
				else:
					r0, s_n0, _ = env.observe(state,action,np_random)
					transM[s_n0] = 1.0-slip_prob
					rewM[s_n0] = r0

				env.slip = 1.0
				if env.stochastic_reward:
					env.slip_r = 0.0
					r0, s_n0, _ = env.observe(state,action,np_random)
					transM[s_n0] = 1.0-slip_prob
					rewM[s_n0] = (1.0-slip_prob_r)*r0

					env.slip_r = 1.0
					r1, s_n1, _ = env.observe(state,action,np_random)
					rewM[s_n1] += slip_prob_r*r1
				else:
					r1, s_n1, _ = env.observe(state,action,np_random)
					transM[s_n1] = slip_prob
					rewM[s_n1] = r1
			else:
				if env.stochastic_reward:
					env.slip_r = 0.0
					r0, s_n0, _ = env.observe(state,action,np_random)
					transM[s_n0] = 1.0
					rewM[s_n0] = (1.0-slip_prob_r)*r0

					env.slip_r = 1.0
					r1, s_n1, _ = env.observe(state,action,np_random)
					if s_n1 != s_n0:
						print("Transition is stochastic!")
					rewM[s_n1] += slip_prob_r*r1
				else:
					r0, s_n0, _ = env.observe(state,action,np_random)
					transM[s_n0] = 1.0
					rewM[s_n0] = r0

			trans_dict[(state,action)] = transM
			rew_dict[(state,action)] = rewM
	it = 0
	env.slip = slip_prob
	if env.stochastic_reward:
		env.slip_r = slip_prob_r
	while(not p_stable):
		delta = 1.0
		t = 0
		while(delta > threshold and t < T):
			delta = 0
			for s in env.eff_states:
				v_prev = V[s]
				V[s] = sum([ trans_dict[(s,policy[s])][s_next] * (rew_dict[(s,policy[s])][s_next] \
						+ int((s_next<env.goal[0]) or (s_next>=env.goal[1]))*discount*V[s_next]) \
						for s_next in range(env.snum)])
				delta = max(delta, abs(v_prev-V[s]))
			t += 1
		p_stable = True
		for s in env.eff_states:
			u_old = policy[s]
			q_val = [sum([ trans_dict[(s,u)][s_next] * (rew_dict[(s,u)][s_next] \
					+ int((s_next<env.goal[0]) or (s_next>=env.goal[1]))*discount*V[s_next]) \
					for s_next in range(env.snum)]) for u in range(env.anum)]

			if max(q_val) - min(q_val) < 0.001:
				policy[s] = 0
			else:
				policy[s] = np.argmax(q_val)
				if not(u_old == policy[s]):
					p_stable = False
		it+=1
	print("after %d iterations"%it)
	Q = np.zeros((env.snum,env.anum))
	for s in env.eff_states:
		for a in range(env.anum):
			Q[s][a] = sum([ trans_dict[(s,a)][s_next] * (rew_dict[(s,a)][s_next] \
						+ int((s_next<env.goal[0]) or (s_next>=env.goal[1]))*discount*V[s_next]) \
						for s_next in range(env.snum)])

	return V, Q, policy

def plot_V_pi(Vs, pis, env):
    for (V, pi) in zip(Vs, pis):
        plt.figure(figsize=(3, 3))
        w = int(np.sqrt(V.shape[0]))
        plt.imshow(V.reshape(w, w), cmap='gray',
                   interpolation='none', clim=(0, 1))
        ax = plt.gca()
        ax.set_xticks(np.arange(w) - .5)
        ax.set_yticks(np.arange(w) - .5)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        Y, X = np.mgrid[0:w, 0:w]
        a2uv = {0: (-1, 0), 1: (0, -1), 2: (1, 0), 3: (-1, 0)}
        Pi = pi.reshape(w, w)
        for y in range(w):
            for x in range(w):
                a = Pi[y, x]
                u, v = a2uv[a]
                plt.arrow(x, y, u * .3, -v * .3, color='m',
                          head_width=0.1, head_length=0.1)
                plt.text(x, y, str(env.desc[y, x].item().decode()),
                         color='g', size=12,  verticalalignment='center',
                         horizontalalignment='center', fontweight='bold')
        plt.grid(color='b', lw=2, ls='-')


