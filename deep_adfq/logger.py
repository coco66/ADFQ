"""
Customized functions for logging training data.
"""
import pickle, os, time
import numpy as np
from tabulate import tabulate
import envs

class Logger():
    def __init__(self, env_id, env_type, variables=None, save_dir=".", **kwargs):
        self.epoch_num = 0
        self.ep_history = {'loss':[], 'q':[], 'q_err':[], 'ep_rewards':[0.0]}
        self.step_history = {'loss':[], 'q':[], 'q_err':[]}
        self.records = {'online_reward':[], 'test_reward':[], 'q':[], 'q_err':[],
            'loss':[], 'learning_rate':[], 'time_spent':[]}
        if variables:
            for v in variables:
                self.ep_history[v] = []
                self.step_history[v] = []
                self.records[v] = []

        self.save_file = os.path.join(save_dir, "records.pkl")
        self.save_dir = save_dir
        self.s_time = time.time()
        self.env_type = env_type
        self.eval_f = lambda x : evaluation(x, env_id, env_type, **kwargs)
        if env_type == 'target_tracking':
            self.records['mean_nlogdetcov'] = []

    def log_reward(self, reward):
        self.ep_history['ep_rewards'][-1] += reward

    def log_step(self, loss, q, q_err, **kwargs):
        self.step_history['loss'].append(loss)
        self.step_history['q'].append(q)
        self.step_history['q_err'].append(q_err)
        for (k,v) in kwargs.items():
            if not(k in self.step_history):
                raise KeyError("No key exists - %s"%k)
            self.step_history[k].append(v)

    def log_ep(self, info=None):
        if ((self.env_type=='atari') and (info['ale.lives'] == 0)) or not(self.env_type=='atari'):
            for (k,v) in self.ep_history.items():
                if k == 'ep_rewards':
                    self.ep_history[k].append(0.0)
                else:
                    if self.step_history[k]:
                        v.append(np.mean(self.step_history[k]))
                        self.step_history[k] = []

    def log_epoch(self, act, lr, **kwargs):
        test_reward, mean_nlogdetcov = self.eval_f(act)
        for (k,v) in self.records.items():
            if k == 'online_reward':
                if len(self.ep_history['ep_rewards']) > 1:
                    v.append(round(np.mean(self.ep_history['ep_rewards'][-101:-1]),1))
            elif k == 'test_reward':
                v.append(test_reward)
            elif k == 'mean_nlogdetcov':
                v.append(mean_nlogdetcov)
            elif k == 'learning_rate':
                v.append(lr)
            elif k == 'time_spent':
                v.append(time.time() - self.s_time)
                self.s_time = time.time()
            else:
                if self.ep_history[k]:
                    v.append(np.mean(self.ep_history[k]))

        pickle.dump(self.records, open(self.save_file, "wb"))
        print("============ EPOCH %d ============"%(self.epoch_num))
        print(tabulate([[k,v[-1]] for (k,v) in self.records.items() if v], floatfmt=".4f"))
        self.epoch_num += 1

    def get_100ep_reward(self):
        return round(np.mean(self.ep_history['ep_rewards'][-101:-1]), 1)

    def get_num_episode(self):
        return len(self.ep_history['ep_rewards'])

    def finish(self, nb_train_steps, nb_epoch_steps, nb_warmup_steps=0):
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt
        from util import mstd

        x_vals = range(0, nb_train_steps+1, nb_epoch_steps)

        # Results to plot except online reward and test reward.
        N = len(self.records) - 3 - int('mean_nlogdetcov' in self.records)
        nrows = 3 if N == 6 else 2
        f, ax = plt.subplots(nrows=nrows, ncols=int((N-1)/nrows)+1,
                                sharex=True, sharey=False)
        i = 0
        for (k,v) in self.records.items():
            if len(v) > 0:
                if k == 'test_reward' or k == 'mean_nlogdetcov' or k == 'online_reward':
                    f0, ax0 = plt.subplots()
                    if len(np.array(v).shape) == 2:
                        m, ids25, ids75 = mstd(np.array(v).T)
                        # If the number of warmup steps > the number of epoch
                        # step or there is no episode completed, v may not have
                        # the same length with the x_vals.
                        _ = ax0.plot(x_vals[-len(v):], m, color='b')
                        _ = ax0.fill_between(x_vals[-len(v):], list(ids75),
                                            list(ids25), facecolor='b', alpha=0.2)
                    else:
                        _ = ax0.plot(x_vals[-len(v):], v, color='b')
                    _ = ax0.grid()
                    _ = ax0.set_ylabel(k)
                    _ = ax0.set_xlabel('Learning Steps')
                    if x_vals[-len(v)] < nb_warmup_steps:
                        _ = ax0.axvline(x=nb_warmup_steps, color='k')
                    _ = f0.savefig(os.path.join(self.save_dir, "%s.png"%k))
                else:
                    if k != 'time_spent':
                        row = i%nrows
                        col = int(i/nrows)
                        _ = ax[row][col].plot(x_vals[-len(v):], np.array(v, dtype=np.float16))
                        _ = ax[row][col].set_ylabel(k)
                        _ = ax[row][col].grid()
                        if col == int((N-1)/nrows):
                            _ = ax[row][col].yaxis.tick_right()
                        if row == nrows-1:
                            _ = ax[row][col].set_xlabel('Learning Steps')
                        i += 1
        _ = f.savefig(os.path.join(self.save_dir, "result.png"))

def evaluation(act, env_id, env_type, nb_test_steps=None, nb_itrs=5, render=False, **kwargs):
    """Evaluate the current model with a semi-greedy action policy.
    Parameters
    -------
    act: ActWrapper
        Wrapper over act function. Action policy for the evaluation.
    env_id: str
        name of an environment. (e.g. CartPole-v0)
    env_type: str
        type of an environment. (e.g. 'atari', 'classic_control', 'target_tracking')
    nb_test_steps: int
        the number of steps for the evaluation at each iteration. If None, it
        evaluates until an episode ends.
    nb_itrs: int
        the number of test iterations.
    render: bool
        display if True.

    Returns
    -------
    total_rewards: np.array with shape=(nb_itrs,)
        cumulative rewards.
    total_nlogdetcov : np.array with shape=(nb_itrs,)
        cumulative negative mean of logdetcov only for a target tracking env.
    """

    total_rewards = []
    if env_type == 'target_tracking':
        total_nlogdetcov = []
    env = envs.make(env_id, env_type, render=render, is_training=False, **kwargs)
    for _ in range(nb_itrs):
        obs = env.reset()
        if nb_test_steps is None: # Evaluate until an episode ends.
            done = False
            episode_reward = 0
            if env_type == 'target_tracking':
                episode_nlogdetcov = 0 # For target tracking env only.
            t = 0
            while not done:
                if render:
                    env.render()
                action = act(np.array(obs)[None])[0]
                obs, rew, done, info = env.step(action)
                episode_reward += rew
                if env_type == 'target_tracking':
                    episode_nlogdetcov += info['mean_nlogdetcov']
                t += 1
                if done and (env_type=='atari') and (info['ale.lives'] != 0):
                    done = False
            total_rewards.append(episode_reward)
            if env_type == 'target_tracking':
                total_nlogdetcov.append(episode_nlogdetcov)
        else:
            t = 0
            episodes = []
            episode_reward = 0
            while(t < nb_test_steps):
                if render:
                    env.render()
                action = act(np.array(obs)[None])[0]
                obs, rew, done, info = env.step(action)
                episode_reward += rew
                t += 1
                if done:
                    obs = env.reset()
                    if ((env_type=='atari') and (info['ale.lives'] == 0)) or not(env_type=='atari'):
                        episodes.append(episode_reward)
                        episode_reward = 0
            if not(episodes):
                episodes.append(episode_reward)
            total_rewards.append(np.mean(episodes))

    if render:
        env.close()
    if env_type == 'target_tracking':
        return np.array(total_rewards, dtype=np.float32), np.array(total_nlogdetcov, dtype=np.float32)
    else:
        return np.array(total_rewards, dtype=np.float32), None
