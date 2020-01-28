"""
Customized functions for logging training data.
"""
import pickle, os, time
import numpy as np
from tabulate import tabulate
import envs

class Logger():
    def __init__(self, env_id, env_type, variables=None, save_dir=".",
                                                eval_type='random', **kwargs):
        """
        Parameters
        ---------
        env_id : environment name to make a new environment.
        env_type : one of ['atari', 'classic_control', 'classic_mdp','target_tracking']
        variables : additional variables to log.
        save_dir : a path to a directory to save logging results.
        eval_type : only matters for ttenv environments. One of ['random', 'random_zone', 'fixed']
        """
        self.epoch_num = 0
        self.step_history = {'loss':[]}
        self.ep_history = {'ep_rewards':[0.0], 'loss':[]}
        self.records = {'online_reward':[], 'test_reward':[], 'time_spent':[]}
        if variables:
            for v in variables:
                self.step_history[v] = []
                self.ep_history[v] = []
                self.records[v] = []

        self.save_file = os.path.join(save_dir, "records.pkl")
        self.save_dir = save_dir
        self.s_time = time.time()
        self.env_type = env_type

        if env_type == 'target_tracking':
            self.records['mean_nlogdetcov'] = []
            init_pose_list = pickle.load(open(kwargs['init_file_path'], "rb")) if eval_type == 'fixed' else []
            nb_itrs = len(init_pose_list) if eval_type == 'fixed' else 5
            self.eval_f = lambda x : evaluation_ttenv(x, env_id, eval_type=eval_type,
                            nb_itrs=nb_itrs, init_pose_list=init_pose_list, **kwargs)
        else:
            self.eval_f = lambda x : evaluation(x, env_id, env_type, **kwargs)

    def log_reward(self, reward):
        self.ep_history['ep_rewards'][-1] += reward

    def log_step(self, **kwargs):
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
                    v.append(np.mean(self.step_history[k]))
                    self.step_history[k] = []

    def log_epoch(self, act, **kwargs):
        test_reward, mean_nlogdetcov = self.eval_f(act)
        for (k,v) in self.records.items():
            if k == 'online_reward':
                if len(self.ep_history['ep_rewards']) > 1:
                    v.append(round(np.mean(self.ep_history['ep_rewards'][-101:-1]),1))
            elif k == 'test_reward':
                v.append(test_reward)
            elif k == 'mean_nlogdetcov':
                v.append(mean_nlogdetcov)
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
        # Plot online reward and test reward.
        for (k,v) in self.records.items():
            v = np.array(v)
            if len(v.shape) == 3:
                for i in range(v.shape[1]):
                    f0, ax0 = plt.subplots()
                    m, ids25, ids75 = mstd(v[:,i,:].T)
                    # If the number of warmup steps > the number of epoch
                    # step or there is no episode completed, v may not have
                    # the same length with the x_vals.
                    _ = ax0.plot(x_vals[-len(v):], m, color='b')
                    _ = ax0.fill_between(x_vals[-len(v):], list(ids75),
                                        list(ids25), facecolor='b', alpha=0.2)
                    if k == 'mean_nlogdetcov':
                        _ = ax0.set_ylim(-1500, 3000)
                    _ = ax0.grid()
                    _ = ax0.set_ylabel(k)
                    _ = ax0.set_xlabel('Learning Steps')
                    if x_vals[-len(v)] < nb_warmup_steps:
                        _ = ax0.axvline(x=nb_warmup_steps, color='k')
                    _ = f0.savefig(os.path.join(self.save_dir, "%s_eval_%d.png"%(k,i)))

            elif len(v) > 0 and (k == 'test_reward' or k == 'mean_nlogdetcov' or k == 'online_reward'):
                f0, ax0 = plt.subplots()
                if len(v.shape) == 2:
                    m, ids25, ids75 = mstd(v.T)
                    # If the number of warmup steps > the number of epoch
                    # step or there is no episode completed, v may not have
                    # the same length with the x_vals.
                    _ = ax0.plot(x_vals[-len(v):], m, color='b')
                    _ = ax0.fill_between(x_vals[-len(v):], list(ids75),
                                        list(ids25), facecolor='b', alpha=0.2)
                else:
                    _ = ax0.plot(x_vals[-len(v):], v, color='b')

                if k == 'mean_nlogdetcov':
                    _ = ax0.set_ylim(-1500, 3000)
                _ = ax0.grid()
                _ = ax0.set_ylabel(k)
                _ = ax0.set_xlabel('Learning Steps')

                if x_vals[-len(v)] < nb_warmup_steps:
                    _ = ax0.axvline(x=nb_warmup_steps, color='k')
                _ = f0.savefig(os.path.join(self.save_dir, "%s.png"%k))

def evaluation(act, env_id, env_type, nb_test_steps=None, nb_itrs=5,
                render=False, **kwargs):
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
    env = envs.make(env_id, env_type, render=render, is_training=False, **kwargs)
    for _ in range(nb_itrs):
        obs = env.reset()
        if nb_test_steps is None: # Evaluate until an episode ends.
            done = False
            episode_reward, t = 0, 0
            while not done:
                if render:
                    env.render()
                action = act(np.array(obs)[None])[0]
                obs, rew, done, info = env.step(action)
                episode_reward += rew
                t += 1
                if done and (env_type=='atari') and (info['ale.lives'] != 0):
                    done = False
            total_rewards.append(episode_reward)
        else:
            t, episode_reward = 0, 0
            episodes = []
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
    return np.array(total_rewards, dtype=np.float32), None

def evaluation_ttenv(act, env_id, eval_type='random', nb_itrs=5, render=False, **kwargs):
    """
    Evaluation for the ttenv environments in a given set of different sampling
    zones. The set of different sampling zones is defined in TTENV_EVAL_SET.
    """
    if eval_type == 'random':
        params_set = [{}]
    elif eval_type == 'random_zone':
        params_set = TTENV_EVAL_SET
    elif eval_type == 'fixed':
        params_set = [{'init_pose_list':kwargs['init_pose_list']}]
    else:
        raise ValueError("Wrong evaluation type for ttenv.")

    env = envs.make(env_id, 'target_tracking', render=render, is_training=False, **kwargs)
    total_rewards, total_nlogdetcov = [], []
    for params in params_set:
        total_rewards_k, total_nlogdetcov_k = [], []
        for _ in range(nb_itrs):
            obs = env.reset(**params)
            done = False
            episode_reward, episode_nlogdetcov, t = 0, 0, 0
            while not done:
                if render:
                    env.render()
                action = act(np.array(obs)[None])[0]
                obs, rew, done, info = env.step(action)
                episode_reward += rew
                episode_nlogdetcov += info['mean_nlogdetcov']
                t += 1
            total_rewards_k.append(episode_reward)
            total_nlogdetcov_k.append(episode_nlogdetcov)
        total_rewards.append(total_rewards_k)
        total_nlogdetcov.append(total_nlogdetcov_k)
    if render:
        env.close()
    if len(total_rewards) == 1:
        total_rewards = total_rewards[0]
        total_nlogdetcov = total_nlogdetcov[0]
    return np.array(total_rewards, dtype=np.float32), np.array(total_nlogdetcov, dtype=np.float32)

def batch_plot(list_records, save_dir, nb_train_steps, nb_epoch_steps, is_target_tracking=False):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from util import mstd

    results = {'online_reward':[], 'test_reward':[]}
    if is_target_tracking:
        results['mean_nlogdetcov'] = []
    for r in list_records:
        for (k,v) in results.items():
            r[k] = np.array(r[k])
            v.append(r[k].T)
    x_vals = range(0, nb_train_steps+1, nb_epoch_steps)
    for (k,v) in results.items():
        v = np.array(v)
        if len(v.shape) == 4:
            for i in range(v.shape[2]):
                v_i = np.concatenate(v[:,:,i,:], axis=0)
                f0, ax0 = plt.subplots()
                m, ids25, ids75 = mstd(v_i)
                _ = ax0.plot(x_vals[-v_i.shape[1]:], m, color='k')
                _ = ax0.fill_between(x_vals[-v_i.shape[1]:], list(ids75), list(ids25),
                                facecolor='k', alpha=0.2)
                _ = ax0.plot(x_vals[-v_i.shape[1]:], np.max(v_i, axis=0), color='b')
                _ = ax0.plot(x_vals[-v_i.shape[1]:], np.min(v_i, axis=0), color='r')
                _ = ax0.grid()
                if k == 'mean_nlogdetcov':
                    ax0.set_ylim(-1500, 3000)
                _ = f0.savefig(os.path.join(save_dir, "%s_eval_%d.png"%(k,i)))
                plt.close()
        else:
            if len(v.shape) == 3:
                v = np.concatenate(v, axis=0)
            f0, ax0 = plt.subplots()
            m, ids25, ids75 = mstd(v)
            _ = ax0.plot(x_vals[-v.shape[1]:], m, color='k')
            _ = ax0.fill_between(x_vals[-v.shape[1]:], list(ids75), list(ids25),
                            facecolor='k', alpha=0.2)
            _ = ax0.plot(x_vals[-v.shape[1]:], np.max(v, axis=0), color='b')
            _ = ax0.plot(x_vals[-v.shape[1]:], np.min(v, axis=0), color='r')
            _ = ax0.grid()
            if k == 'mean_nlogdetcov':
                ax0.set_ylim(-1500, 3000)
            _ = f0.savefig(os.path.join(save_dir, k+".png"))
            plt.close()

TTENV_EVAL_SET = [{
        'lin_dist_range':(5.0, 10.0),
        'ang_dist_range_target':(-0.5*np.pi, 0.5*np.pi),
        'ang_dist_range_belief':(-0.25*np.pi, 0.25*np.pi),
        'blocked':False
        },
        {
        'lin_dist_range':(10.0, 15.0),
        'ang_dist_range_target':(-0.5*np.pi, 0.5*np.pi),
        'ang_dist_range_belief':(-0.25*np.pi, 0.25*np.pi),
        'blocked':True
        },
        { # target and beleif in the opposite direction
        'lin_dist_range':(5.0, 10.0),
        'ang_dist_range_target':(0.5*np.pi, -0.5*np.pi),
        'ang_dist_range_belief':(-0.25*np.pi, 0.25*np.pi),
        'blocked':False
        },
        { # target and beleif in the opposite direction
        'lin_dist_range':(10.0, 15.0),
        'ang_dist_range_target':(0.5*np.pi, -0.5*np.pi),
        'ang_dist_range_belief':(-0.25*np.pi, 0.25*np.pi),
        'blocked':True
        },
        { # target in the opposite direction but belief in the same direction
        'lin_dist_range':(5.0, 10.0),
        'ang_dist_range_target':(0.5*np.pi, -0.5*np.pi),
        'ang_dist_range_belief':(0.75*np.pi, -0.75*np.pi),
        'blocked':False
        },
        { #target in the opposite direction but belief in the same direction
        'lin_dist_range':(10.0, 15.0),
        'ang_dist_range_target':(0.5*np.pi, -0.5*np.pi),
        'ang_dist_range_belief':(0.75*np.pi, -0.75*np.pi),
        'blocked':True
        },
]
