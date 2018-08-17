"""
This code was slightly modified from the baselines/baselines/deepq/simple.py in order to use
a different evaluation method. In order to run, simply replace the original code with this code
in the original directory.
"""
import os
import tempfile
from tabulate import tabulate
import pickle

import tensorflow as tf
import zipfile
import cloudpickle
import numpy as np

import gym
import baselines.common.tf_util as U
from baselines.common.tf_util import load_state, save_state
from baselines import logger
from baselines.common.schedules import LinearSchedule
from baselines.common.input import observation_input

from baselines import deepq
from baselines.deepq.build_graph import build_act_greedy
from baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from baselines.deepq.utils import ObservationInput


class ActWrapper(object):
    def __init__(self, act, act_params):
        self._act = act
        self._act_params = act_params

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            model_data, act_params = cloudpickle.load(f)
        act = build_act_greedy(**act_params)
        sess = tf.Session()
        sess.__enter__()
        with tempfile.TemporaryDirectory() as td:
            arc_path = os.path.join(td, "packed.zip")
            with open(arc_path, "wb") as f:
                f.write(model_data)

            zipfile.ZipFile(arc_path, 'r', zipfile.ZIP_DEFLATED).extractall(td)
            load_state(os.path.join(td, "model"))

        return ActWrapper(act, act_params)

    def __call__(self, *args, **kwargs):
        return self._act(*args, **kwargs)

    def save(self, path=None):
        """Save model to a pickle located at `path`"""
        if path is None:
            path = os.path.join(logger.get_dir(), "model.pkl")

        with tempfile.TemporaryDirectory() as td:
            save_state(os.path.join(td, "model"))
            arc_name = os.path.join(td, "packed.zip")
            with zipfile.ZipFile(arc_name, 'w') as zipf:
                for root, dirs, files in os.walk(td):
                    for fname in files:
                        file_path = os.path.join(root, fname)
                        if file_path != arc_name:
                            zipf.write(file_path, os.path.relpath(file_path, td))
            with open(arc_name, "rb") as f:
                model_data = f.read()
        with open(path, "wb") as f:
            cloudpickle.dump((model_data, self._act_params), f)


def load(path):
    """Load act function that was returned by learn function.

    Parameters
    ----------
    path: str
        path to the act function pickle

    Returns
    -------
    act: ActWrapper
        function that takes a batch of observations
        and returns actions.
    """
    return ActWrapper.load(path)


def learn(env,
          q_func,
          lr=5e-4,
          max_timesteps=100000,
          buffer_size=50000,
          exploration_fraction=0.1,
          exploration_final_eps=0.02,
          train_freq=1,
          batch_size=32,
          print_freq=100,
          checkpoint_freq=10000,
          checkpoint_path=None,
          learning_starts=1000,
          gamma=1.0,
          target_network_update_freq=500,
          prioritized_replay=False,
          prioritized_replay_alpha=0.6,
          prioritized_replay_beta0=0.4,
          prioritized_replay_beta_iters=None,
          prioritized_replay_eps=1e-6,
          param_noise=False,
          callback=None,
          epoch_steps=20000,
          gpu_memory=1.0,
          double_q=False,
          scope="deepq",
          directory='.',
          nb_test_steps=10000,
          ):
    """Train a deepq model.

    Parameters
    -------
    env: gym.Env
        environment to train on
    q_func: (tf.Variable, int, str, bool) -> tf.Variable
        the model that takes the following inputs:
            observation_in: object
                the output of observation placeholder
            num_actions: int
                number of actions
            scope: str
            reuse: bool
                should be passed to outer variable scope
        and returns a tensor of shape (batch_size, num_actions) with values of every action.
    lr: float
        learning rate for adam optimizer
    max_timesteps: int
        number of env steps to optimizer for
    buffer_size: int
        size of the replay buffer
    exploration_fraction: float
        fraction of entire training period over which the exploration rate is annealed
    exploration_final_eps: float
        final value of random action probability
    train_freq: int
        update the model every `train_freq` steps.
        set to None to disable printing
    batch_size: int
        size of a batched sampled from replay buffer for training
    print_freq: int
        how often to print out training progress
        set to None to disable printing
    checkpoint_freq: int
        how often to save the model. This is so that the best version is restored
        at the end of the training. If you do not wish to restore the best version at
        the end of the training set this variable to None.
    learning_starts: int
        how many steps of the model to collect transitions for before learning starts
    gamma: float
        discount factor
    target_network_update_freq: int
        update the target network every `target_network_update_freq` steps.
    prioritized_replay: True
        if True prioritized replay buffer will be used.
    prioritized_replay_alpha: float
        alpha parameter for prioritized replay buffer
    prioritized_replay_beta0: float
        initial value of beta for prioritized replay buffer
    prioritized_replay_beta_iters: int
        number of iterations over which beta will be annealed from initial value
        to 1.0. If set to None equals to max_timesteps.
    prioritized_replay_eps: float
        epsilon to add to the TD errors when updating priorities.
    callback: (locals, globals) -> None
        function called at every steps with state of the algorithm.
        If callback returns true training stops.

    Returns
    -------
    act: ActWrapper
        Wrapper over act function. Adds ability to save it and load it.
        See header of baselines/deepq/categorical.py for details on the act function.
    """
    # Create all the functions necessary to train the model
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.per_process_gpu_memory_fraction = gpu_memory
    config.gpu_options.polling_inactive_delay_msecs = 25
    sess = tf.Session(config=config)
    sess.__enter__()

    # capture the shape outside the closure so that the env object is not serialized
    # by cloudpickle when serializing make_obs_ph

    def make_obs_ph(name):
        return ObservationInput(env.observation_space, name=name)

    act, act_greedy, train, update_target, debug = deepq.build_train(
        make_obs_ph=make_obs_ph,
        q_func=q_func,
        num_actions=env.action_space.n,
        optimizer=tf.train.AdamOptimizer(learning_rate=lr),
        gamma=gamma,
        grad_norm_clipping=10,
        param_noise=param_noise,
        double_q = bool(double_q),
        scope=scope
    )

    act_params = {
        'make_obs_ph': make_obs_ph,
        'q_func': q_func,
        'num_actions': env.action_space.n,
    }

    act = ActWrapper(act, act_params)

    # Create the replay buffer
    if prioritized_replay:
        replay_buffer = PrioritizedReplayBuffer(buffer_size, alpha=prioritized_replay_alpha)
        if prioritized_replay_beta_iters is None:
            prioritized_replay_beta_iters = max_timesteps
        beta_schedule = LinearSchedule(prioritized_replay_beta_iters,
                                       initial_p=prioritized_replay_beta0,
                                       final_p=1.0)
    else:
        replay_buffer = ReplayBuffer(buffer_size)
        beta_schedule = None
    # Create the schedule for exploration starting from 1.
    exploration = LinearSchedule(schedule_timesteps=int(exploration_fraction * max_timesteps),
                                 initial_p=1.0,
                                 final_p=exploration_final_eps)

    # Initialize the parameters and copy them to the target network.
    U.initialize()
    update_target()

    episode_rewards = [0.0]
    saved_mean_reward = None
    obs = env.reset()
    reset = True

    #recording
    records = { 'loss':[], 'online_reward':[], 'test_reward':[]}

    with tempfile.TemporaryDirectory() as td:
        td = checkpoint_path or td

        model_file = os.path.join(td, "model")
        model_saved = False
        if tf.train.latest_checkpoint(td) is not None:
            load_state(model_file)
            logger.log('Loaded model from {}'.format(model_file))
            model_saved = True
        
        ep_losses, ep_means, losses = [], [], []
        print("===== LEARNING STARTS =====")
        for t in range(max_timesteps):
            if callback is not None:
                if callback(locals(), globals()):
                    break
            # Take action and update exploration to the newest value
            kwargs = {}
            if not param_noise:
                update_eps = exploration.value(t)
                update_param_noise_threshold = 0.
            else:
                update_eps = 0.
                # Compute the threshold such that the KL divergence between perturbed and non-perturbed
                # policy is comparable to eps-greedy exploration with eps = exploration.value(t).
                # See Appendix C.1 in Parameter Space Noise for Exploration, Plappert et al., 2017
                # for detailed explanation.
                update_param_noise_threshold = -np.log(1. - exploration.value(t) + exploration.value(t) / float(env.action_space.n))
                kwargs['reset'] = reset
                kwargs['update_param_noise_threshold'] = update_param_noise_threshold
                kwargs['update_param_noise_scale'] = True
            action = act(np.array(obs)[None], update_eps=update_eps, **kwargs)[0]
            env_action = action
            reset = False
            new_obs, rew, done, _ = env.step(env_action)
            # Store transition in the replay buffer.
            timelimit_env = env
            while( not hasattr(timelimit_env, '_elapsed_steps')):
                timelimit_env = timelimit_env.env

            if timelimit_env._elapsed_steps < timelimit_env._max_episode_steps:
            # Store transition in the replay buffer.
                replay_buffer.add(obs, action, rew, new_obs, float(done))
            else:
                replay_buffer.add(obs, action, rew, new_obs, float(not done))
            
            obs = new_obs

            episode_rewards[-1] += rew
            if done:
                obs = env.reset()
                episode_rewards.append(0.0)
                reset = True
                if losses:
                    ep_losses.append(np.mean(losses))
                    losses = []

            if t > learning_starts and t % train_freq == 0:
                # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                if prioritized_replay:
                    experience = replay_buffer.sample(batch_size, beta=beta_schedule.value(t))
                    (obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes) = experience
                else:
                    obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(batch_size)
                    weights, batch_idxes = np.ones_like(rewards), None
                td_errors = train(obses_t, actions, rewards, obses_tp1, dones, weights)
                losses.append(td_errors)
                if prioritized_replay:
                    new_priorities = np.abs(td_errors) + prioritized_replay_eps
                    replay_buffer.update_priorities(batch_idxes, new_priorities)

            if t > learning_starts and t % target_network_update_freq == 0:
                # Update target network periodically.
                update_target()

            mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
            num_episodes = len(episode_rewards)
            
            if (t+1) % epoch_steps == 0 and (t+1) > learning_starts:
                test_reward = test(env, act_greedy, nb_test_steps=nb_test_steps)
                records['test_reward'].append(test_reward)
                records['loss'].append(np.mean(ep_losses))
                records['online_reward'].append(round(np.mean(episode_rewards[-101:-1]), 1))
                pickle.dump(records, open(os.path.join(directory,"records.pkl"),"wb"))
                print("==== EPOCH %d ==="%((t+1)/epoch_steps))
                print(tabulate([[k,v[-1]] for (k,v) in records.items()]))
            
            if done and print_freq is not None and len(episode_rewards) % print_freq == 0:
                logger.record_tabular("steps", t)
                logger.record_tabular("episodes", num_episodes)
                logger.record_tabular("mean 100 episode reward", mean_100ep_reward)
                logger.record_tabular("% time spent exploring", int(100 * exploration.value(t)))
                logger.dump_tabular()

            if (checkpoint_freq is not None and (t+1) > learning_starts and
                    num_episodes > 100 and (t+1) % checkpoint_freq == 0):
                print("Saving model to model_%d.pkl"%(t+1))
                act.save(os.path.join(directory,"model_"+str(t+1)+".pkl"))
                if saved_mean_reward is None or mean_100ep_reward > saved_mean_reward:
                    if print_freq is not None:
                        logger.log("Saving model due to mean reward increase: {} -> {}".format(
                                   saved_mean_reward, mean_100ep_reward))
                    save_state(model_file)
                    model_saved = True
                    saved_mean_reward = mean_100ep_reward
                    
        if model_saved:
            if print_freq is not None:
                logger.log("Restored model with mean reward: {}".format(saved_mean_reward))
            load_state(model_file)
    return act, records

def test(env0, act_greedy, nb_itrs=5, nb_test_steps=10000):
    
    env = env0
    while( hasattr(env, 'env')):
        env = env.env

    total_rewards = []
    for _ in range(nb_itrs):
        if hasattr(env, 'ale'):
            from baselines.common.atari_wrappers import make_atari
            env_new = make_atari(env.spec.id)
            env_new = deepq.wrap_atari_dqn(env_new)
        else:
            env_new = gym.make(env.spec.id)
        obs = env_new.reset()

        if nb_test_steps is None:
            done = False
            episode_reward = 0
            while not done:
                action = act_greedy(np.array(obs)[None])[0]
                obs, rew, done, _ = env_new.step(action)
                episode_reward += rew
            total_rewards.append(episode_reward)
        else:
            t = 0
            episodes = []
            episode_reward = 0
            while(t < nb_test_steps):
                action = act_greedy(np.array(obs)[None])[0]
                obs, rew, done, _ = env_new.step(action)
                episode_reward += rew
                if done:
                    episodes.append(episode_reward)
                    episode_reward = 0
                    obs = env_new.reset()
                t += 1
            total_rewards.append(np.mean(episodes))

    return np.array(total_rewards, dtype=np.float32)


