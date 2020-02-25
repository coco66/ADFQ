"""
This code was slightly modified from the baselines0/baselines0/deepq/simple.py in order to use
a different evaluation method. In order to run, simply replace the original code with this code
in the original directory.
"""
import os
import tempfile

import tensorflow as tf
import zipfile
import cloudpickle
import numpy as np

import baselines0.common.tf_util as U
from baselines0 import logger
from baselines0.common.schedules import LinearSchedule

from baselines0 import setdeepq
from baselines0.setdeepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from baselines0.setdeepq.utils import BatchInput, load_state, save_state

class ActWrapper(object):
    def __init__(self, act, act_params):
        self._act = act
        self._act_params = act_params

    @staticmethod
    def load(path, act_params_new=None):
        with open(path, "rb") as f:
            model_data, act_params = cloudpickle.load(f)
        if act_params_new:
            for (k,v) in act_params_new.items():
                act_params[k] = v

        act = setdeepq.build_act_greedy(reuse=tf.compat.v1.AUTO_REUSE, **act_params)
        sess = tf.compat.v1.Session()
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


def load(path, act_params=None):
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
    return ActWrapper.load(path, act_params)


def learn(env,
          q_func,
          lr=5e-4,
          lr_decay_factor = 0.99,
          lr_growth_factor = 1.01,
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
          scope="setdeepq",
          double_q=False,
          epoch_steps=20000,
          eval_logger=None,
          save_dir='.',
          test_eps=0.05,
          gpu_memory=1.0,
          render=False,
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
    target_network_update_freq: float
        update the target network every `target_network_update_freq` steps.
        If it is less than 1, it performs the soft target network update with
        the given rate.
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
    scope : str
        scope of the network.
    double_q : bool
        True if you run Double DQN.
    epoch_step : int
        the number of steps per epoch.
    eval_logger : Logger()
        the Logger() class object under deep_adfq folder.
    save_dir : str
        path for saving results.
    test_eps : float
        epsilon of the epsilon greedy action policy during testing.
    init_t : int
        an initial learning step if you start training from a pre-trained model.
    gpu_memory : float
        a fraction of a gpu memory when running multiple programs in the same gpu.

    Returns
    -------
    act: ActWrapper
        Wrapper over act function. Adds ability to save it and load it.
        See header of baselines0/deepq/categorical.py for details on the act function.
    """
    # Create all the functions necessary to train the model
    config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
    config.gpu_options.per_process_gpu_memory_fraction = gpu_memory
    config.gpu_options.polling_inactive_delay_msecs = 25
    sess = tf.compat.v1.Session(config=config)
    sess.__enter__()

    # capture the shape outside the closure so that the env object is not serialized
    # by cloudpickle when serializing make_obs_ph
    # Make observation a Set with variable size [N,?,d_obs]
    observation_space_shape = [None] + list(env.observation_space.shape)

    def make_obs_ph(name):
        return BatchInput(observation_space_shape, name=name)

    target_network_update_rate = np.minimum(target_network_update_freq, 1.0)
    target_network_update_freq = np.maximum(target_network_update_freq, 1.0)

    act, act_test, q_values, train, update_target, lr_decay_op, lr_growth_op, _ = setdeepq.build_train(
        make_obs_ph=make_obs_ph,
        q_func=q_func,
        num_actions=env.action_space.n,
        optimizer_f=tf.compat.v1.train.AdamOptimizer,
        gamma=gamma,
        grad_norm_clipping=10,
        param_noise=param_noise,
        double_q=bool(double_q),
        scope=scope,
        test_eps=test_eps,
        lr_init=lr,
        lr_decay_factor=lr_decay_factor,
        lr_growth_factor=lr_growth_factor,
        reuse=tf.compat.v1.AUTO_REUSE,
        tau=target_network_update_rate,
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

    file_writer = tf.compat.v1.summary.FileWriter(save_dir, sess.graph)
    # Initialize the parameters and copy them to the target network.
    U.initialize()
    update_target()

    saved_mean_reward = None
    timelimit_env = env
    while(not hasattr(timelimit_env, '_elapsed_steps')):
        timelimit_env = timelimit_env.env
    obs = env.reset()
    reset = True

    with tempfile.TemporaryDirectory() as td:
        td = checkpoint_path or td
        model_file = os.path.join(td, "model")
        model_saved = False
        if tf.train.latest_checkpoint(td) is not None:
            load_state(model_file)
            logger.log('Loaded model from {}'.format(model_file))
            model_saved = True

        checkpt_loss = []
        eval_logger.log_epoch(act_test)

        for t in range(max_timesteps):
            if callback is not None and callback(locals(), globals()):
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

            action_dict = {}
            for agent_id, a_obs in obs.items():
                action_dict[agent_id] = act(np.array(a_obs)[None], update_eps=update_eps, **kwargs)[0]
            reset = False
            new_obs, rew, done, info = env.step(action_dict)
            # Store transition in the replay buffer.
            for agent_id, a_obs in obs.items():
                if timelimit_env._elapsed_steps < timelimit_env._max_episode_steps:
                    replay_buffer.add(a_obs, action_dict[agent_id], rew['__all__'], new_obs[agent_id], float(done['__all__']))
                else:
                    replay_buffer.add(a_obs, action_dict[agent_id], rew['__all__'], new_obs[agent_id], float(not done))

            obs = new_obs
            eval_logger.log_reward(rew['__all__'])

            if type(done) is not dict:
                obs = env.reset()
                reset = True
                eval_logger.log_ep(info['mean_nlogdetcov'])

            if t > learning_starts and (t+1) % train_freq == 0:
                # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                if prioritized_replay:
                    experience = replay_buffer.sample(batch_size, beta=beta_schedule.value(t))
                    (obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes) = experience
                else:
                    obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(batch_size)
                    weights, batch_idxes = np.ones_like(rewards), None

                td_errors, loss, summary = train(obses_t, actions, rewards, obses_tp1, dones, weights)

                file_writer.add_summary(summary, t)
                eval_logger.log_step(loss=loss)

                if prioritized_replay:
                    new_priorities = np.abs(td_errors) + prioritized_replay_eps
                    replay_buffer.update_priorities(batch_idxes, new_priorities)
                if render:
                    env.render()

            if t > learning_starts and (t+1) % target_network_update_freq == 0:
                # Update target network periodically.
                update_target()

            if (t+1) % epoch_steps == 0:
                eval_logger.log_epoch(act_test)

            if (checkpoint_freq is not None and t > learning_starts and
                    (t+1) % checkpoint_freq == 0 and eval_logger.get_num_episode() > 10):
                mean_loss = np.float16(np.mean(eval_logger.ep_history['loss']))
                if len(checkpt_loss) > 2 and mean_loss > np.float16(max(checkpt_loss[-3:])) and lr_decay_factor < 1.0:
                    sess.run(lr_decay_op)
                    print("Learning rate decayed due to an increase in loss: %.4f -> %.4f"%(np.float16(max(checkpt_loss[-3:])),mean_loss))
                elif len(checkpt_loss) > 2 and mean_loss < np.float16(min(checkpt_loss[-3:])) and lr_growth_factor > 1.0:
                    sess.run(lr_growth_op)
                    print("Learning rate grown due to a decrease in loss: %.4f -> %.4f"%( np.float16(min(checkpt_loss[-3:])),mean_loss))
                checkpt_loss.append(mean_loss)
                # print("Saving model to model_%d.pkl"%(t+1))
                # act.save(os.path.join(save_dir,"model_"+str(t+1)+".pkl"))
                mean_100ep_reward = eval_logger.get_100ep_reward()
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
    eval_logger.finish(max_timesteps, epoch_steps, learning_starts)
    return act
