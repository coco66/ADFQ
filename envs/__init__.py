""" envs/ folder is for openAIgym-like simulation environments
To use,
>>> import envs
>>> env = envs.make("NAME_OF_ENV")

"""
import gym

def make(env_name, type, render=False, record=False, directory='', **kwargs):
    """
    env_name : str
        name of an environment. (e.g. 'Cartpole-v0')
    type : str
        type of an environment. One of ['atari', 'classic_control',
        'classic_mdp','target_tracking']
    """
    if type == 'atari':
        from baselines0.common.atari_wrappers import make_atari
        from baselines0.common.atari_wrappers import wrap_deepmind
        from baselines0 import bench, logger

        env = make_atari(env_name)
        env = bench.Monitor(env, logger.get_dir())
        env = wrap_deepmind(env, frame_stack=True, scale=True)
        if record:
            env = Monitor(env, directory=directory)

    elif type == 'classic_control':
        env = gym.make(env_name)
        if record:
            env = Monitor(env, directory=directory)

    elif type == 'classic_mdp':
        from envs import classic_mdp
        env = classic_mdp.model_assign(env_name)

    elif type == 'target_tracking':
        import ttenv
        env = ttenv.make(env_name, render=render, record=record,
                                                directory=directory, **kwargs)
    elif type == 'ma_target_tracking':
        import maTTenv
        env = maTTenv.make(env_name, render=render, record=record,
                                                directory=directory, **kwargs)
    else:
        raise ValueError('Designate the right type of the environment.')

    return env
