""" envs/ folder is for openAIgym-like simulation environments
To use,
>>> import envs
>>> env = envs.make("NAME_OF_ENV")

"""
import gym

def make(env_name, type, render=False, figID=0, record=False,
                    ros=False, directory='', T_steps=None, **kwargs):
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
        from gym import wrappers
        import envs.target_tracking.target_tracking as ttenv

        if T_steps is None:
            if kwargs['num_targets'] > 1:
                T_steps = 150
            else:
                T_steps = 100
        if env_name == 'TargetTracking-v0':
            env0 = ttenv.TargetTrackingEnv0(**kwargs)
        elif env_name == 'TargetTracking-v1':
            env0 = ttenv.TargetTrackingEnv1(**kwargs)
        elif env_name == 'TargetTracking-v2':
            env0 = ttenv.TargetTrackingEnv2(**kwargs)
        elif env_name == 'TargetTracking-v3':
            env0 = ttenv.TargetTrackingEnv3(**kwargs)
        elif env_name == 'TargetTracking-v4':
            env0 = ttenv.TargetTrackingEnv4(**kwargs)
        elif env_name == 'TargetTracking-v5':
            from envs.target_tracking.target_imtracking import TargetTrackingEnv5
            env0 = TargetTrackingEnv5(**kwargs)

        elif env_name == 'TargetTracking-vRNN':
            from envs.target_tracking.target_tracking_advanced import TargetTrackingEnvRNN
            env0 = TargetTrackingEnvRNN(**kwargs)
            T_steps = 200
        elif env_name == 'TargetTracking-info1':
            from envs.target_tracking.inforplanner.target_tracking_infoplanner import TargetTrackingInfoPlanner1
            env0 = TargetTrackingInfoPlanner1(**kwargs)
        else:
            raise ValueError('no such environments')

        env = wrappers.TimeLimit(env0, max_episode_steps=T_steps)
        if ros:
            from envs.ros_wrapper import Ros
            env = Ros(env)
        if render:
            from envs.target_tracking import display_wrapper
            env = display_wrapper.Display2D(env, figID=figID)
        if record:
            from envs.target_tracking import display_wrapper
            env = display_wrapper.Video2D(env, dirname = directory)
    else:
        raise ValueError('Designate the right type of the environment.')

    return env
