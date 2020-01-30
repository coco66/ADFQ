"""
This code was slightly modified from the baselines0/baselines0/deepq/train_cartpole.py in order to use
a different evaluation method. In order to run, simply replace the original code with this code
in the original directory.
"""
import argparse
import tensorflow as tf
import datetime, json, os
import numpy as np

from baselines0.common import set_global_seeds
from baselines0 import deepq

import envs
from deep_adfq.logger import Logger

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--env', help='environment ID', default='TargetTracking-v1')
parser.add_argument('--seed', help='RNG seed', type=int, default=0)
parser.add_argument('--prioritized', type=int, default=0)
parser.add_argument('--prioritized-replay-alpha', type=float, default=0.6)
parser.add_argument('--double_q', type=int, default=0)
parser.add_argument('--mode', choices=['train', 'test'], default='train')
parser.add_argument('--dueling', type=int, default=0)
parser.add_argument('--nb_train_steps', type=int, default=5000)
parser.add_argument('--buffer_size', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--nb_warmup_steps', type=int, default = 100)
parser.add_argument('--nb_epoch_steps', type=int, default = 100)
parser.add_argument('--target_update_freq', type=float, default=50) # This should be smaller than epoch_steps
parser.add_argument('--nb_test_steps',type=int, default = None)
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--learning_rate_decay_factor', type=float, default=1.0)
parser.add_argument('--learning_rate_growth_factor', type=float, default=1.0)
parser.add_argument('--gamma', type=float, default=.99)
parser.add_argument('--num_layers', type=int, default=3)
parser.add_argument('--num_units', type=int, default=128)
parser.add_argument('--log_dir', type=str, default='.')
parser.add_argument('--log_fname', type=str, default='model.pkl')
parser.add_argument('--eps_fraction', type=float, default=0.1)
parser.add_argument('--eps_min', type=float, default=.02)
parser.add_argument('--test_eps', type=float, default=.05)
parser.add_argument('--device', type=str, default='/gpu:0')
parser.add_argument('--record',type=int, default=0)
parser.add_argument('--render', type=int, default=0)
parser.add_argument('--gpu_memory',type=float, default=1.0)
parser.add_argument('--repeat', type=int, default=1)
parser.add_argument('--scope',type=str, default='deepq')
parser.add_argument('--ros', type=int, default=0)
parser.add_argument('--ros_log', type=int, default=0)
parser.add_argument('--map', type=str, default="emptySmall")
parser.add_argument('--nb_targets', type=int, default=1)
parser.add_argument('--im_size', type=int, default=50)
parser.add_argument('--pooling', type=int, default=0)

args = parser.parse_args()

def train(seed, save_dir):
    set_global_seeds(seed)
    save_dir_0 = os.path.join(save_dir, 'batch_%d'%seed)
    os.makedirs(save_dir_0)

    env = envs.make(args.env,
                    'target_tracking',
                    render=bool(args.render),
                    record=bool(args.record),
                    directory=save_dir_0,
                    ros=bool(args.ros),
                    map_name=args.map,
                    num_targets=args.nb_targets,
                    im_size=args.im_size,
                    )

    with tf.device(args.device):
        model = deepq.models.mlp([args.num_units]*args.num_layers)
        act = deepq.learn(
            env,
            q_func=model,
            lr=args.learning_rate,
            lr_decay_factor=args.learning_rate_decay_factor,
            lr_growth_factor=args.learning_rate_growth_factor,
            max_timesteps=args.nb_train_steps,
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            exploration_fraction=args.eps_fraction,
            exploration_final_eps=args.eps_min,
            target_network_update_freq=args.target_update_freq,
            print_freq=10,
            checkpoint_freq=int(args.nb_train_steps/10),
            learning_starts=args.nb_warmup_steps,
            gamma = args.gamma,
            prioritized_replay=bool(args.prioritized),
            prioritized_replay_alpha=args.prioritized_replay_alpha,
            callback=None,#callback,
            double_q = args.double_q,
            scope=args.scope,
            epoch_steps = args.nb_epoch_steps,
            eval_logger=Logger(args.env, 'target_tracking',
                    save_dir=save_dir_0,
                    render=bool(args.render),
                    figID=1,
                    ros=bool(args.ros),
                    map_name=args.map,
                    num_targets=args.nb_targets,
                    im_size=args.im_size),
            save_dir=save_dir_0,
            test_eps = args.test_eps,
            gpu_memory=args.gpu_memory,
            render = (bool(args.render) or bool(args.ros)),
        )
        print("Saving model to model.pkl")
        act.save(os.path.join(save_dir_0,"model.pkl"))
    if args.record == 1:
        env.moviewriter.finish()

def test():
    learning_prop = json.load(open(os.path.join(args.log_dir, 'learning_prop.json'),'r'))
    env = envs.make(args.env,
                    'target_tracking',
                    render=bool(args.render),
                    record=bool(args.record),
                    directory=args.log_dir,
                    ros=bool(args.ros),
                    map_name=args.map,
                    num_targets=learning_prop['nb_targets'],
                    im_size=learning_prop['im_size'],
                    is_training=False,
                    )
    timelimit_env = env
    while( not hasattr(timelimit_env, '_elapsed_steps')):
        timelimit_env = timelimit_env.env

    act_params = {'scope': learning_prop['scope'], 'eps': args.test_eps}
    act = simple.load(os.path.join(args.log_dir, args.log_fname), act_params)

    if args.ros_log:
        from envs.target_tracking.ros_wrapper import RosLog
        ros_log = RosLog(num_targets=args.nb_targets, wrapped_num=args.ros + args.render + args.record + 1)

    ep = 0
    init_pos = []
    ep_nlogdetcov = ['Episode nLogDetCov']
    time_elapsed = ['Elapsed Time (sec)']
    while(ep < args.nb_test_steps): # test episode
        ep += 1
        obs, done = env.reset(), False
        episode_rew = 0
        init_pos.append({'agent':timelimit_env.env.agent.state,
                            'targets':[timelimit_env.env.targets[i].state for i in range(args.nb_targets)],
                            'belief_targets':[timelimit_env.env.belief_targets[i].state for i in range(args.nb_targets)]})
        s_time = time.time()
        nlogdetcov = 0
        while not done:
            if args.render:
                env.render()
            if args.ros_log:
                ros_log.log(env)
            obs, rew, done, _ = env.step(act(obs[None])[0])
            episode_rew += rew
            nlogdetcov += info['test_reward']

        time_elapsed.append(time.time() - s_time)
        ep_nlogdetcov.append(nlogdetcov)
        print("Episode reward : %.2f, Episode nLogDetCov : %.2f"%(episode_rew, nlogdetcov))

    if args.record :
        env.moviewriter.finish()
    if args.ros_log :
        ros_log.save(args.log_dir)
    import pickle
    pickle.dump(init_pos, open(os.path.join(args.log_dir,'test_init_log.pkl'), 'wb'))

if __name__ == '__main__':
    if args.mode == 'train':
        save_dir = os.path.join(args.log_dir, '_'.join([args.env, datetime.datetime.now().strftime("%m%d%H%M")]))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        else:
            ValueError("The directory already exists...", save_dir)
        json.dump(vars(args), open(os.path.join(save_dir, 'learning_prop.json'), 'w'))
        seed = args.seed
        for _ in range(args.repeat):
            print("===== TRAIN A TARGET TRACKING RL AGENT : SEED %d ====="%seed)
            results = train(seed, save_dir)
            seed += 1
        notes = input("Any notes for this experiment? : ")
        f = open(os.path.join(save_dir, "notes.txt"), 'w')
        f.write(notes)
        f.close()
    elif args.mode =='test':
        test()
