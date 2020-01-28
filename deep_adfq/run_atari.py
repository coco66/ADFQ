"""
This code was modified from a OpenAI baseline code - baselines0/baselines0/deepq/experiments/run_atari.py
for running ADFQ in Atari enviornment
"""

from baselines0.common import set_global_seeds
from baselines0 import logger

import models
import deepadfq
from logger import Logger
import envs

import numpy as np
import tensorflow as tf
import os, datetime, json, argparse, time

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--env', help='environment ID', default='BreakoutNoFrameskip-v4')
parser.add_argument('--seed', help='RNG seed', type=int, default=0)
parser.add_argument('--prioritized', type=int, default=1)
parser.add_argument('--prioritized-replay-alpha', type=float, default=0.6)
parser.add_argument('--mode', choices=['train', 'test'], default='train')
parser.add_argument('--dueling', type=int, default=0)
parser.add_argument('--nb_train_steps', type=int, default=int(5*1e6))
parser.add_argument('--nb_warmup_steps', type=int, default = 10000)
parser.add_argument('--nb_test_steps',type=int, default = 10000)
parser.add_argument('--nb_epoch_steps', type=int, default = 50000)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--target_update_freq', type=float, default=1000)
parser.add_argument('--learning_rate', type=float, default=0.00025)
parser.add_argument('--lr_decay_factor', type=float, default=1.0)
parser.add_argument('--lr_growth_factor', type=float, default=1.0)
parser.add_argument('--gamma', type=float, default=.99)
parser.add_argument('--num_layers', type=int, default=1)
parser.add_argument('--num_units', type=int, default=256)
parser.add_argument('--log_dir', type=str, default='.')
parser.add_argument('--log_fname', type=str, default='model.pkl')
parser.add_argument('--reload_path', type=str, default='')
parser.add_argument('--init_t', type=int, default=0)
parser.add_argument('--buffer_size', type=int, default=50000)
parser.add_argument('--eps_fraction', type=float, default=0.1)
parser.add_argument('--eps_min', type=float, default=.01)
parser.add_argument('--test_eps', type=float, default=.0)
parser.add_argument('--init_mean', type =float, default=1.)
parser.add_argument('--init_sd', type=float, default=50.)
parser.add_argument('--device', type=str, default='/cpu:0')
parser.add_argument('--gpu_memory',type=float, default=1.0)
parser.add_argument('--alg', choices=['adfq','adfq-v2'], default='adfq')
parser.add_argument('--act_policy', choices=['egreedy','bayesian'], default='egreedy')
parser.add_argument('--record',type=int, default=0)
parser.add_argument('--render', type=int, default=0)
parser.add_argument('--varth', type=float,default=1e-5)
parser.add_argument('--noise', type=float,default=0.0)
parser.add_argument('--repeat', type=int, default=1)
parser.add_argument('--scope',type=str, default='deepadfq')

args = parser.parse_args()

def train(seed, save_dir):
    set_global_seeds(seed)
    save_dir_0 = os.path.join(save_dir, 'seed_%d'%seed)
    os.makedirs(save_dir_0)

    env = envs.make(args.env, 'atari', record = bool(args.record), directory=save_dir_0)

    nb_test_steps = args.nb_test_steps if args.nb_test_steps > 0 else None
    reload_path = args.reload_path if args.reload_path else None

    with tf.device(args.device):
        with tf.compat.v1.variable_scope('seed_%d'%seed):
            model = models.cnn_to_mlp(
                convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
                hiddens=[args.num_units]*args.num_layers,
                dueling=bool(args.dueling),
                init_mean = args.init_mean,
                init_sd = args.init_sd,
            )

            act = deepadfq.learn(
                env,
                q_func=model,
                lr=args.learning_rate,
                lr_decay_factor=args.lr_decay_factor,
                lr_growth_factor=args.lr_growth_factor,
                max_timesteps=args.nb_train_steps,
                buffer_size=args.buffer_size,
                batch_size=args.batch_size,
                exploration_fraction=args.eps_fraction,
                exploration_final_eps=args.eps_min,
                train_freq=4,
                print_freq=args.nb_epoch_steps,
                checkpoint_freq=int(args.nb_train_steps/5),
                checkpoint_path=reload_path,
                learning_starts=args.nb_warmup_steps,
                target_network_update_freq=args.target_update_freq,
                gamma=args.gamma,
                prioritized_replay=bool(args.prioritized),
                prioritized_replay_alpha=args.prioritized_replay_alpha,
                scope=args.scope,
                alg=args.alg,
                sdMin=np.sqrt(args.varth),
                noise=args.noise,
                act_policy=args.act_policy,
                epoch_steps=args.nb_epoch_steps,
                eval_logger=Logger(args.env, 'atari',
                        nb_test_steps=nb_test_steps, save_dir=save_dir_0,
                        render=bool(args.render)),
                save_dir=save_dir_0,
                test_eps=args.test_eps,
                init_t=args.init_t,
                gpu_memory=args.gpu_memory,
                render=bool(args.render)
            )
            print("Saving model to model.pkl")
            act.save(os.path.join(save_dir_0,"model.pkl"))
    env.close()
    if args.record == 1:
        env.moviewriter.finish()

def test():
    env = envs.make(args.env, 'atari', render = bool(args.render),
                    record = bool(args.record), directory=args.log_dir)
    learning_prop = json.load(open(os.path.join(args.log_dir, '../learning_prop.json'),'r'))
    act_params = {'scope': "seed_%d"%learning_prop['seed']+"/"+learning_prop['scope'], 'eps': args.test_eps}
    act = deepadfq.load(os.path.join(args.log_dir, args.log_fname), act_params)
    episode_rew = 0
    t = 0
    while True:
        obs, done = env.reset(), False
        while(not done):
            if args.render:
                env.render()
                time.sleep(0.05)
            obs, rew, done, info = env.step(act(obs[None])[0])
            # Reset only the enviornment but not the recorder
            if args.record and done:
                obs, done = env.env.reset(), False
            episode_rew += rew
            t += 1
        if info['ale.lives'] == 0:
            print("Episode reward %.2f after %d steps"%(episode_rew, t))
            episode_rew = 0
            t = 0

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
            print("===== TRAIN AN ATARI RL AGENT : SEED %d ====="%seed)
            train(seed, save_dir)
            seed += 1
        notes = input("Any notes for this experiment? : ")
        f = open(os.path.join(save_dir, "notes.txt"), 'w')
        f.write(notes)
        f.close()
    elif args.mode =='test':
        test()
