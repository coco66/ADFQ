from baselines.common import set_global_seeds
from baselines.common.atari_wrappers import make_atari

from baselines import bench
import argparse
import logger
import models
import simple
import tensorflow as tf
import os
import datetime, json

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='BreakoutNoFrameskip-v4')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--prioritized', type=int, default=1)
    parser.add_argument('--prioritized-replay-alpha', type=float, default=0.6)

    parser.add_argument('--dueling', type=int, default=0)
    parser.add_argument('--num_timesteps', type=int, default=int(3*1e6))
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--nb_steps_warmup', type=int, default = 10000)
    parser.add_argument('--epoch_steps', type=int, default = 20000)
    parser.add_argument('--target_update_freq', type=int, default=1000)
    parser.add_argument('--learning_rate', type=float, default=0.00025)
    parser.add_argument('--gamma', type=float, default=.99)
    parser.add_argument('--log_dir', type=str, default='.')
    parser.add_argument('--buffer_size', type=int, default=10000)
    parser.add_argument('--eps_max', type=float, default=0.1)
    parser.add_argument('--eps_min', type=float, default=.01)
    parser.add_argument('--init_mean', type =float, default=1.)
    parser.add_argument('--init_sd', type=float, default=50.)
    parser.add_argument('--device', type=str, default='/gpu:0')
    parser.add_argument('--alg', choices=['softapprox','hardapprox'], default='softapprox')
    parser.add_argument('--record',type=int, default=0)
    parser.add_argument('--gpu_memory',type=float, default=1.0)
    parser.add_argument('--varth', type=float,default=1e-5)
    parser.add_argument('--noise', type=float,default=1e-5)
    parser.add_argument('--act_policy', choices=['egreedy','bayesian'], default='egreedy')

    args = parser.parse_args()
    logger.configure()
    set_global_seeds(args.seed)

    directory = datetime.datetime.now().strftime("%m%d%H%M")
    if not os.path.exists(directory):
            os.makedirs(directory)
    else:
            ValueError("The directory already exists...", directory)
    json.dump(vars(args), open(os.path.join(directory, 'learning_prop.json'), 'w'))

    env = make_atari(args.env)
    env = bench.Monitor(env, logger.get_dir())
    env = models.wrap_atari_dqn(env)

    if args.record == 1:
        env = Monitor(env, directory=args.log_dir)
    
    with tf.device(args.device):
        model = models.cnn_to_mlp(
            convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
            hiddens=[256],
            dueling=bool(args.dueling),
            init_mean = args.init_mean, 
            init_sd = args.init_sd,
        )

        simple.learn(
            env,
            q_func=model,
            lr=args.learning_rate,
            max_timesteps=args.num_timesteps,
            buffer_size=args.buffer_size,
            exploration_fraction=args.eps_max,
            exploration_final_eps=args.eps_min,
            train_freq=4,
            print_freq=1000,
            checkpoint_freq=args.epoch_steps,
            learning_starts=args.nb_steps_warmup,
            target_network_update_freq=args.target_update_freq,
            gamma=0.99,
            prioritized_replay=bool(args.prioritized),
            prioritized_replay_alpha=args.prioritized_replay_alpha,
            env_name = args.env,
            epoch_steps = args.epoch_steps,
            alg = args.alg,
            noise = args.noise,
            gpu_memory = args.gpu_memory,
            varTH=args.varth,
            act_policy=args.act_policy,
            directory=directory
        )

    env.close()


if __name__ == '__main__':
    main()
