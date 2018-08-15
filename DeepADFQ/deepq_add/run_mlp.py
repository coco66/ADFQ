"""
This code was slightly modified from the baselines/baselines/deepq/train_cartpole.py in order to use 
a different evaluation method. In order to run, simply replace the original code with this code 
in the original directory.
"""
import gym
import argparse
import tensorflow as tf
import datetime, json, os

from baselines.common import set_global_seeds

from baselines import deepq
from gym.wrappers import Monitor
import numpy as np
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--env', help='environment ID', default='CartPole-v0')
parser.add_argument('--seed', help='RNG seed', type=int, default=0)
parser.add_argument('--prioritized', type=int, default=0)
parser.add_argument('--prioritized-replay-alpha', type=float, default=0.6)
parser.add_argument('--double_q', type=int, default=0)
parser.add_argument('--mode', choices=['train', 'test'], default='train')
parser.add_argument('--dueling', type=int, default=0)
parser.add_argument('--nb_train_steps', type=int, default=100000)
parser.add_argument('--buffer_size', type=int, default=50000)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--nb_warmup_steps', type=int, default = 1000)
parser.add_argument('--nb_epoch_steps', type=int, default = 1000)
parser.add_argument('--target_update_freq', type=int, default=500) # This should be smaller than epoch_steps
parser.add_argument('--nb_test_steps',type=int, default = None)
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--gamma', type=float, default=.99)
parser.add_argument('--log_dir', type=str, default='.')
parser.add_argument('--log_fname', type=str, default='model.pkl')
parser.add_argument('--eps_fraction', type=float, default=0.1)
parser.add_argument('--eps_min', type=float, default=.02)
parser.add_argument('--device', type=str, default='/gpu:0')
parser.add_argument('--record',type=int, default=0)
parser.add_argument('--gpu_memory',type=float, default=1.0)
parser.add_argument('--repeat', type=int, default=1)

args = parser.parse_args()

def callback(lcl, _glb):
    # stop training if reward exceeds 199
    is_solved = lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= 199
    return is_solved


def train():
    set_global_seeds(args.seed)
    directory = os.path.join(args.log_dir, '_'.join([args.env, datetime.datetime.now().strftime("%m%d%H%M")]))
    if not os.path.exists(directory):
            os.makedirs(directory)
    else:
            ValueError("The directory already exists...", directory)
    json.dump(vars(args), open(os.path.join(directory, 'learning_prop.json'), 'w'))

    env = gym.make(args.env)

    with tf.device(args.device):
        model = deepq.models.mlp([64])
        act, records = deepq.learn(
            env,
            q_func=model,
            lr=args.learning_rate,
            max_timesteps=args.nb_train_steps,
            buffer_size=args.buffer_size,
            exploration_fraction=args.eps_fraction,
            exploration_final_eps=args.eps_min,
            print_freq=10,
            checkpoint_freq=int(args.nb_train_steps/10),
            learning_starts=args.nb_warmup_steps,
            gamma = args.gamma,
            callback=None,#callback,
            epoch_steps = args.nb_epoch_steps,
            gpu_memory=args.gpu_memory,
            directory=directory,
            double_q = args.double_q,
            nb_test_steps=args.nb_test_steps,
        )
        print("Saving model to model.pkl")
        act.save(os.path.join(directory,"model.pkl"))
    plot(records, directory)
        
def test():
    env = gym.make(args.env)
    act = deepq.load(os.path.join(args.log_dir, args.log_fname))
    if args.record:
        env = Monitor(env, directory=args.log_dir)
    while True:
        obs, done = env.reset(), False
        episode_rew = 0
        while not done:
            if not(args.record):
                env.render()
            obs, rew, done, _ = env.step(act(obs[None])[0])
            episode_rew += rew
        print("Episode reward", episode_rew)

def plot(records, directory):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    
    m = len(records['loss'])
    x_vals = range(0 , args.nb_epoch_steps*m, args.nb_epoch_steps)
    
    f0, ax0 = plt.subplots()
    ax0.plot(x_vals, records['loss'])
    ax0.set_ylabel('Loss')
    ax0.set_xlabel('Learning Steps')

    f1, ax1 = plt.subplots()
    ax1.plot(x_vals, records['online_reward'])
    ax1.set_ylabel('Average recent 100 rewards')
    ax1.set_xlabel('Learning Steps')

    f2, ax2 = plt.subplots()
    m, ids25, ids75 = iqr(np.array(records['test_reward']).T)
    ax2.plot(x_vals, m, color='b')
    ax2.fill_between(x_vals, list(ids75), list(ids25), facecolor='b', alpha=0.2)
    ax2.set_ylabel('Test Rewards')
    ax2.set_xlabel('Learning Steps')

    f0.savefig(os.path.join(directory, "result.png"))
    f1.savefig(os.path.join(directory, "online_reward.png"))
    f2.savefig(os.path.join(directory, "test_reward.png"))

def iqr(x):
    """Interquantiles
    x has to be a 2D np array. The interquantiles are computed along with the axis 1
    """
    i25 = int(0.25*x.shape[0])
    i75 = int(0.75*x.shape[0])
    x=x.T
    ids25=[]
    ids75=[]
    m = []
    for y in x:
        tmp = np.sort(y)
        ids25.append(tmp[i25])
        ids75.append(tmp[i75])
        m.append(np.mean(tmp,dtype=np.float32))
    return m, ids25, ids75

if __name__ == '__main__':
    if args.mode == 'train':
        i = 0
        while(i < args.repeat):
            train()
            i += 1
    elif args.mode =='test':
        test()

