import gym
import models
import simple
import numpy as np
import tensorflow as tf
import argparse


def callback(lcl, _glb):
    # stop training if reward exceeds 199
    is_solved = lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= 199
    return is_solved


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='CartPole-v0')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--prioritized', type=int, default=1)
    parser.add_argument('--prioritized-replay-alpha', type=float, default=0.6)
    
    parser.add_argument('--dueling', type=int, default=0)
    parser.add_argument('--num_timesteps', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--nb_steps_warmup', type=int, default = 10000)
    parser.add_argument('--epoch_steps', type=int, default = 200)
    parser.add_argument('--target_update_freq', type=int, default=1000)
    parser.add_argument('--learning_rate', type=float, default=5*1e-4)
    parser.add_argument('--gamma', type=float, default=.99)
    parser.add_argument('--log_dir', type=str, default='.')
    parser.add_argument('--buffer_size', type=int, default=50000)
    parser.add_argument('--eps_max', type=float, default=0.1)
    parser.add_argument('--eps_min', type=float, default=.02)
    parser.add_argument('--init_mean', type =float, default=1.)
    parser.add_argument('--init_sd', type=float, default=30.)
    parser.add_argument('--device', type=str, default='/gpu:0')
    parser.add_argument('--alg', choices=['softapprox','hardapprox'], default='softapprox')
    parser.add_argument('--act_policy', choices=['egreedy','bayesian'], default='egreedy')
    parser.add_argument('--record',type=int, default=0)
    parser.add_argument('--gpu_memory',type=float, default=1.0)
    parser.add_argument('--varth', type=float,default=1e-5)
    parser.add_argument('--noise', type=float,default=1e-5)


    args = parser.parse_args()
    tf.set_random_seed(0)
    np.random.seed(0)

    env = gym.make(args.env)

    model = models.mlp([64], init_mean=args.init_mean, init_sd=args.init_sd)

    act = simple.learn(
        env,
        q_func=model,
        lr=args.learning_rate,
        max_timesteps=args.num_timesteps,
        buffer_size=args.buffer_size,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        target_network_update_freq=500,
        print_freq=10,
        callback=None,#callback,
        env_name=args.env,
        epoch_steps = args.epoch_steps,
        noise = args.noise,
        varTH=args.varth,
        alg = args.alg,
        gpu_memory=args.gpu_memory,
        act_policy=args.act_policy,
    )
    print("Saving model to cartpole_model.pkl")
    act.save("cartpole_model.pkl")


def plot():
    import matplotlib.pyplot as plt
    data = np.load('temp.npy')




    mean = data[()]['mean']
    std = data[()]['std']
    r = data[()]['reward']
    loss = data[()]['loss']

    #means = np.mean(mean, axis=1)
    #std = np.mean(std, axis=1)

    plt.figure(1)
    plt.plot(std)
    plt.ylim(0.0, 10.)
    plt.xlabel('iterations')
    plt.ylabel('std')


    plt.figure(2)
    plt.plot(r)
    plt.xlabel('iterations')
    plt.ylabel('avg reward')

    plt.figure(3)
    plt.plot(loss)
    plt.xlabel('iterations')
    plt.ylabel('loss')

    plt.show()

if __name__ == '__main__':
    main()
    plot()

