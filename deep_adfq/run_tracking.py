"""
This code was modified from a OpenAI baseline code - baselines0/baselines0/deepq/experiments/train_cartpole.py for running ADFQ
"""
from baselines0.common import set_global_seeds

import models
import numpy as np
import tensorflow as tf
import datetime, json, os, argparse, time, pickle

import deepadfq
from logger import Logger, batch_plot
import envs

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--env', help='environment ID', default='TargetTracking-v1')
parser.add_argument('--seed', help='RNG seed', type=int, default=0)
parser.add_argument('--prioritized', type=int, default=1)
parser.add_argument('--prioritized-replay-alpha', type=float, default=0.6)
parser.add_argument('--mode', choices=['train', 'test'], default='train')
parser.add_argument('--dueling', type=int, default=0)
parser.add_argument('--nb_train_steps', type=int, default=10000)
parser.add_argument('--buffer_size', type=int, default=5000)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--nb_warmup_steps', type=int, default = 500)
parser.add_argument('--nb_epoch_steps', type=int, default = 500)
parser.add_argument('--checkpoint_freq', type=int, default=100)
parser.add_argument('--target_update_freq', type=float, default=500) # This should be smaller than epoch_steps
parser.add_argument('--nb_test_steps',type=int, default = None)
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--learning_rate_decay_factor', type=float, default=1.0)
parser.add_argument('--learning_rate_growth_factor', type=float, default=1.0)
parser.add_argument('--gamma', type=float, default=.99)
parser.add_argument('--hiddens', type=str, default='64:128:64')
parser.add_argument('--log_dir', type=str, default='.')
parser.add_argument('--log_fname', type=str, default='model.pkl')
parser.add_argument('--eps_fraction', type=float, default=0.1)
parser.add_argument('--eps_min', type=float, default=.02)
parser.add_argument('--test_eps', type=float, default=.05)
parser.add_argument('--init_mean', type =float, default=1.)
parser.add_argument('--init_sd', type=float, default=10.)
parser.add_argument('--device', type=str, default='/cpu:0')
parser.add_argument('--gpu_memory',type=float, default=1.0)
parser.add_argument('--alg', choices=['adfq','adfq-v2'], default='adfq')
parser.add_argument('--act_policy', choices=['egreedy','bayesian'], default='bayesian') # egreedy or thompson sampling (bayesian)
parser.add_argument('--record',type=int, default=0)
parser.add_argument('--render', type=int, default=0)
parser.add_argument('--varth', type=float,default=1e-10)
parser.add_argument('--noise', type=float,default=1.0)
parser.add_argument('--repeat', type=int, default=1)
parser.add_argument('--scope',type=str, default='deepadfq')
parser.add_argument('--ros', type=int, default=0)
parser.add_argument('--ros_log', type=int, default=0)
parser.add_argument('--map', type=str, default="emptySmall")
parser.add_argument('--nb_targets', type=int, default=1)
parser.add_argument('--im_size', type=int, default=50)
parser.add_argument('--eval_type', choices=['random', 'random_zone', 'fixed'], default='random')
parser.add_argument('--init_file_path', type=str, default=".")

args = parser.parse_args()

def train(seed, save_dir):
    set_global_seeds(seed)
    save_dir_0 = os.path.join(save_dir, 'seed_%d'%seed)
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
        with tf.compat.v1.variable_scope('seed_%d'%seed):
            hiddens = args.hiddens.split(':')
            hiddens = [int(h) for h in hiddens]
            if args.env == 'TargetTracking-v5':
                model = models.cnn_plus_mlp(
                                convs=[(4, 8, 4), (8, 4, 2)],
                                hiddens= hiddens,
                                dueling=bool(args.dueling),
                                init_mean = args.init_mean,
                                init_sd = args.init_sd,
                                inpt_dim = (args.im_size, args.im_size),
                )
            else:
                model = models.mlp(hiddens, init_mean=args.init_mean, init_sd=args.init_sd)
            act = deepadfq.learn(
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
                checkpoint_freq=args.checkpoint_freq,
                learning_starts=args.nb_warmup_steps,
                gamma=args.gamma,
                prioritized_replay=bool(args.prioritized),
                prioritized_replay_alpha=args.prioritized_replay_alpha,
                callback=None,#callback,
                alg=args.alg,
                scope=args.scope,
                sdMin=np.sqrt(args.varth),
                noise=args.noise,
                act_policy=args.act_policy,
                epoch_steps=args.nb_epoch_steps,
                eval_logger=Logger(args.env,
                                env_type='target_tracking',
                                save_dir=save_dir_0,
                                render=bool(args.render),
                                figID=1,
                                ros=bool(args.ros),
                                map_name=args.map,
                                num_targets=args.nb_targets,
                                im_size=args.im_size,
                                eval_type=args.eval_type,
                                init_file_path=args.init_file_path,
                                ),
                save_dir=save_dir_0,
                test_eps=args.test_eps,
                gpu_memory=args.gpu_memory,
                render=(bool(args.render) or bool(args.ros)),
            )
            print("Saving model to model.pkl")
            act.save(os.path.join(save_dir_0, "model.pkl"))
    if args.record == 1:
        env.moviewriter.finish()

def test():
    learning_prop = json.load(open(os.path.join(args.log_dir, '../learning_prop.json'),'r'))
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
    act_params = {'scope': "seed_%d"%learning_prop['seed']+"/"+learning_prop['scope'], 'eps': args.test_eps}
    act = deepadfq.load(os.path.join(args.log_dir, args.log_fname), act_params)

    if args.ros_log:
        from envs.target_tracking.ros_wrapper import RosLog
        ros_log = RosLog(num_targets=args.nb_targets, wrapped_num=args.ros + args.render + args.record + 1)

    ep = 0
    ep_nlogdetcov = ['Episode nLogDetCov']
    time_elapsed = ['Elapsed Time (sec)']
    given_init_pose, test_init_pose = [], []
    # Use a fixed set of initial positions if given.
    if args.init_file_path != '.':
        import pickle
        given_init_pose = pickle.load(open(args.init_file_path, "rb"))

    while(ep < args.nb_test_steps): # test episode
        ep += 1
        episode_rew, nlogdetcov = 0, 0
        obs, done = env.reset(init_pose_list=given_init_pose), False
        test_init_pose.append({'agent':timelimit_env.env.agent.state,
                            'targets':[timelimit_env.env.targets[i].state for i in range(args.nb_targets)],
                            'belief_targets':[timelimit_env.env.belief_targets[i].state for i in range(args.nb_targets)]})
        s_time = time.time()
        while not done:
            if args.render:
                env.render()
            if args.ros_log:
                ros_log.log(env)
            obs, rew, done, info = env.step(act(obs[None])[0])
            episode_rew += rew
            nlogdetcov += info['mean_nlogdetcov']

        time_elapsed.append(time.time() - s_time)
        ep_nlogdetcov.append(nlogdetcov)
        print("Ep.%d - Episode reward : %.2f, Episode nLogDetCov : %.2f"%(ep, episode_rew, nlogdetcov))

    if args.record :
        env.moviewriter.finish()
    if args.ros_log :
        ros_log.save(args.log_dir)

    import pickle, tabulate
    pickle.dump(test_init_pose, open(os.path.join(args.log_dir,'test_init_pose.pkl'), 'wb'))
    f_result = open(os.path.join(args.log_dir, 'test_result.txt'), 'w')
    f_result.write(tabulate.tabulate([ep_nlogdetcov, time_elapsed], tablefmt='presto'))
    f_result.close()

if __name__ == '__main__':
    if args.mode == 'train':
        save_dir = os.path.join(args.log_dir, '_'.join([args.env, datetime.datetime.now().strftime("%m%d%H%M")]))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        else:
            ValueError("The directory already exists...", save_dir)
        json.dump(vars(args), open(os.path.join(save_dir, 'learning_prop.json'), 'w'))

        notes = input("Any notes for this experiment? : ")
        f = open(os.path.join(save_dir, "notes.txt"), 'w')
        f.write(notes)
        f.close()

        seed = args.seed
        list_records = []
        for _ in range(args.repeat):
            print("===== TRAIN A TARGET TRACKING RL AGENT : SEED %d ====="%seed)
            train(seed, save_dir)
            list_records.append(pickle.load(open(os.path.join(save_dir, "seed_%d"%seed, "records.pkl"), "rb")))
            seed += 1

        batch_plot(list_records, save_dir, args.nb_train_steps,
            args.nb_epoch_steps, is_target_tracking=True)

    elif args.mode =='test':
        test()
