import datetime, json, os, argparse, time
import pickle, tabulate
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from baselines0.common import set_global_seeds

class Test:
    def __init__(self):
        pass

    def test(self, args, env, act):
        seed = args.seed
        env.seed(seed)
        set_global_seeds(seed)

        timelimit_env = env
        while( not hasattr(timelimit_env, '_elapsed_steps')):
            timelimit_env = timelimit_env.env

        if args.ros_log:
            from envs.target_tracking.ros_wrapper import RosLog
            ros_log = RosLog(num_targets=args.nb_targets, wrapped_num=args.ros + args.render + args.record + 1)

        ep = 0
        init_pos = []
        ep_nlogdetcov = ['Episode nLogDetCov']
        time_elapsed = ['Elapsed Time (sec)']
        graph_nlogdetcov = []
        given_init_pose, test_init_pose = [], []
        # Use a fixed set of initial positions if given.
        if args.init_file_path != '.':
            given_init_pose = pickle.load(open(args.init_file_path, "rb"))

        while(ep < args.nb_test_steps): # test episode
            ep += 1
            episode_rew, nlogdetcov = 0, 0
            done = {}
            obs = env.reset(init_pose_list=given_init_pose)
            test_init_pose.append({'agents':[timelimit_env.env.agents[i].state for i in range(args.nb_agents)],
                                'targets':[timelimit_env.env.targets[i].state for i in range(args.nb_targets)],
                                'belief_targets':[timelimit_env.env.belief_targets[i].state for i in range(args.nb_targets)]})
            s_time = time.time()

            action_dict = {}
            while type(done) is dict:
                if args.render:
                    env.render()
                if args.ros_log:
                    ros_log.log(env)
                for agent_id, a_obs in obs.items():
                    action_dict[agent_id] = act(np.array(a_obs)[None])[0]
                obs, rew, done, info = env.step(action_dict)
                episode_rew += rew['__all__']
                nlogdetcov += info['mean_nlogdetcov']

            time_elapsed.append(time.time() - s_time)
            ep_nlogdetcov.append(nlogdetcov)
            graph_nlogdetcov.append(nlogdetcov)
            if ep % 100 == 0:
                print("Ep.%d - Episode reward : %.2f, Episode nLogDetCov : %.2f"%(ep, episode_rew, nlogdetcov))

        if args.record :
            env.moviewriter.finish()
        if args.ros_log :
            ros_log.save(args.log_dir)

        # Eval plots and saves
        eval_dir = os.path.join(args.log_dir, 'eval_seed%d'%(seed))
        if not os.path.exists(eval_dir):
            os.makedirs(eval_dir)

        matplotlib.use('Agg')
        f0, ax0 = plt.subplots()
        _ = ax0.plot(graph_nlogdetcov, '.')
        _ = ax0.set_xlabel('episode number')
        _ = ax0.set_ylabel('mean_nlogdetcov')
        _ = ax0.grid()
        _ = f0.savefig(os.path.join(eval_dir, "%da%dt_%d_eval_eps.png"%
                                            (args.nb_agents, args.nb_targets, args.nb_test_steps)))
        plt.close()
        pickle.dump(graph_nlogdetcov, open(os.path.join(eval_dir,'%da%dt_%d_eval_eps.pkl'%
                                            (args.nb_agents, args.nb_targets, args.nb_test_steps)), 'wb'))
        
        pickle.dump(test_init_pose, open(os.path.join(args.log_dir,'test_init_pose.pkl'), 'wb'))
        f_result = open(os.path.join(args.log_dir, 'test_result.txt'), 'w')
        f_result.write(tabulate.tabulate([ep_nlogdetcov, time_elapsed], tablefmt='presto'))
        f_result.close()