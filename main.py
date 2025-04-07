import argparse
import gym
import numpy as np
import os
import sys
import torch
import json
import d4rl
import setproctitle
import logging
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from utils.logger import Log
from utils.replaybuffer import ReplayBuffer
from utils.dataset import get_dataset
# from TD3_BC import TD3_BC as Agent
from wope import WOPE as Agent

def setup_logger(exp):
    logger = logging.getLogger(exp)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # Clear existing handlers
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(message)s"))
    logger.addHandler(handler)
    logger.info(f"Logger initialized for experiment {exp}.")
    return logger


def eval_policy(policy, env_name, seed, mean, std, logger, eval_episodes=10):
    scores = []
    for eposide in range(eval_episodes):
        eval_env = gym.make(env_name)
        eval_env.seed(seed + 100 + eposide)
        traj_return = 0.
        state, done = eval_env.reset(), False
        while not done:
            state = (np.array(state).reshape(1,-1) - mean)/std
            action = policy.sample_action(np.array(state))
            state, reward, done, _ = eval_env.step(action)
            traj_return += reward
        scores.append(traj_return)
        eval_env.close()
    avg_reward = np.mean(scores)
    std_reward = np.std(scores)
    normalized_scores = [eval_env.get_normalized_score(s) for s in scores]
    avg_norm_score = eval_env.get_normalized_score(avg_reward)
    std_norm_score = np.std(normalized_scores)
    logger.info(f"Evaluation over {eval_episodes} episodes | Average Score is: {avg_reward:.2f} | Normalized Average Score is :{avg_norm_score:.2f} ")
    return avg_reward, std_reward, avg_norm_score, std_norm_score


def main(args):
    logger=setup_logger(args.exp)
    setproctitle.setproctitle(args.exp)
    save_path=Path(args.log_dir)/args.env_name/args.exp
    log = Log(Path(args.log_dir)/args.env_name, vars(args), args.exp)
    logger.info(f"Save Information on {save_path}")

    env = gym.make(args.env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] 
    max_action = float(env.action_space.high[0])
    logger.info(f"Test Algorithm on {args.env_name} | state_dim is {state_dim} | action dim is {action_dim} |")
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    replay_buffer = ReplayBuffer(state_dim, action_dim, device=args.device)
    offline_dataset, traj_lens, traj_returns = get_dataset(args.env_name, logger)
    mean,std= replay_buffer.convert_D4RL(args.env_name, offline_dataset, traj_lens, traj_returns, args.lamda, args.normalize_state, label_type=args.label_type)
    agent = Agent(state_dim=state_dim, 
                action_dim=action_dim, 
                max_action=max_action, 
                device=args.device, 
                discount=args.discount, 
                tau=args.tau, 
                cql=args.use_cql, 
                cql_alpha=args.cql_alpha,
                omar=args.use_omar, 
                num_gaussian=args.num_gaussian, )
    #agent = Agent(state_dim=state_dim,  action_dim=action_dim, max_action=max_action, )

    evaluations = []
    training_iters = 0
    max_timesteps = args.train_epochs * args.num_steps_per_epoch
    while (training_iters < max_timesteps) :
        iterations = int(args.eval_freq * args.num_steps_per_epoch)
        loss_metric = agent.train(replay_buffer,
                                  iterations=iterations,
                                  batch_size=args.batch_size)
        training_iters += iterations
        curr_epoch = int(training_iters // int(args.num_steps_per_epoch))
        eval_res, eval_res_std, eval_norm_res, eval_norm_res_std = eval_policy(agent, args.env_name, args.seed,mean,std, logger=logger, eval_episodes=args.eval_episodes)
        evaluations.append([eval_res, eval_res_std, eval_norm_res, eval_norm_res_std, curr_epoch])
        log.row({
                    'return mean': eval_res,
                    'return std': eval_res_std,
                    'normalized return mean': eval_norm_res,
                    'normalized return std': eval_norm_res_std,
                    'mean_critic_loss':np.mean(loss_metric["critic_loss"]),
                    'mean_bc_loss':np.mean(loss_metric["bc_loss"]),
                    'mean_q_loss':np.mean(loss_metric["ql_loss"]),
                    'mean_actor_loss':np.mean(loss_metric["actor_loss"]),
                })
        
    np.save(os.path.join(save_path, "eval"), evaluations)
    torch.save(agent.actor.state_dict(), save_path/'final_actor.pt')
    torch.save(agent.critic.state_dict(), save_path/'final_critic.pt')
    log.close()

    scores = np.array(evaluations)
    best_id = np.argmax(scores[:, 2])
    best_res = {'epoch': scores[best_id, -1],
                'best normalized score avg': scores[best_id, 2],
                'best normalized score std': scores[best_id, 3],
                'best raw score avg': scores[best_id, 0],
                'best raw score std': scores[best_id, 1]}
    with open(os.path.join(save_path, f"best_score.txt"), 'w') as f:
        f.write(json.dumps(best_res))
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ### Experimental Setups ###
    parser.add_argument("--exp", default='Paper1', type=str)                             
    parser.add_argument("--env_name", default="halfcheetah-medium-expert-v2", type=str)  
    parser.add_argument('--device', default="cuda:0", type=str)     
    parser.add_argument("--log_dir", default="results", type=str)                    
    parser.add_argument("--seed", default=0, type=int)                        
    
    
    ### RL Setup ###
    parser.add_argument("--discount", default=0.99, type=float)
    parser.add_argument("--tau", default=0.005, type=float)
    parser.add_argument("--num_steps_per_epoch", default=1000, type=int)
    parser.add_argument("--train_epochs", default=1000, type=int)
    parser.add_argument("--eval_freq", default=25, type=int)
    parser.add_argument("--eval_episodes", default=10, type=int)

    ### Optimization Setups ###
    parser.add_argument("--lr", default='1e-4', type=float)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument('--normalize_state', default=True, type=bool)

    ### WOPE Parameters ###
    parser.add_argument('--lamda', default=0.3, type=float)
    parser.add_argument('--use_cql', default=True, type=bool)
    parser.add_argument('--cql_alpha', default=1.0, type=float)
    parser.add_argument('--use_omar', default=True, type=bool)
    parser.add_argument("--num_gaussian" ,type=int, default=1)
    

    parser.add_argument("--label_type",type=int, default=1)
    main(parser.parse_args())
    