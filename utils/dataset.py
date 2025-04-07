import gym
import numpy as np
import collections
import pickle
import d4rl

def get_dataset(env_name, logger, save_trajectories=False):
    env = gym.make(env_name)
    dataset = env.get_dataset()
    N = dataset['rewards'].shape[0]
    data_ = collections.defaultdict(list)

    max_timesteps= env._max_episode_steps

    use_timeouts = False
    if 'timeouts' in dataset:
        use_timeouts = True

    episode_step = 0
    paths = []
    for i in range(N):
        done_bool = bool(dataset['terminals'][i])
        if use_timeouts:
            final_timestep = dataset['timeouts'][i]
        else:
            final_timestep = (episode_step == max_timesteps-1)
        if "next_observations" in dataset.keys():
            for k in ["observations", "actions", "rewards", "terminals", "timeouts"]:
                data_[k].append(dataset[k][i])
        else:
            for k in ["observations", "actions", "rewards", "terminals", "timeouts"]:
                data_[k].append(dataset[k][i])

            if done_bool or final_timestep:
                episode_step = 0
                episode_data = {}
                for k in data_:
                    episode_data[k] = np.array(data_[k])
                paths.append(episode_data)
                data_ = collections.defaultdict(list)
            episode_step += 1

    returns = np.array([np.sum(p['rewards']) for p in paths])
    num_samples = np.sum([p['rewards'].shape[0] for p in paths])
    logger.info(f'Number of samples collected: {num_samples}')
    logger.info(f'Trajectory returns: mean = {np.mean(returns)}, std = {np.std(returns)}, max = {np.max(returns)}, min = {np.min(returns)}')

    states, actions, next_states,rewards, dones, traj_lens, traj_returns =  [], [], [], [], [], [], []
    for path in paths:
        states.append(path['observations'][:-2,:])
        actions.append(path["actions"][:-2,:])
        next_states.append(path["observations"][1:,:])
        rewards.append(path["rewards"][:-2,])
        dones.append(path["terminals"][:-2,])
        traj_lens.append(len(path['observations'][:-2,]))
        traj_returns.append(path['rewards'].sum())
    traj_lens = np.array(traj_lens)
    dataset={
        "observations":np.concatenate(states,axis=0),
        "actions":np.concatenate(actions,axis=0),
        "next_observations":np.concatenate(next_states,axis=0),
        "rewards":np.concatenate(rewards,axis=0).reshape(-1,1),
        "terminals":np.concatenate(dones,axis=0).reshape(-1,1),
    }

    if save_trajectories:
        with open(f'./data/{env_name}.pkl', 'wb') as f:
            pickle.dump(paths, f)
        logger.info(f'Trajectories saved at ./data/{env_name}.pkl')
    return dataset, traj_lens, traj_returns
