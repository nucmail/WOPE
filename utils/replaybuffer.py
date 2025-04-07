import d4rl.infos
import numpy as np
import torch
import pickle
import d4rl

class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6),device="cuda:0"):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))
        self.device = device

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device),
            torch.FloatTensor(self.labels[ind]).to(self.device)
        )

    def convert_D4RL(self, env_name, dataset, traj_lens, traj_returns, lamda=0.5, normalize_state=True, eps=5e-4, label_type=1):
        self.state = dataset['observations']
        self.action = dataset['actions']
        self.next_state = dataset['next_observations']
        self.reward = dataset['rewards'].reshape(-1,1)
        self.not_done = 1. - dataset['terminals'].reshape(-1,1)
        self.size = self.state.shape[0]
        self.labels=np.zeros(self.size)

        current_index=0

        if label_type==1:
            # if "walker" in env_name:
            #     max_value,min_value=4600.0,1.0
            # elif "half" in env_name:
            #     max_value,min_value=12500.0,-300
            # elif "hopper" in env_name:
            #     max_value,min_value=3300.0,-20.272305
            # elif "ant" in env_name:
            #     max_value=d4rl.infos.REF_MAX_SCORE[env_name]
            max_value=d4rl.infos.REF_MAX_SCORE[env_name]
            min_value=d4rl.infos.REF_MIN_SCORE[env_name]

            for lens, returns in zip(traj_lens, traj_returns):
                normalized_returns=(returns-min_value)/(max_value-min_value)
                self.labels[current_index:current_index+lens]=normalized_returns
                current_index+=lens
            self.labels = 1. - self.labels* lamda - eps
            self.labels=torch.from_numpy(self.labels).to(dtype=torch.float32)
        else:
            traj_labels=np.load(f"/home/ubuntu/cwy_file/0_Data/single_agent/D4RL/label/{env_name}.npy")
            current_index=0
            for traj_len,label in zip(traj_lens,traj_labels):
                if label=="random":
                    self.labels[current_index:current_index+traj_len]=0.
                elif label=="medium":
                    self.labels[current_index:current_index+traj_len]=1.
                elif label=="expert":
                    self.labels[current_index:current_index+traj_len]=2.
                current_index+=traj_len
            self.labels=0.99-self.labels*0.22
            self.labels=torch.from_numpy(self.labels).to(dtype=torch.float32)

        if normalize_state:
            mean,std=self.normalize_states()
        else:
            mean,std=0,1
        return mean,std
    
    def normalize_states(self, eps = 1e-3):
        mean = self.state.mean(0,keepdims=True)
        std = self.state.std(0,keepdims=True) + eps
        self.state = (self.state - mean)/std
        self.next_state = (self.next_state - mean)/std
        return mean, std
    