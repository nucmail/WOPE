import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.actor_model=nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )
        
        self.max_action = max_action
        

    def forward(self, state):
        return self.max_action * self.actor_model(state)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.q1_model = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, 1))

        self.q2_model = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, 1))

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1_model(x), self.q2_model(x)

    def q1(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1_model(x)

    def q_min(self, state, action):
        q1, q2 = self.forward(state, action)
        return torch.min(q1, q2)


class WOPE(object):
    def __init__(self,
                 state_dim,
                 action_dim,
                 max_action,
                 device,
                 discount,
                 tau,
                 cql,
                 cql_alpha,
                 omar,
                 num_gaussian,
                 lr=3e-4,
                 ):
        self.step=0

        self.actor=Actor(state_dim, action_dim, max_action).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.target_actor=copy.deepcopy(self.actor)
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.state_dim = state_dim
        self.max_action = max_action
        self.action_dim = action_dim

        self.discount = discount
        self.tau = tau
        self.device = device
        self.num_gaussian=num_gaussian

        self.cql=cql
        self.cql_alpha=cql_alpha
        self.omar=omar

        self.actor_lr_scheduler = CosineAnnealingLR(self.actor_optimizer, T_max=int(2e6), eta_min=0.)
        self.critic_lr_scheduler = CosineAnnealingLR(self.critic_optimizer, T_max=int(2e6), eta_min=0.)

    def calc_gaussian_pdf(self, samples, mu=0):
        pdfs = 1 / (0.2 * np.sqrt(2 * np.pi)) * torch.exp( - (samples - mu)**2 / (2 * 0.2**2) )
        pdf = torch.prod(pdfs, dim=-1)
        return pdf

    def get_policy_actions(self, state, network):
        action = network(state)
        formatted_action = action.unsqueeze(1).repeat(1, 10, 1).view(action.shape[0] * 10, action.shape[1])
        random_noises = torch.FloatTensor(formatted_action.shape[0], formatted_action.shape[1])
        random_noises = random_noises.normal_() * 0.3
        random_noises_log_pi = self.calc_gaussian_pdf(random_noises).view(action.shape[0], 10, 1).cuda()
        random_noises = random_noises.cuda()
        noisy_action = (formatted_action + random_noises).clamp(-self.max_action, self.max_action)
        return noisy_action, random_noises_log_pi

    def train(self, replay_buffer, iterations, batch_size=100, log_writer=None):
        
        metric = {'bc_loss': [], 'ql_loss': [], 'actor_loss': [], 'critic_loss': []}
        for _ in range(iterations):
            # Sample replay buffer / batch
            state, action, next_state, reward, not_done, labels = replay_buffer.sample(batch_size)
            next_action = self.target_actor(next_state)
            target_q = self.critic_target.q_min(next_state, next_action)
            target_q = (reward + not_done * self.discount * target_q).detach()
            current_q1, current_q2 = self.critic(state, action)
            critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
            if self.cql:
                formatted_obs = state.unsqueeze(1).repeat(1, 10, 1).view(-1, state.shape[1])
                random_action = (torch.FloatTensor(action.shape[0] * 10, action.shape[1]).uniform_(-1, 1)).cuda()
                random_action_log_pi = np.log(0.5 ** random_action.shape[-1])
                curr_action, curr_action_log_pi = self.get_policy_actions(state, self.actor)
                new_curr_action, new_curr_action_log_pi = self.get_policy_actions(next_state, self.actor)
                random_Q1, random_Q2 = self.critic(formatted_obs, random_action)
                curr_Q1, curr_Q2 = self.critic(formatted_obs, curr_action)
                new_curr_Q1, new_curr_Q2 = self.critic(formatted_obs, new_curr_action)
                random_Q1, random_Q2 = random_Q1.view(state.shape[0], 10, 1), random_Q2.view(state.shape[0], 10, 1)
                curr_Q1, curr_Q2 = curr_Q1.view(state.shape[0], 10, 1), curr_Q2.view(state.shape[0], 10, 1)
                new_curr_Q1, new_curr_Q2 = new_curr_Q1.view(state.shape[0], 10, 1), new_curr_Q2.view(state.shape[0], 10, 1)
                cat_q1 = torch.cat([random_Q1 - random_action_log_pi, new_curr_Q1 - new_curr_action_log_pi, curr_Q1 - curr_action_log_pi], 1)
                cat_q2 = torch.cat([random_Q2 - random_action_log_pi, new_curr_Q2 - new_curr_action_log_pi, curr_Q2 - curr_action_log_pi], 1)
                policy_qvals1 = torch.logsumexp(cat_q1 / 1.0, dim=1) * 1.0
                policy_qvals2 = torch.logsumexp(cat_q2 / 1.0, dim=1) * 1.0

                dataset_q_vals1 = current_q1
                dataset_q_vals2 = current_q2
                cql_term1,cql_term2 = policy_qvals1 - dataset_q_vals1, policy_qvals2 - dataset_q_vals2
                cql_term = (cql_term1 + cql_term2)*self.cql_alpha
                critic_loss+=cql_term.mean()

            #if self.omar:
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm(self.critic.parameters(), 0.5)
            self.critic_optimizer.step()

            """ Policy Training """
            new_action = self.actor(state)
            q1_new_action=self.critic.q1(state,new_action)
            q_loss=q1_new_action.mean()
            if self.omar:
                if self.num_gaussian==1:
                    init_mean=[0.]
                    init_std=1.
                    num_sample_action=10
                    num_select=4
                elif self.num_gaussian==2:
                    init_mean=[-0.4,0.4]
                    init_std=0.8
                    num_sample_action=6
                    num_select=3
                else:
                    init_mean=[-0.8,0,0.8]
                    init_std=0.6
                    num_sample_action=4
                    num_select=2

                self.omar_coe=labels.to(dtype=torch.float32, device=self.device).view(-1,1)
                formatted_obs = state.unsqueeze(1).repeat(1, num_sample_action, 1).view(-1, state.shape[1])
                last_top_k_qvals, last_elite_acs = None, None
                for dist_idx in range(self.num_gaussian):
                    self.omar_mu=torch.cuda.FloatTensor(action.shape[0], action.shape[1]).zero_() + init_mean[dist_idx]
                    self.omar_sigma = torch.cuda.FloatTensor(action.shape[0], action.shape[1]).zero_() + init_std
                    for iter_idx in range(2):
                        self.omar_sigma=torch.clamp(self.omar_sigma,min=0.000001)
                        dist = torch.distributions.Normal(self.omar_mu, self.omar_sigma)
                        cem_sampled_acs = dist.sample((num_sample_action,)).detach().permute(1, 0, 2).clamp(-self.max_action, self.max_action)
                        formatted_cem_sampled_acs = cem_sampled_acs.reshape(-1, cem_sampled_acs.shape[-1])
                        all_pred_qvals = self.critic.q1(formatted_obs,formatted_cem_sampled_acs).view(action.shape[0], -1, 1)
                        
                        if iter_idx > 0 or dist_idx > 0:
                            all_pred_qvals = torch.cat((all_pred_qvals, last_top_k_qvals), dim=1)
                            cem_sampled_acs = torch.cat((cem_sampled_acs, last_elite_acs), dim=1)

                        top_k_qvals, top_k_inds = torch.topk(all_pred_qvals, num_select, dim=1)
                        elite_ac_inds = top_k_inds.repeat(1, 1, action.shape[1])
                        elite_acs = torch.gather(cem_sampled_acs, 1, elite_ac_inds)
                        last_top_k_qvals, last_elite_acs = top_k_qvals, elite_acs

                        updated_mu = torch.mean(elite_acs, dim=1)
                        updated_sigma = torch.std(elite_acs, dim=1)

                        self.omar_mu = updated_mu
                        self.omar_sigma = updated_sigma

                top_qvals, top_inds = torch.topk(all_pred_qvals, 1, dim=1)
                top_ac_inds = top_inds.repeat(1, 1, action.shape[1])
                top_acs = torch.gather(cem_sampled_acs, 1, top_ac_inds)

                cem_qvals = top_qvals
                pol_qvals = q1_new_action.unsqueeze(1)
                cem_acs = top_acs
                pol_acs = new_action.unsqueeze(1)

                candidate_qvals = torch.cat([pol_qvals, cem_qvals], 1)
                candidate_acs = torch.cat([pol_acs, cem_acs], 1)
                max_qvals, max_inds = torch.max(candidate_qvals, 1, keepdim=True)
                max_ac_inds = max_inds.repeat(1, 1, action.shape[1])
                max_acs = torch.gather(candidate_acs, 1, max_ac_inds).squeeze(1)
                max_acs = max_acs.detach()
                mimic_term = F.mse_loss(new_action, max_acs,reduction="none")
                bc_loss=mimic_term.mean()
                tau_broadcasted = self.omar_coe.expand(-1, self.action_dim)
                actor_loss = (tau_broadcasted * mimic_term).sum(dim=1).mean() - ((1 - self.omar_coe) * q1_new_action).mean()
            else:
                bc_loss=F.mse_loss(new_action, action) 
                lmbda=self.alpha/q1_new_action.abs().mean().detach()
                actor_loss = -lmbda * q_loss + bc_loss
            actor_loss += (new_action ** 2).mean() * 1e-3
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm(self.actor.parameters(), max_norm=1.0, norm_type=2)
            self.actor_optimizer.step()

            if self.step%5==0:
                for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            self.step += 1
            self.actor_lr_scheduler.step()
            self.critic_lr_scheduler.step()

            metric['actor_loss'].append(actor_loss.item())
            metric['bc_loss'].append(bc_loss.item())
            metric['ql_loss'].append(q_loss.item())
            metric['critic_loss'].append(critic_loss.item())

        return metric

    def sample_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()

    def save_model(self, dir, id=None):
        if id is not None:
            torch.save(self.actor.state_dict(), f'{dir}/actor_{id}.pth')
            torch.save(self.critic.state_dict(), f'{dir}/critic_{id}.pth')
        else:
            torch.save(self.actor.state_dict(), f'{dir}/actor.pth')
            torch.save(self.critic.state_dict(), f'{dir}/critic.pth')

    def load_model(self, dir, id=None):
        if id is not None:
            self.actor.load_state_dict(torch.load(f'{dir}/actor_{id}.pth'))
            self.critic.load_state_dict(torch.load(f'{dir}/critic_{id}.pth'))
        else:
            self.actor.load_state_dict(torch.load(f'{dir}/actor.pth'))
            self.critic.load_state_dict(torch.load(f'{dir}/critic.pth'))


