import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy

from .actor_critic_sac import ActorCriticSAC
from .rollout_storage_sac import RolloutStorage

class SAC:
    actor_critic : ActorCriticSAC
    def __init__(
                self, 
                actor_critic,
                num_learning_epochs=1, 
                num_mini_batches=1,
                gamma=0.99, 
                tau=0.005, 
                gradient_clip = 1.0,
                # learning_rate=1e-3,
                actor_lr = 1e-4,
                critic_lr= 1e-3,
                alpha_lr = 1e-3,
                initial_alpha=1.0, 
                device='cpu',
    ):
        
        # SAC parameters
        self.device = device
        self._gradient_clip = gradient_clip
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.alpha_lr = alpha_lr
        # self.learning_rate = learning_rate
        # self.alpha = self.log_alpha.exp()       
        self.gamma = gamma
        self.tau = tau
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches

        
        # SAC components
        self.actor_critic = actor_critic.to(self.device)
        self.actor = self.actor_critic.actor
        self.critic = self.actor_critic.critic
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.storage = None # initialized later
        self.transition = RolloutStorage.Transition()
        self.max_action = self.actor_critic.max_action

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        # Dynamic entropy coefficient
        self.log_alpha = torch.tensor(np.log(initial_alpha), requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)
        self.target_entropy = -np.prod(actor_critic.num_actions).item()  

        # SAC parameters
        # self.alpha = self.log_alpha.exp()       
        self.gamma = gamma
        self.tau = tau
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        
    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        self.storage = RolloutStorage(num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, self.device)

    def test_mode(self):
        self.actor_critic.test()
    
    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs):
        actions, log_probs = self.actor_critic.act(obs)
        self.transition.actions = actions.detach()
        Q1,Q2 = self.actor_critic.evaluate(critic_obs,self.transition.actions)
        self.transition.values = torch.min(Q1,Q2).detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        return self.transition.actions

    def process_env_step(self,obs,critic_obs,rewards, dones, infos):

        self.transition.next_observations = obs.clone()
        self.transition.next_critic_observations = critic_obs.clone()
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)
    @property
    def alpha(self):
        return self.log_alpha.exp()
    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_alpha_loss = 0
        param_diff = 0

        # Iterate over mini-batches
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for obs_batch, critic_obs_batch, actions_batch,rewards_batch, next_obs_batch,next_critic_obs_batch,dones_batch,hid_states_batch,masks_batch in generator:

            # 更新 Critic 网络
            with torch.no_grad():
                next_actions_batch, next_log_probs = self.actor_critic.act(next_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
                target_q1, target_q2 = self.critic_target.evaluate(next_critic_obs_batch, next_actions_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
                target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs
                target_q = rewards_batch + self.gamma * (1 - dones_batch) * target_q

            current_q1, current_q2 = self.actor_critic.evaluate(critic_obs_batch, actions_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            critic_loss = nn.functional.mse_loss(current_q1, target_q) + nn.functional.mse_loss(current_q2, target_q)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), self._gradient_clip)
            self.critic_optimizer.step()

            # 更新 Actor 网络
            actions, log_probs = self.actor_critic.act(obs_batch)
            q1_new, q2_new = self.actor_critic.evaluate(critic_obs_batch, actions)
            actor_loss = (self.alpha.detach() * log_probs - torch.min(q1_new, q2_new)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self._gradient_clip)
            self.actor_optimizer.step()       

            # 动态调整熵系数
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            # self.alpha = self.log_alpha.exp()  # 更新 alpha

            # 更新目标网络
            self._soft_update(self.critic, self.critic_target, self.tau)

            # 计算两个网络参数的差异
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                param_diff += torch.norm(param - target_param).item()
            print("Parameter difference:", param_diff)


            mean_surrogate_loss += actor_loss.item()
            mean_value_loss += critic_loss.item()
            mean_alpha_loss += alpha_loss.item()
            
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_alpha_loss /= num_updates
        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss,mean_alpha_loss

    def _soft_update(self, source, target, tau):
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)











































