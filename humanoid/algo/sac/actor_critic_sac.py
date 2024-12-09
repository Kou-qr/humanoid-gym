import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np


class ActorSAC(nn.Module):
    def __init__(self, 
                 num_actor_obs,
                 num_actions,
                 actor_hidden_dims=[256, 256],
                 activation=nn.ReLU(),
                 max_action=1.0,
                 log_std_init = 1.0,
                 log_std_max = 4.0,
                 log_std_min = -20.0,
                 **kwargs):
        if kwargs:
            print("ActorSAC.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(ActorSAC, self).__init__()

        self.max_action = max_action
        self.log_std_init = log_std_init
        self.log_std_max = log_std_max
        self.log_std_min = log_std_min

        # Actor network
        actor_layers = []
        actor_layers.append(nn.Linear(num_actor_obs, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for i in range(len(actor_hidden_dims) - 1):
            # if i == len((actor_hidden_dims) - 2):
            #     actor_layers.append(nn.Linear(actor_hidden_dims[i], actor_hidden_dims[i + 1]))
            # else:
            actor_layers.append(nn.Linear(actor_hidden_dims[i], actor_hidden_dims[i + 1]))                
            actor_layers.append(activation)
        self.adaptor = nn.Sequential(*actor_layers)
        print(f"Actor_adaptor MLP: {self.adaptor}")
        #actor network
        self.mu = nn.Linear(actor_hidden_dims[-1], num_actions)  # Mean μ layer
        self.log_std = nn.Linear(actor_hidden_dims[-1], num_actions)  # std
        # self.std = nn.Linear(actor_hidden_dims[-1], num_actions)  # std

        # Action noise
        self.noise_distribution = None

    def forward(self,observations):
        x = self.adaptor(observations)
        mu=self.mu(x)
        # std=F.softplus(self.std(x))
        log_std = (self.log_std_init + self.log_std(x)).clamp(self.log_std_min, self.log_std_max)
        std = log_std.exp()
        return mu, std
    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        actions = self.distribution.rsample() 
        log_prob=self.distribution.log_prob(actions)
        # actions=torch.tanh(actions)
        log_prob=log_prob - torch.log(1-torch.tanh(actions).pow(2) + 1e-6)
        actions=torch.tanh(actions) * self.max_action
        return actions, log_prob
    
    # def forward(self,observations):
    #     x = self.adaptor(observations)
    #     mu = self.mu(x)
    #     log_std = torch.clamp(self.log_std(x), -20, 2)  # 限制标准差范围
    #     std = torch.exp(log_std)
    #     return mu, std
    # def act(self, observations, **kwargs):
    #     self.update_distribution(observations)
    #     actions = self.distribution.sample() 
    #     log_prob = self.distribution.log_prob(actions).sum(dim=-1, keepdim=True)
    #     log_prob -= (2 * (np.log(2) - actions - F.softplus(-2 * actions))).sum(dim=-1, keepdim=True)  # 修正Tanh分布的log_prob
    #     actions = torch.tanh(actions) * self.max_action
    #     return actions, log_prob
    
    def update_distribution(self, observations):
        mean,std= self.forward(observations)
        self.distribution = torch.distributions.Normal(mean,std)
    
    def act_inference(self, observations):
        x = self.adaptor(observations)
        actions_mean = self.mu(x)  # 只提取均值
        return actions_mean

    # def act_inference(self, observations):
    #     actions_mean,_ = self.forward(observations)
    #     return actions_mean

class CriticSAC(torch.nn.Module):
    def __init__(self, 
                 num_critic_obs,
                 num_actions,
                 critic_hidden_dims=[256, 256],
                 activation=nn.ReLU(),
                 **kwargs):
        if kwargs:
            print("CriticSAC.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(CriticSAC, self).__init__()
        # Critic networks (Q1 and Q2)
        # Q1
        q1_layers = []
        q1_layers.append(nn.Linear(num_critic_obs + num_actions, critic_hidden_dims[0]))
        q1_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                q1_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                q1_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                q1_layers.append(activation)
        self.q1 = nn.Sequential(*q1_layers)

        # Q2
        q2_layers = []
        q2_layers.append(nn.Linear(num_critic_obs + num_actions, critic_hidden_dims[0]))
        q2_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                q2_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                q2_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                q2_layers.append(activation)
        self.q2 = nn.Sequential(*q2_layers)

        print(f"Critic1 MLP: {self.q1}")
        print(f"Critic2 MLP: {self.q2}")

    def forward(self, observations, actions,**kwargs):
        """Compute Q1 and Q2 values for the given state-action pair."""
        sa = torch.cat([observations, actions], dim=1)
        q1 = self.q1(sa)
        q2 = self.q2(sa)
        return q1, q2
        
    def evaluate(self, observations, actions, **kwargs):
        """
        Define the forward pass for the Critic.
        This method is required by PyTorch.
        """
        return self.forward(observations, actions)

class ActorCriticSAC(nn.Module):
    def __init__(self,  num_actor_obs,
                        num_critic_obs,
                        num_actions,
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        activation = nn.ELU(),
                        max_action = 1.0,
                        log_std_init = 1.0,
                        log_std_max = 4.0,
                        log_std_min = -20.0,
                        **kwargs):
        if kwargs:
            print("ActorCriticSAC.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(ActorCriticSAC, self).__init__()
        self.max_action = max_action
        self.num_actions = num_actions

        self.actor = ActorSAC(
                    num_actor_obs=num_actor_obs,
                    num_actions=num_actions,
                    actor_hidden_dims=actor_hidden_dims,
                    activation=activation,
                    max_action=max_action,
                    log_std_init=log_std_init,
                    log_std_max=log_std_max,
                    log_std_min=log_std_min,
        )
        self.critic = CriticSAC(
                 num_critic_obs=num_critic_obs,
                 num_actions = num_actions,
                 critic_hidden_dims=critic_hidden_dims,
                 activation=activation,
        )

    def reset(self, dones=None):
        pass
    def act(self, observations, **kwargs):
        actions, log_prob = self.actor.act(observations)
        return actions, log_prob

    def act_inference(self, observations):
        actions_mean = self.actor.act_inference(observations)
        return actions_mean
    
    def evaluate(self, observations, actions,**kwargs):
        q1,q2 = self.critic(observations, actions)
        return q1, q2

        

    


    