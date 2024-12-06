import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def store(self, transition):
        self.buffer.append(transition)

    def sample(self):
        batch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(np.array(actions), dtype=torch.float32),
            torch.tensor(np.array(rewards), dtype=torch.float32).unsqueeze(1),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(np.array(dones), dtype=torch.float32).unsqueeze(1),
        )

    def size(self):
        return len(self.buffer)
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, max_action):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.mu = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        self.max_action = max_action

    def forward(self, state):
        x = self.net(state)
        mu = self.mu(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, -20, 2)  # 限制标准差范围
        std = torch.exp(log_std)
        return mu, std

    def sample(self, state):
        mu, std = self(state)
        dist = torch.distributions.Normal(mu, std)
        action = dist.rsample()  # 使用重参数化技巧
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        log_prob -= (2 * (np.log(2) - action - F.softplus(-2 * action))).sum(dim=-1, keepdim=True)  # 修正Tanh分布的log_prob
        action = torch.tanh(action) * self.max_action
        return action, log_prob
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Critic, self).__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1(x), self.q2(x)
class SACAgent:
    def __init__(self, state_dim, action_dim, max_action, hidden_dim=256, 
                 actor_lr=3e-4, critic_lr=3e-4, alpha=0.2, gamma=0.99, tau=0.005):
        self.actor = Actor(state_dim, action_dim, hidden_dim, max_action).to(device)
        self.critic = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.alpha = alpha
        self.gamma = gamma
        self.tau = tau

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        action, _ = self.actor.sample(state)
        return action.detach().cpu().numpy().flatten()

    def train(self, replay_buffer):
        # 从经验回放缓冲中采样
        states, actions, rewards, next_states, dones = replay_buffer.sample()

        # 更新 Critic 网络
        with torch.no_grad():
            next_actions, log_probs = self.actor.sample(next_states)
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.alpha * log_probs
            target_q = rewards + self.gamma * (1 - dones) * target_q

        current_q1, current_q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 更新 Actor 网络
        new_actions, log_probs = self.actor.sample(states)
        q1_new, q2_new = self.critic(states, new_actions)
        actor_loss = (self.alpha * log_probs - torch.min(q1_new, q2_new)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 更新目标网络
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = ...  # 环境初始化
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

replay_buffer = ReplayBuffer(buffer_size=1_000_000, batch_size=256)
agent = SACAgent(state_dim, action_dim, max_action)

num_episodes = 1000
for episode in range(num_episodes):
    state = env.reset()
    episode_reward = 0

    for t in range(1000):
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        replay_buffer.store((state, action, reward, next_state, done))

        state = next_state
        episode_reward += reward

        if replay_buffer.size() >= 1000:  # 经验池足够大时开始训练
            agent.train(replay_buffer)

        if done:
            break

    print(f"Episode {episode}, Reward: {episode_reward}")
