import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import json

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh()
        )
        self.mean = nn.Linear(64, output_dim)
        self.log_std = nn.Parameter(torch.zeros(output_dim))

    def forward(self, x):
        x = self.fc(x)
        return self.mean(x), self.log_std.exp()

    def get_action(self, state):
        mean, std = self.forward(state)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        return action, dist.log_prob(action).sum(dim=-1), dist

class ValueNetwork(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.value = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.value(x)

def compute_gae(rewards, masks, values, gamma=0.99, tau=0.95):
    advantages = []
    gae = 0
    next_value = 0
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * next_value * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        advantages.insert(0, gae)
        next_value = values[step]
    return advantages

def train_ppo(env_name):
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    policy = PolicyNetwork(obs_dim, act_dim)
    value_fn = ValueNetwork(obs_dim)
    optimizer_policy = optim.Adam(policy.parameters(), lr=3e-4)
    optimizer_value = optim.Adam(value_fn.parameters(), lr=1e-3)

    max_episodes = 100
    steps_per_update = 2048
    log_dir = f"logs/ppo_{env_name}"
    os.makedirs(log_dir, exist_ok=True)

    for episode in range(max_episodes):
        states, actions, rewards, log_probs, values, masks = [], [], [], [], [], []
        state = env.reset()
        ep_reward = 0

        for _ in range(steps_per_update):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action, log_prob, dist = policy.get_action(state_tensor)
            value = value_fn(state_tensor)

            next_state, reward, done, _ = env.step(action.detach().numpy())
            ep_reward += reward

            states.append(state_tensor)
            actions.append(action)
            rewards.append(torch.tensor([reward], dtype=torch.float32))
            log_probs.append(log_prob)
            values.append(value.squeeze(0))
            masks.append(torch.tensor([1 - done], dtype=torch.float32))

            state = next_state
            if done:
                state = env.reset()

        states = torch.cat(states)
        actions = torch.cat(actions)
        old_log_probs = torch.stack(log_probs).detach()
        values = torch.stack(values).detach()
        returns = compute_gae(rewards, masks, values)
        returns = torch.tensor(returns)
        advantages = returns - values

        for _ in range(4):  # PPO update epochs
            new_mean, new_std = policy(states)
            dist = torch.distributions.Normal(new_mean, new_std)
            new_log_probs = dist.log_prob(actions).sum(dim=-1)
            ratio = (new_log_probs - old_log_probs).exp()

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 0.8, 1.2) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = (returns - value_fn(states).squeeze()).pow(2).mean()

            optimizer_policy.zero_grad()
            policy_loss.backward()
            optimizer_policy.step()

            optimizer_value.zero_grad()
            value_loss.backward()
            optimizer_value.step()

        with open(f"{log_dir}/ep{episode:03d}.json", "w") as f:
            json.dump({"reward": float(ep_reward)}, f)
        print(f"[PPO] Episode {episode} | Reward: {ep_reward:.2f}")

    env.close()
