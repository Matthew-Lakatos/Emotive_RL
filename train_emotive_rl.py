import torch
import torch.nn as nn
import torch.optim as optim
import gym
import numpy as np

from emotion_model.predictive_emotion import PredictiveEmotion
from neuromodulation.modulator import modulate_reward
from rl_agent.advantage_adjust import adjusted_advantage

# --------- Policy and Value Networks for PPO ---------
class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, act_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.net(x)

class ValueNetwork(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x)

# --------- Main Training Loop ---------
def train(env_name="CartPole-v1", episodes=1000):
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    policy_net = PolicyNetwork(obs_dim, act_dim)
    value_net = ValueNetwork(obs_dim)
    emotion_model = PredictiveEmotion(input_dim=obs_dim, hidden_dim=32)

    optimizer_policy = optim.Adam(policy_net.parameters(), lr=3e-4)
    optimizer_value = optim.Adam(value_net.parameters(), lr=1e-3)
    optimizer_emotion = optim.Adam(emotion_model.parameters(), lr=1e-3)

    gamma = 0.99

    for ep in range(episodes):
        obs = env.reset()
        done = False

        log_probs = []
        rewards = []
        values = []
        states = []

        emotion_inputs = []

        while not done:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)  # Shape: [1, obs_dim]
            dist = policy_net(obs_tensor)
            value = value_net(obs_tensor)

            action = torch.distributions.Categorical(dist).sample()
            log_prob = torch.log(dist.squeeze(0)[action])

            next_obs, reward, done, _ = env.step(action.item())

            # Store step data
            states.append(obs_tensor)
            emotion_inputs.append(obs_tensor)
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)

            obs = next_obs

        # --------- Compute Discounted Returns ---------
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns)

        # --------- Emotion Prediction (et vs êt+1) ---------
        x_seq = torch.stack(emotion_inputs).unsqueeze(0)  # [1, seq_len, obs_dim]
        e_hat = emotion_model(x_seq).squeeze()  # predicted internal state
        et = returns.mean().detach()  # use mean reward as a proxy for true emotion, used generic implement here for simplicity
        delta_t = et - e_hat
        delta_t = torch.clamp(delta_t, -1.0, 1.0)  # stabilize training

        # --------- Advantage Calculation ---------
        values = torch.cat(values).squeeze()
        advantage = returns - values.detach()
        adj_adv = adjusted_advantage(advantage, delta_t)

        # --------- PPO Policy Update (with emotion-adjusted advantage) ---------
        log_probs = torch.stack(log_probs)
        policy_loss = -(log_probs * adj_adv).mean()

        optimizer_policy.zero_grad()
        policy_loss.backward()
        optimizer_policy.step()

        # --------- Value Function Update ---------
        value_loss = nn.MSELoss()(values, returns)
        optimizer_value.zero_grad()
        value_loss.backward()
        optimizer_value.step()

        # --------- Emotion Model Update (minimize δt = et - êt) ---------
        emotion_loss = delta_t.pow(2).mean()
        optimizer_emotion.zero_grad()
        emotion_loss.backward()
        optimizer_emotion.step()

        # --------- Log episode result ---------
        if ep % 10 == 0:
            print(f"[Episode {ep}] Total Reward = {sum(rewards):.2f} | δt = {delta_t.item():.4f}")

    # --------- Save Trained Models ---------
    env.close()
    torch.save(policy_net.state_dict(), "checkpoints/policy.pt")
    torch.save(emotion_model.state_dict(), "checkpoints/emotion_model.pt")

if __name__ == "__main__":
    train()
