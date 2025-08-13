import argparse
import importlib
import torch
import numpy as np

from agents.agent_ppo import PPOAgent
from agents.agent_emotion_mod import EmotionModAgent
from agents.agent_emotive_rl import EmotiveRLAgent

AGENT_CLASSES = {
    "ppo": PPOAgent,
    "emotion_mod": EmotionModAgent,
    "emotive_rl": EmotiveRLAgent
}

def train_agent(agent_name, env_name, episodes=500):
    # Load env dynamically
    env_module = importlib.import_module(f"experiments.{env_name}")
    env = env_module.make_env()

    # Create agent
    AgentClass = AGENT_CLASSES[agent_name]
    state_size = env.reset().shape[0]
    action_size = 2  # simple assumption; could be env.action_space.n
    agent = AgentClass(state_size, action_size)

    for ep in range(episodes):
        state = env.reset()
        ep_reward = 0
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.learn(state, action, reward, next_state, done)
            state = next_state
            ep_reward += reward
        if (ep+1) % 50 == 0:
            print(f"[{agent_name} | {env_name}] Episode {ep+1}/{episodes} Reward: {ep_reward:.2f}")

    # Save model
    torch.save(agent.policy.state_dict(), f"models/{agent_name}_{env_name}.pt")
    print(f"Model saved to models/{agent_name}_{env_name}.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", required=True, choices=AGENT_CLASSES.keys())
    parser.add_argument("--env", required=True)
    parser.add_argument("--episodes", type=int, default=500)
    args = parser.parse_args()

    train_agent(args.agent, args.env, args.episodes)
