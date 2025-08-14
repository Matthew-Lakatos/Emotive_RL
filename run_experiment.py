import importlib
import torch
import numpy as np
import os

from agents.agent_ppo import PPOAgent
from agents.agent_emotion_mod import EmotionModAgent
from agents.agent_emotive_rl import EmotiveRLAgent

AGENT_CLASSES = {
    "ppo": PPOAgent,
    "emotion_mod": EmotionModAgent,
    "emotive_rl": EmotiveRLAgent
}

EXPERIMENTS = [
    "affective_tutor",
    "conflict_resolution",
    "emotion_exploration",
    "human_in_loop_co_creation",
    "long_haul_mission",
    "resource_gathering",
    "social_navigation"
]

def evaluate(agent_name, env_name, episodes=20):
    env_module = importlib.import_module(f"experiments.{env_name}")
    env = env_module.make_env()

    # Init agent & load weights
    state_size = env.reset().shape[0]
    action_size = 2
    AgentClass = AGENT_CLASSES[agent_name]
    agent = AgentClass(state_size, action_size)

    model_path = f"models/{agent_name}_{env_name}.pt"
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return None
    agent.policy.load_state_dict(torch.load(model_path))
    agent.policy.eval()

    rewards = []
    for _ in range(episodes):
        state = env.reset()
        ep_reward = 0
        done = False
        while not done:
            action = agent.act(state)
            state, reward, done, _ = env.step(action)
            ep_reward += reward
        rewards.append(ep_reward)

    return np.mean(rewards)

if __name__ == "__main__":
    results = {}
    for agent_name in AGENT_CLASSES.keys():
        results[agent_name] = {}
        for env_name in EXPERIMENTS:
            avg_reward = evaluate(agent_name, env_name)
            results[agent_name][env_name] = avg_reward
            print(f"{agent_name} on {env_name}: {avg_reward}")
    print(results)
