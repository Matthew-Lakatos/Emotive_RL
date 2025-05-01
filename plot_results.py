import matplotlib.pyplot as plt
import json
import os

def load_rewards(log_dir):
    rewards = []
    for fname in sorted(os.listdir(log_dir)):
        if fname.endswith(".json"):
            with open(os.path.join(log_dir, fname)) as f:
                data = json.load(f)
                rewards.append(data["reward"])
    return rewards

def plot_rewards():
    envs = ["HalfCheetah-v2", "Walker2d-v2", "Ant-v2"]
    agents = ["ppo", "emotion_mod", "emotive_rl"]

    for env in envs:
        plt.figure(figsize=(10, 6))
        for agent in agents:
            rewards = load_rewards(f"logs/{agent}_{env}")
            plt.plot(rewards, label=agent)
        plt.title(f"Training Rewards - {env}")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"plots/{env}_rewards.png")
        plt.close()

plot_rewards()