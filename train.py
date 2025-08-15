import argparse
import importlib
import torch
import numpy as np
import sys

def run_with_agent_class(agent_module_name, agent_class_name, env_name, episodes):
    # dynamically import the environment
    env_module = importlib.import_module(f"environments.{env_name}")
    env = getattr(env_module, env_name)()

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # dynamically import the agent class
    module = importlib.import_module(agent_module_name)
    AgentClass = getattr(module, agent_class_name)

    # instantiate the agent and train in a loop
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

        if (ep + 1) % 50 == 0:
            print(f"[{agent_class_name} | {env_name}] Episode {ep + 1}/{episodes} Reward: {ep_reward:.2f}")

    torch.save(agent.state_dict(), f"{agent_class_name}_{env_name}_model.pth")


def fallback_train_script(agent_name, env_name, episodes):
    """
    Fall back to a standalone training script if the agent class
    (e.g., agent_ppo.PPOAgent) is not available.
    """
    script_module_name = f"train_{agent_name}"
    train_function_name = f"train_{agent_name}"

    try:
        module = importlib.import_module(script_module_name)
    except ImportError:
        raise ImportError(
            f"Neither agent class nor standalone training script '{script_module_name}.py' was found."
        )

    if not hasattr(module, train_function_name):
        raise AttributeError(
            f"Training script '{script_module_name}.py' does not define a '{train_function_name}(...)' function."
        )

    train_fn = getattr(module, train_function_name)
    train_fn(env_name, episodes)


def main(agent_name, env_name, episodes):
    """
    1) Try the class-based agent (original functionality)
    2) If that fails, fall back to the standalone training script
    """
    AGENT_MAP = {
        "ppo": ("agent_ppo", "PPOAgent"),
        "emotion_mod": ("agent_emotion_mod", "EmotionModAgent"),
        "emotive_rl": ("agent_emotive_rl", "EmotiveRLAgent"),
    }

    if agent_name not in AGENT_MAP:
        raise ValueError(f"Unknown agent: {agent_name}")

    agent_module_name, agent_class_name = AGENT_MAP[agent_name]

    try:
        run_with_agent_class(agent_module_name, agent_class_name, env_name, episodes)
    except (ImportError, ModuleNotFoundError) as e:
        print(f"[INFO] Agent class could not be imported: {e}")
        print("[INFO] Falling back to the standalone training script.")
        fallback_train_script(agent_name, env_name, episodes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str, required=True)
    parser.add_argument("--env", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=500)
    args = parser.parse_args()

    main(args.agent, args.env, args.episodes)
