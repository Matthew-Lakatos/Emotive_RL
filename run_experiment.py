import argparse
from train_ppo import train_ppo
from train_emotion_mod import train_emotion_mod
from train_emotive_rl import train_emotive_rl

agent_map = {
    "ppo": train_ppo,
    "emotion_mod": train_emotion_mod,
    "emotive_rl": train_emotive_rl
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=agent_map.keys(), required=True)
    parser.add_argument("--env", type=str, required=True)
    args = parser.parse_args()

    agent_map[args.agent](env_name=args.env)
