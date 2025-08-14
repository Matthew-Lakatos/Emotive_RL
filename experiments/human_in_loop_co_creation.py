import numpy as np

class CoCreationEnv:
    """
    State: [creativity between interval [0,1], alignment between interval [0,1], pace between interval [0,1]]
    Actions:
      0 = propose (inc. creativity, risks  decr. in alignment if too fast)
      1 = ask_feedback (inc. alignment, slows pace, small inc. in creativity)
      2 = iterate (moderate inc. creativity & inc. in alignment, mild pace inc.)
    Reward: creativity + alignment, penalize extreme pace (> 0.8).
    """
    action_space_n = 3

    def __init__(self, max_steps=25, rng=None):
        self.max_steps = max_steps
        self.rng = np.random.default_rng(None if rng is None else rng)

    def reset(self):
        self.steps = 0
        self.creativity = self.rng.uniform(0.2, 0.5)
        self.alignment = self.rng.uniform(0.3, 0.6)
        self.pace = self.rng.uniform(0.3, 0.6)
        return self._state()

    def step(self, action: int):
        if action == 0:  # propose
            self.creativity = np.clip(self.creativity + 0.10 + 0.02*self.rng.normal(), 0.0, 1.0)
            self.pace = np.clip(self.pace + 0.08, 0.0, 1.0)
            if self.pace > 0.7:
                self.alignment = np.clip(self.alignment - 0.05 + 0.02*self.rng.normal(), 0.0, 1.0)
        elif action == 1:  # ask_feedback
            self.alignment = np.clip(self.alignment + 0.09 + 0.02*self.rng.normal(), 0.0, 1.0)
            self.pace = np.clip(self.pace - 0.06, 0.0, 1.0)
            self.creativity = np.clip(self.creativity + 0.02 + 0.02*self.rng.normal(), 0.0, 1.0)
        else:  # iterate
            self.creativity = np.clip(self.creativity + 0.06 + 0.02*self.rng.normal(), 0.0, 1.0)
            self.alignment = np.clip(self.alignment + 0.05 + 0.02*self.rng.normal(), 0.0, 1.0)
            self.pace = np.clip(self.pace + 0.03, 0.0, 1.0)

        self.steps += 1
        done = self.steps >= self.max_steps

        reward = 0.8*self.creativity + 0.8*self.alignment - 0.6*max(0.0, self.pace - 0.8)
        return self._state(), float(reward), bool(done), {}

    def _state(self):
        return np.array([self.creativity, self.alignment, self.pace], dtype=np.float32)

def make_env():
    return CoCreationEnv()

def run(agent=None, episodes=100):
    env = make_env()
    rewards = []
    for _ in range(episodes):
        s = env.reset()
        done = False
        ep_r = 0.0
        while not done:
            a = agent.act(s) if agent else np.random.randint(env.action_space_n)
            ns, r, done, _ = env.step(a)
            if agent:
                agent.learn(s, a, r, ns, done)
            s = ns
            ep_r += r
        rewards.append(ep_r)
    return float(np.mean(rewards[-10:]))
