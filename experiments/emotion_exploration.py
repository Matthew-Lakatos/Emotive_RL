import numpy as np

class EmotionExplorationEnv:
    """
    State: [valence between interval [0,1], arousal between interval [0,1], stability between interval [0,1]]
    Actions:
      0 = soothe (decr. arousal, inc. stability, slight inc. valence)
      1 = excite (inc. arousal, possible inc. valence if not over-stimulated)
      2 = reflect (inc. stability, small decr. arousal)
    Reward: closeness to target emotional zone (valence roughly 0.7, arousal roughly 0.5) + stability.
    """
    action_space_n = 3
    TARGET = np.array([0.7, 0.5], dtype=np.float32)

    def __init__(self, max_steps=25, rng=None):
        self.max_steps = max_steps
        self.rng = np.random.default_rng(None if rng is None else rng)

    def reset(self):
        self.steps = 0
        self.valence = self.rng.uniform(0.2, 0.6)
        self.arousal = self.rng.uniform(0.3, 0.7)
        self.stability = self.rng.uniform(0.2, 0.6)
        return self._state()

    def step(self, action: int):
        if action == 0:  # soothe
            self.arousal = np.clip(self.arousal - 0.08 + 0.02*self.rng.normal(), 0.0, 1.0)
            self.valence = np.clip(self.valence + 0.03 + 0.02*self.rng.normal(), 0.0, 1.0)
            self.stability = np.clip(self.stability + 0.05, 0.0, 1.0)
        elif action == 1:  # excite
            self.arousal = np.clip(self.arousal + 0.10 + 0.03*self.rng.normal(), 0.0, 1.0)
            bump = 0.06 if self.arousal < 0.75 else -0.04
            self.valence = np.clip(self.valence + bump + 0.02*self.rng.normal(), 0.0, 1.0)
            self.stability = np.clip(self.stability - 0.03, 0.0, 1.0)
        else:  # reflect
            self.stability = np.clip(self.stability + 0.07, 0.0, 1.0)
            self.arousal = np.clip(self.arousal - 0.03 + 0.02*self.rng.normal(), 0.0, 1.0)

        self.steps += 1
        done = self.steps >= self.max_steps

        dist = np.linalg.norm(np.array([self.valence, self.arousal]) - self.TARGET)
        reward = 1.2*(1.0 - dist) + 0.4*self.stability
        return self._state(), float(reward), bool(done), {}

    def _state(self):
        return np.array([self.valence, self.arousal, self.stability], dtype=np.float32)

def make_env():
    return EmotionExplorationEnv()

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
