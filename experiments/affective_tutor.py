import numpy as np

class AffectiveTutorEnv:
    """
    State: [mood between interval [0,1], knowledge between interval [0,1], fatigue between interval [0,1]]
    Actions:
      0 = encourage (inc. mood, small knowledge gain)
      1 = challenge (inc. knowledge more if mood high, else penalty)
      2 = rest (decr. fatigue, small mood gain)
    Reward: weighted sum of knowledge gain and positive mood, penalize high fatigue.
    """
    action_space_n = 3

    def __init__(self, max_steps=30, rng=None):
        self.max_steps = max_steps
        self.rng = np.random.default_rng(None if rng is None else rng)

    def reset(self):
        self.steps = 0
        self.mood = self.rng.uniform(0.3, 0.7)
        self.knowledge = self.rng.uniform(0.1, 0.3)
        self.fatigue = self.rng.uniform(0.2, 0.5)
        return self._state()

    def step(self, action: int):
        prev_knowledge = self.knowledge

        if action == 0:  # encourage
            self.mood = min(1.0, self.mood + 0.08 + 0.02*self.rng.normal())
            self.knowledge = min(1.0, self.knowledge + 0.02 + 0.01*self.rng.normal())
            self.fatigue = min(1.0, self.fatigue + 0.02)
        elif action == 1:  # challenge
            gain = 0.10 if self.mood > 0.5 else -0.03
            self.knowledge = np.clip(self.knowledge + gain + 0.02*self.rng.normal(), 0.0, 1.0)
            self.mood = np.clip(self.mood - 0.05 + 0.02*self.rng.normal(), 0.0, 1.0)
            self.fatigue = min(1.0, self.fatigue + 0.06)
        else:  # rest
            self.fatigue = max(0.0, self.fatigue - 0.10 + 0.02*self.rng.normal())
            self.mood = min(1.0, self.mood + 0.03 + 0.02*self.rng.normal())

        self.steps += 1
        done = self.steps >= self.max_steps

        knowledge_gain = max(0.0, self.knowledge - prev_knowledge)
        reward = 1.5*knowledge_gain + 0.3*self.mood - 0.4*self.fatigue
        return self._state(), float(reward), bool(done), {}

    def _state(self):
        return np.array([self.mood, self.knowledge, self.fatigue], dtype=np.float32)

def make_env():
    return AffectiveTutorEnv()

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
