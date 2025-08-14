import numpy as np

class SocialNavigationEnv:
    """
    State: [reputation between interval [0,1], empathy between interval [0,1], network between interval [0,1]]
    Actions:
      0 = help (inc. reputation, inc. empathy, slow network growth)
      1 = self_promote (inc. network, risks decr. in reputation if empathy low)
      2 = gossip (fast inc. network but decr. reputation, decr. empathy)
    Reward: network growth + reputation, with penalties if reputation collapses.
    """
    action_space_n = 3

    def __init__(self, max_steps=25, rng=None):
        self.max_steps = max_steps
        self.rng = np.random.default_rng(None if rng is None else rng)

    def reset(self):
        self.steps = 0
        self.reputation = self.rng.uniform(0.4, 0.7)
        self.empathy = self.rng.uniform(0.4, 0.7)
        self.network = self.rng.uniform(0.2, 0.5)
        return self._state()

    def step(self, action: int):
        if action == 0:  # help
            self.reputation = np.clip(self.reputation + 0.07 + 0.02*self.rng.normal(), 0.0, 1.0)
            self.empathy = np.clip(self.empathy + 0.05 + 0.02*self.rng.normal(), 0.0, 1.0)
            self.network = np.clip(self.network + 0.03 + 0.01*self.rng.normal(), 0.0, 1.0)
        elif action == 1:  # self_promote
            self.network = np.clip(self.network + 0.08 + 0.02*self.rng.normal(), 0.0, 1.0)
            if self.empathy < 0.5:
                self.reputation = np.clip(self.reputation - 0.05 + 0.02*self.rng.normal(), 0.0, 1.0)
        else:  # gossip
            self.network = np.clip(self.network + 0.10 + 0.02*self.rng.normal(), 0.0, 1.0)
            self.reputation = np.clip(self.reputation - 0.08 + 0.02*self.rng.normal(), 0.0, 1.0)
            self.empathy = np.clip(self.empathy - 0.05 + 0.02*self.rng.normal(), 0.0, 1.0)

        self.steps += 1
        done = self.steps >= self.max_steps

        rep_penalty = 0.7*max(0.0, 0.3 - self.reputation)
        reward = 1.0*self.network + 0.7*self.reputation - rep_penalty
        return self._state(), float(reward), bool(done), {}

    def _state(self):
        return np.array([self.reputation, self.empathy, self.network], dtype=np.float32)

def make_env():
    return SocialNavigationEnv()

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
