import numpy as np

class LongHaulMissionEnv:
    """
    State: [resources between interval [0,1], morale between interval [0,1], progress between interval [0,1]]
    Actions:
      0 = conserve (inc. resources, slight decr. in progress, stabilizes morale)
      1 = push_ahead (inc. progress, consumes resources, decr. morale if too low)
      2 = team_build (inc. morale, slight resources use)
    Reward: progress with penalties for resource starvation and very low morale.
    """
    action_space_n = 3

    def __init__(self, max_steps=35, rng=None):
        self.max_steps = max_steps
        self.rng = np.random.default_rng(None if rng is None else rng)

    def reset(self):
        self.steps = 0
        self.resources = self.rng.uniform(0.4, 0.8)
        self.morale = self.rng.uniform(0.4, 0.8)
        self.progress = 0.0
        return self._state()

    def step(self, action: int):
        if action == 0:  # conserve
            self.resources = np.clip(self.resources + 0.06 + 0.02*self.rng.normal(), 0.0, 1.0)
            self.progress = np.clip(self.progress - 0.01 + 0.01*self.rng.normal(), 0.0, 1.0)
            self.morale = np.clip(self.morale + 0.01 + 0.01*self.rng.normal(), 0.0, 1.0)
        elif action == 1:  # push_ahead
            self.progress = np.clip(self.progress + 0.08 + 0.02*self.rng.normal(), 0.0, 1.0)
            self.resources = np.clip(self.resources - 0.08 + 0.02*self.rng.normal(), 0.0, 1.0)
            if self.resources < 0.3:
                self.morale = np.clip(self.morale - 0.06 + 0.02*self.rng.normal(), 0.0, 1.0)
        else:  # team_build
            self.morale = np.clip(self.morale + 0.08 + 0.02*self.rng.normal(), 0.0, 1.0)
            self.resources = np.clip(self.resources - 0.03 + 0.02*self.rng.normal(), 0.0, 1.0)

        self.steps += 1
        done = self.steps >= self.max_steps or self.progress >= 1.0

        starvation_penalty = 0.7*max(0.0, 0.2 - self.resources)
        morale_penalty = 0.5*max(0.0, 0.25 - self.morale)
        reward = 1.5*self.progress - starvation_penalty - morale_penalty
        return self._state(), float(reward), bool(done), {}

    def _state(self):
        return np.array([self.resources, self.morale, self.progress], dtype=np.float32)

def make_env():
    return LongHaulMissionEnv()

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
