import numpy as np

class ResourceGatheringEnv:
    """
    State: [inventory between interval [0,1], hazard between interval [0,1], intel between interval [0,1]]
    Actions:
      0 = gather (inc. inventory, risk inc. hazard unless intel is high)
      1 = scout (inc. intel, decr. hazard)
      2 = return_base (cash-in inventory -> bonus, reset hazard a bit)
    Reward: inventory growth and periodic cash-ins; penalize high hazard.
    """
    action_space_n = 3

    def __init__(self, max_steps=30, rng=None):
        self.max_steps = max_steps
        self.rng = np.random.default_rng(None if rng is None else rng)

    def reset(self):
        self.steps = 0
        self.inventory = 0.0
        self.hazard = self.rng.uniform(0.2, 0.5)
        self.intel = self.rng.uniform(0.2, 0.5)
        self.bank = 0.0  # accumulated value from cash-ins
        return self._state()

    def step(self, action: int):
        reward = 0.0
        if action == 0:  # gather
            gain = 0.10 + 0.02*self.rng.normal()
            self.inventory = np.clip(self.inventory + gain, 0.0, 1.0)
            hazard_up = max(0.01, 0.08*(1.0 - self.intel))
            self.hazard = np.clip(self.hazard + hazard_up + 0.02*self.rng.normal(), 0.0, 1.0)
            reward += 0.3*gain
        elif action == 1:  # scout
            self.intel = np.clip(self.intel + 0.10 + 0.02*self.rng.normal(), 0.0, 1.0)
            self.hazard = np.clip(self.hazard - 0.08 + 0.02*self.rng.normal(), 0.0, 1.0)
            reward += 0.1
        else:  # return_base
            reward += 1.2*self.inventory
            self.bank += self.inventory
            self.inventory = 0.0
            self.hazard = np.clip(self.hazard - 0.05, 0.0, 1.0)

        self.steps += 1
        done = self.steps >= self.max_steps

        hazard_penalty = 0.6*self.hazard
        reward -= hazard_penalty
        return self._state(), float(reward), bool(done), {"bank": self.bank}

    def _state(self):
        return np.array([self.inventory, self.hazard, self.intel], dtype=np.float32)

def make_env():
    return ResourceGatheringEnv()

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
