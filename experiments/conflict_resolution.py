import numpy as np

class ConflictResolutionEnv:
    """
    State: [tension between interval [0,1], trust_A between interval [0,1], trust_B between interval [0,1]]
    Actions:
      0 = mediate (decr. tension, small inc. trusts)
      1 = take_side (decr. tension if choose 'right' side by luck, but hurts other trust)
      2 = delay (tiny decr. in tension, risks decay in trust)
    Reward: bonus for lower tension and balanced high trusts, penalize polarization.
    """
    action_space_n = 3

    def __init__(self, max_steps=25, rng=None):
        self.max_steps = max_steps
        self.rng = np.random.default_rng(None if rng is None else rng)

    def reset(self):
        self.steps = 0
        self.tension = self.rng.uniform(0.5, 0.9)
        self.trustA = self.rng.uniform(0.3, 0.6)
        self.trustB = self.rng.uniform(0.3, 0.6)
        return self._state()

    def step(self, action: int):
        if action == 0:  # mediate
            self.tension = max(0.0, self.tension - 0.08 + 0.02*self.rng.normal())
            self.trustA = min(1.0, self.trustA + 0.03 + 0.01*self.rng.normal())
            self.trustB = min(1.0, self.trustB + 0.03 + 0.01*self.rng.normal())
        elif action == 1:  # take_side
            pick_A = self.rng.random() < 0.5
            if pick_A:
                self.trustA = min(1.0, self.trustA + 0.08 + 0.02*self.rng.normal())
                self.trustB = max(0.0, self.trustB - 0.10 + 0.02*self.rng.normal())
            else:
                self.trustB = min(1.0, self.trustB + 0.08 + 0.02*self.rng.normal())
                self.trustA = max(0.0, self.trustA - 0.10 + 0.02*self.rng.normal())
            self.tension = np.clip(self.tension - 0.04 + 0.04*self.rng.normal(), 0.0, 1.0)
        else:  # delay
            self.tension = max(0.0, self.tension - 0.02 + 0.03*self.rng.normal())
            self.trustA = np.clip(self.trustA - 0.01 + 0.02*self.rng.normal(), 0.0, 1.0)
            self.trustB = np.clip(self.trustB - 0.01 + 0.02*self.rng.normal(), 0.0, 1.0)

        self.steps += 1
        done = self.steps >= self.max_steps

        balance_penalty = abs(self.trustA - self.trustB)
        reward = (1.0 - self.tension) + 0.4*(self.trustA + self.trustB)/2 - 0.5*balance_penalty
        return self._state(), float(reward), bool(done), {}

    def _state(self):
        return np.array([self.tension, self.trustA, self.trustB], dtype=np.float32)

def make_env():
    return ConflictResolutionEnv()

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
