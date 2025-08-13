# experiments/affective_tutor.py
import numpy as np

class AffectiveTutorEnv:
    def __init__(self, max_steps=20):
        self.max_steps = max_steps

    def reset(self):
        self.steps = 0
        self.mood = np.random.rand()
        self.knowledge = np.random.rand()
        return np.array([self.mood, self.knowledge], dtype=np.float32)

    def step(self, action):
        reward = 0.0
        if action == 0:
            self.mood = min(1.0, self.mood + 0.1)
            reward += 0.5
        elif action == 1:
            self.knowledge = min(1.0, self.knowledge + 0.1)
            reward += 1.0 if self.mood > 0.5 else -0.5

        self.steps += 1
        done = self.steps >= self.max_steps
        return np.array([self.mood, self.knowledge], dtype=np.float32), reward, done, {}

def make_env():
    return AffectiveTutorEnv()
