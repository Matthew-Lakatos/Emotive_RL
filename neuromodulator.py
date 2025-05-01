import torch

def modulate_reward(base_reward, neuromodulator_signal):
    """Apply neuromorphic reward modulation"""
    return base_reward * (1 + torch.tanh(neuromodulator_signal))