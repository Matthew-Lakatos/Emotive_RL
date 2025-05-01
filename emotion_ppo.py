import torch

def adjusted_advantage(A_t, delta_t):
    """Affect-weighted advantage function"""
    return A_t * (1 + delta_t.abs())