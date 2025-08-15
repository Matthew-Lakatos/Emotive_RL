import torch

def adjusted_advantage(A_t, delta_t):

  # functionality described in the paper
  
  return A_t * (1 + delta_t.abs())
