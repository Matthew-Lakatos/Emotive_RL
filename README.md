
# Emotive_RL: Emotion-Driven Reinforcement Learning with Predictive and Neuromorphic Modulation

## Overview

**Emotive_RL** is a biologically inspired reinforcement learning (RL) framework that simulates emotional cognition to improve learning and adaptability in autonomous agents. Grounded in **Friston’s Free Energy Principle** and the **Bayesian Brain Hypothesis**, this codebase implements emotion modeling as recursive, predictive belief updates. The emotional states influence action selection via neuromodulatory signals modeled with spiking dynamics, inspired by neurotransmitter activity (dopamine, serotonin).

This project is based on the supplementary study:
**"Predictive Emotion Dynamics and Neuromorphic Mood Embeddings in Reinforcement Learning Agents"**  
Matthew Lakatos (2025)

## Core Concepts

- **Predictive Emotion Coding**: Uses recurrent models (e.g., GRUs) to anticipate future emotional states, minimizing prediction error through time.
- **Neuromorphic Mood Embeddings**: Encodes mood states as dopaminergic/serotonergic analogs that influence the reward signal.
- **Affect-Weighted RL**: Integrates emotional prediction error into the advantage function of PPO, modulating action values.
- **Free Energy Minimization**: Emotions guide agents to minimize uncertainty and "surprise" in complex environments.

## Features

- Emotionally augmented Proximal Policy Optimization (PPO)
- Predictive emotion module using GRU-based RNNs
- Neuromodulatory reward shaping
- ROS deployment support for real-world robots

## Directory Structure

```
Emotive_RL-main/
|-- emotion_ppo.py          # PPO agent with emotional neuromodulation
|-- neuromodulator.py       # Simulates neurotransmitter dynamics
|-- predictive_emotion.py   # GRU-based predictive emotion model
|-- train.py                # Baseline PPO training
|-- run_experiment.py       # Command-line interface for experiment control
|-- ros_deployment.py       # ROS integration and deployment
|-- README.md               # Markdown README version
    Emotive_RL-main/experiments/
    |-- affective_tutor.py
    |-- conflict_resolution.py
    |-- emotion_exploration.py
    |-- human_in_loop_co_creation.py
    |-- long_haul_mission.py
    |-- resource_gathering.py
    |-- social_navigation.py
        Emotive_RL-main/experiments/results/
        |-- result_table
```

## Requirements

- Python 3.8+
- PyTorch
- NumPy
- Gym (for simulation environments)
- ROS

To train each model (designed to enable copy-paste):

```bash
python train.py --agent ppo --env affective_tutor --episodes 1000
python train.py --agent ppo --env conflict_resolution --episodes 1000
python train.py --agent ppo --env emotion_exploration --episodes 1000
python train.py --agent ppo --env human_in_loop_co_creation --episodes 1000
python train.py --agent ppo --env long_haul_mission --episodes 1000
python train.py --agent ppo --env resource_gathering --episodes 1000
python train.py --agent ppo --env social_navigation --episodes 1000
python train.py --agent emotive_rl --env affective_tutor --episodes 1000
python train.py --agent emotive_rl --env conflict_resolution --episodes 1000
python train.py --agent emotive_rl --env emotion_exploration --episodes 1000
python train.py --agent emotive_rl --env human_in_loop_co_creation --episodes 1000
python train.py --agent emotive_rl --env long_haul_mission --episodes 1000
python train.py --agent emotive_rl --env resource_gathering --episodes 1000
python train.py --agent emotive_rl --env social_navigation --episodes 1000
python train.py --agent emotion_mod --env affective_tutor --episodes 1000
python train.py --agent emotion_mod --env conflict_resolution --episodes 1000
python train.py --agent emotion_mod --env emotion_exploration --episodes 1000
python train.py --agent emotion_mod --env human_in_loop_co_creation --episodes 1000
python train.py --agent emotion_mod --env long_haul_mission --episodes 1000
python train.py --agent emotion_mod --env resource_gathering --episodes 1000
python train.py --agent emotion_mod --env social_navigation --episodes 1000
```

To run each experiment (benchmark testing after 1000 episodes):

```bash
python run_experiment.py
```
