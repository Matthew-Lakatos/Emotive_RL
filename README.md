# Requirements

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
