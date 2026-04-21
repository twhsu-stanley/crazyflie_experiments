# Crazyflie Experiments (work in progress)

This repository contains code for flying Crazyflie with Qualysis

Before starting your experiments, always run ```hover_test.py``` to verify the hardware is good to go.

## 3D Q-learning Closed-loop Control
```q_learning_3d_flight.py``` currently does the following:
1. load or create a 3D Q-table
2. use Qualysis position as feedback signal
3. based on current position, implement the greedy policy w.r.t. the 3D Q-table
