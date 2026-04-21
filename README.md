# Crazyflie Experiments

This repository contains Crazyflie flight experiments that use Qualisys motion capture for closed-loop position feedback. 

## What Is In This Repo

- `hover_test.py`: basic takeoff, hover, and landing test with Qualisys feedback
- `q_learning_3d_flight.py`: 3D waypoint navigation using a greedy policy over a 3D Q-table
- `q_learning_controller.py`: 3D grid environment and Q-learning controller classes
- `cf_mocap.py`: shared Crazyflie and Qualisys integration code
- `visualize_3d.py`: plots a saved 3D trajectory from a CSV log
- `data/`: saved trajectory logs and generated plots

## Prerequisites

You will need:

- A Crazyflie with a working radio link
- A Crazyradio dongle
- A Qualisys system with QTM streaming enabled
- A tracked rigid body in QTM whose name matches `MARKER_DECK_NAME`
- Python with the required libraries installed

Based on the imports in this repo, you should expect to need at least:

- `numpy`
- `matplotlib`
- `scipy`
- `cflib`
- `qtm_rt`

## Before You Fly

Update the user settings at the top of the scripts before running them.

The most important values are:

- `URI`: Crazyradio URI for your drone
- `QTM_IP`: IP address of the Qualisys/QTM server
- `MARKER_DECK_NAME`: rigid body name in QTM
- `MARKER_DECK_IDS`: active marker IDs used by the Crazyflie

For the Q-learning script, also review:

- `GRID_X_MIN`, `GRID_X_MAX`
- `GRID_Y_MIN`, `GRID_Y_MAX`
- `GRID_Z_MIN`, `GRID_Z_MAX`
- `GRID_SIZE`
- `HOVER_Z`
- `WAYPOINT_REACH_THRESHOLD`
- `MAX_NAVIGATION_STEPS`
- `Q_TABLE_FILE`

Make sure the grid bounds match the real flyable mocap volume.

## Recommended Workflow

Run ```hover_test.py``` first every time you bring the system up. It is the quickest way to verify:

- the Crazyflie link is healthy
- Qualisys is streaming valid poses
- the estimator can lock onto the mocap feedback
- takeoff and landing work cleanly

### 1. Hover Test

```bash
python hover_test.py
```

This script:

- connects to the Crazyflie
- starts the Qualisys pose stream
- warms up and resets the estimator
- arms the drone
- performs takeoff, hover, and landing

### 2. 3D Q-Learning Flight

```bash
python q_learning_3d_flight.py
```

This script:

- initializes a 3D grid environment
- loads `Q_TABLE_FILE` if provided, otherwise creates a fake 3D Q-table biased toward the far positive corner of the workspace (i.e., `(nx - 1, ny - 1, nz - 1)`)
- prints the predicted waypoint and action sequence
- asks for user confirmation before flight
- takes off to `HOVER_Z`
- navigates through the grid using the greedy policy $a=\arg\max_a Q(s,a)$ and Qualysis' position feedback
- logs the flown trajectory to `data/`
- optionally saves the generated Q-table to `data/`
- land the drone


## Visualizing A Saved Flight

After a run, visualize a saved trajectory with:

```bash
python visualize_3d.py data/trajectory_3d_YYYYMMDD_HHMMSS.csv
```

You can also choose an explicit output filename:

```bash
python visualize_3d.py data/trajectory_3d_YYYYMMDD_HHMMSS.csv --output my_plot.png
```

The plot is saved next to the CSV by default with a `_trajectory3d.png` suffix.

## Output Files

Typical outputs written to `data/` include:

- `trajectory_3d_<timestamp>.csv`: logged `(timestamp, x, y, z)` samples
- `trajectory_3d_<timestamp>_trajectory3d.png`: 3D trajectory plot
- `q_table_3d_<timestamp>.npy`: generated Q-table snapshot when `SAVE_Q_TABLE = True`

## Code Structure

### `cf_mocap.py`

Shared infrastructure for:

- Crazyflie connection and command streaming
- Qualisys/QTM connection and frame parsing
- mocap-to-Crazyflie frame transformation
- background `extpose` streaming

### `q_learning_controller.py`

Contains:

- `Action3D`: six discrete actions in the grid
- `GridEnvironment3D`: continuous-to-grid and grid-to-continuous conversions
- `QLearningController3D`: greedy policy evaluation and state stepping

### `q_learning_3d_flight.py`

Top-level flight script that combines:

- mocap feedback
- 3D grid discretization
- Q-table action selection
- waypoint flight execution
- logging
