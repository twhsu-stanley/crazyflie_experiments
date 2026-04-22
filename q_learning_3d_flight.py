# q_learning_3d_flight.py
#
# Q-Learning based Crazyflie flight controller for a 3D grid
# - Uses Qualisys mocap position as feedback signal
# - Implements greedy policy based on a fake 3D Q-table
# - Waypoints are 3D grid cell centers

import logging
import time
import queue
from pathlib import Path
import numpy as np
from threading import Thread, Event
from datetime import datetime
import csv

import cflib.crtp

from cf_mocap import CrazyflieClient, QualisysClient, pose_sender
from q_learning_controller import Action3D, GridEnvironment3D, QLearningController3D

logging.basicConfig(level=logging.ERROR)

# =====================================
# USER SETTINGS
# =====================================
URI = "radio://0/80/2M/E7E7E7E7E7"
QTM_IP = "128.174.245.64"
MARKER_DECK_NAME = "Crazyflie"
MARKER_DECK_IDS = [1, 2, 3, 4]

# 3D grid environment parameters
GRID_X_MIN, GRID_X_MAX = 0.0, 2.0
GRID_Y_MIN, GRID_Y_MAX = 0.0, 2.0
GRID_Z_MIN, GRID_Z_MAX = 0.4, 1.6
GRID_SIZE = 0.4

# Flight parameters
HOVER_Z = 0.8
TAKEOFF_TIME = 3.0
LAND_TIME = 2.0
FINAL_Z = 0.10
RATE_HZ = 20

# Navigation parameters
WAYPOINT_REACH_THRESHOLD = 0.18
TIME_AT_WAYPOINT = 0.8
MAX_NAVIGATION_STEPS = 60

# Q-table file (optional, will create a fake 3D one if not found)
Q_TABLE_FILE = None
SAVE_Q_TABLE = True
SAVE_TRAJECTORY = True
DATA_DIR = Path("data")


def create_fake_3d_q_table(grid_env):
    """Create a fake 3D Q-table biased toward the far upper corner."""
    num_states = grid_env.num_states
    num_actions = len(Action3D)
    q_table = np.random.randn(num_states, num_actions) * 0.05

    for state in range(num_states):
        grid_x, grid_y, grid_z = grid_env.state_to_grid(state)
        if grid_x < grid_env.nx - 1:
            q_table[state, Action3D.RIGHT_X] = 1.0
        if grid_y < grid_env.ny - 1:
            q_table[state, Action3D.UP_Y] = 1.0
        if grid_z < grid_env.nz - 1:
            q_table[state, Action3D.UP_Z] = 1.0

        if grid_x == grid_env.nx - 1:
            q_table[state, Action3D.RIGHT_X] = -1.0
        if grid_y == grid_env.ny - 1:
            q_table[state, Action3D.UP_Y] = -1.0
        if grid_z == grid_env.nz - 1:
            q_table[state, Action3D.UP_Z] = -1.0

    return q_table


def load_or_create_q_table(grid_env, q_table_file=None):
    """Load a Q-table from disk or create a fake 3D one."""
    if q_table_file is not None:
        try:
            q_table = np.load(q_table_file)
            print(f"[FLIGHT3D] Loaded Q-table from {q_table_file}")
            return q_table
        except FileNotFoundError:
            print(f"[FLIGHT3D] Q-table file not found: {q_table_file}, creating fake 3D Q-table")
    else:
        print("[FLIGHT3D] No Q-table file specified, creating fake 3D Q-table")

    return create_fake_3d_q_table(grid_env)


def save_q_table(q_table, filename):
    """Save the generated Q-table for later inspection or visualization."""
    np.save(filename, q_table)
    print(f"[LOG] Q-table saved to {filename}")


def get_distance_to_point(current_pos, target_pos):
    """Calculate Euclidean distance between two 3D points."""
    return np.linalg.norm(np.array(current_pos[:3]) - np.array(target_pos[:3]))


def get_current_position(qtm_client, timeout=2.0):
    """Get the latest mocap position as an (x, y, z) tuple."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            pose = qtm_client.pose_queue.get(timeout=0.05)
            return pose[:3]
        except queue.Empty:
            time.sleep(0.05)

    raise TimeoutError("Timed out waiting for a valid mocap pose.")


def wait_until_near_waypoint(qtm_client, target_xyz, threshold, timeout=5.0):
    """Wait until the live mocap position is near the waypoint target."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        current_position = get_current_position(qtm_client, timeout=0.2)
        if get_distance_to_point(current_position, target_xyz) <= threshold:
            return current_position
        time.sleep(0.05)

    return get_current_position(qtm_client, timeout=0.2)


def save_position_history(position_history, filename="position_history_3d.csv"):
    """Save position history to a CSV file."""
    if not position_history:
        print("[LOG] No position history to save")
        return

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "x", "y", "z"])
        for timestamp, x, y, z in position_history:
            writer.writerow([timestamp, f"{x:.6f}", f"{y:.6f}", f"{z:.6f}"])

    print(f"[LOG] Position history saved to {filename} ({len(position_history)} samples)")


def ensure_data_dir():
    """Create the output data directory if it does not exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR


def step_grid_from_action(grid_env, current_grid, action):
    """Apply a 3D action to a grid coordinate and clamp to valid bounds."""
    next_grid_x, next_grid_y, next_grid_z = current_grid

    if action == Action3D.UP_Y:
        next_grid_y += 1
    elif action == Action3D.DOWN_Y:
        next_grid_y -= 1
    elif action == Action3D.RIGHT_X:
        next_grid_x += 1
    elif action == Action3D.LEFT_X:
        next_grid_x -= 1
    elif action == Action3D.UP_Z:
        next_grid_z += 1
    elif action == Action3D.DOWN_Z:
        next_grid_z -= 1

    next_grid_x = int(np.clip(next_grid_x, 0, grid_env.nx - 1))
    next_grid_y = int(np.clip(next_grid_y, 0, grid_env.ny - 1))
    next_grid_z = int(np.clip(next_grid_z, 0, grid_env.nz - 1))
    return next_grid_x, next_grid_y, next_grid_z


def predict_navigation_path(grid_env, controller, start_grid, target_grid, max_steps=50):
    """Predict the greedy 3D navigation path without moving the drone."""
    waypoints = [start_grid]
    actions = []
    current_grid = start_grid

    for _ in range(max_steps):
        current_state = grid_env.grid_to_state(*current_grid)
        action = controller.get_best_action(current_state)
        actions.append(Action3D(action).name)
        current_grid = step_grid_from_action(grid_env, current_grid, action)
        waypoints.append(current_grid)

        if current_grid == target_grid:
            break

    return waypoints, actions


def print_navigation_plan(grid_env, waypoints, actions):
    """Print predicted waypoints and action sequence."""
    print("\n" + "=" * 72)
    print("PREDICTED 3D NAVIGATION PLAN")
    print("=" * 72)
    print(f"\nWaypoint Sequence ({len(waypoints)} total):")
    for i, (gx, gy, gz) in enumerate(waypoints):
        x, y, z = grid_env.grid_to_continuous(gx, gy, gz)
        print(f"  {i:2d}: Grid ({gx}, {gy}, {gz}) -> Continuous ({x:.2f}, {y:.2f}, {z:.2f})")

    print(f"\nAction Sequence ({len(actions)} total):")
    for i, action_name in enumerate(actions):
        print(f"  {i:2d}: {action_name}")
    print("=" * 72 + "\n")


def wait_for_user_confirmation():
    """Wait for user confirmation before starting the drone."""
    while True:
        response = input("[MAIN] Press 'Y' to confirm and start the drone (or 'N' to abort): ").strip().upper()
        if response == "Y":
            print("[MAIN] Confirmed. Starting drone...")
            return True
        if response == "N":
            print("[MAIN] Aborted by user.")
            return False
        print("[MAIN] Invalid input. Please press 'Y' to confirm or 'N' to abort.")


def navigate_with_q_learning_3d(cf_client, qtm_client, grid_env, controller, current_position):
    """Navigate from current position to a 3D target using greedy Q-learning."""
    print("[FLIGHT3D] Starting 3D Q-learning navigation")
    print(
        f"[FLIGHT3D] Start position: "
        f"({current_position[0]:.2f}, {current_position[1]:.2f}, {current_position[2]:.2f})"
    )

    position_history = []
    position_history.append((time.time(), current_position[0], current_position[1], current_position[2]))

    start_grid = grid_env.continuous_to_grid(*current_position)
    target_grid = (grid_env.nx - 1, grid_env.ny - 1, grid_env.nz - 1)

    print(f"[FLIGHT3D] Start grid: {start_grid}")
    print(f"[FLIGHT3D] Target grid: {target_grid}")

    controller.set_start_target(start_grid, target_grid)

    for step in range(MAX_NAVIGATION_STEPS):
        current_position = get_current_position(qtm_client)
        current_grid = grid_env.continuous_to_grid(*current_position)
        current_state = grid_env.grid_to_state(*current_grid)

        print(f"\n[FLIGHT3D] Step {step}: Current grid {current_grid}, State {current_state}")
        action = controller.get_best_action(current_state)
        print(f"[FLIGHT3D] Best action: {Action3D(action).name}")

        next_grid = step_grid_from_action(grid_env, current_grid, action)
        goal_xyz = grid_env.grid_to_continuous(*next_grid)
        print(
            f"[FLIGHT3D] Moving to waypoint {next_grid} "
            f"at ({goal_xyz[0]:.2f}, {goal_xyz[1]:.2f}, {goal_xyz[2]:.2f})"
        )

        start_xyz = tuple(current_position[:3])
        distance = np.linalg.norm(np.array(goal_xyz) - np.array(start_xyz))
        travel_time = max(1.0, distance / 0.5 + 0.5)

        cf_client.go_to(start_xyz, goal_xyz, yaw_deg=0.0, duration=travel_time, rate_hz=RATE_HZ)

        reached_position = wait_until_near_waypoint(
            qtm_client,
            goal_xyz,
            threshold=WAYPOINT_REACH_THRESHOLD,
            timeout=travel_time + TIME_AT_WAYPOINT,
        )

        cf_client.hold_position(
            goal_xyz[0],
            goal_xyz[1],
            goal_xyz[2],
            yaw_deg=0.0,
            duration=TIME_AT_WAYPOINT,
            rate_hz=RATE_HZ,
        )

        current_position = reached_position
        position_history.append((time.time(), current_position[0], current_position[1], current_position[2]))

        if grid_env.continuous_to_grid(*current_position) == target_grid:
            print(f"\n[FLIGHT3D] Reached target grid in {step + 1} steps!")
            return True, position_history

    print(f"\n[FLIGHT3D] Failed to reach target within {MAX_NAVIGATION_STEPS} steps")
    return False, position_history


def main():
    """Connect to Crazyflie and QTM, then perform 3D Q-learning navigation."""
    cflib.crtp.init_drivers()

    cf_client = CrazyflieClient(URI, marker_deck_ids=MARKER_DECK_IDS)
    cf_client.wait_until_ready(timeout=10.0)

    pose_queue = queue.Queue(maxsize=1)
    pose_stop_event = Event()
    pose_thread = Thread(
        target=pose_sender,
        args=(cf_client, pose_queue, pose_stop_event),
        daemon=True,
    )
    pose_thread.start()

    qtm_client = QualisysClient(QTM_IP, MARKER_DECK_NAME, pose_queue)

    print("[MAIN] Waiting for mocap stream...")
    if not qtm_client.pose_streaming.wait(timeout=10.0):
        raise TimeoutError("No Qualisys pose stream received.")

    print("[MAIN] Letting extpose stream for estimator warm-up...")
    time.sleep(2.0)

    cf_client.reset_estimator()
    time.sleep(2.0)

    print("[MAIN] Initializing 3D Q-learning controller...")
    data_dir = ensure_data_dir()
    grid_env = GridEnvironment3D(
        x_min=GRID_X_MIN,
        x_max=GRID_X_MAX,
        y_min=GRID_Y_MIN,
        y_max=GRID_Y_MAX,
        z_min=GRID_Z_MIN,
        z_max=GRID_Z_MAX,
        grid_size=GRID_SIZE,
    )

    q_table = load_or_create_q_table(grid_env, Q_TABLE_FILE)
    controller = QLearningController3D(grid_env, q_table)

    print("\n[MAIN] 3D Q-table shape:")
    print(q_table.shape)

    initial_position = get_current_position(qtm_client)
    start_grid = grid_env.continuous_to_grid(*initial_position)
    target_grid = (grid_env.nx - 1, grid_env.ny - 1, grid_env.nz - 1)
    waypoints, actions = predict_navigation_path(grid_env, controller, start_grid, target_grid, MAX_NAVIGATION_STEPS)
    print_navigation_plan(grid_env, waypoints, actions)

    if not wait_for_user_confirmation():
        print("[MAIN] Exiting without starting drone.")
        qtm_client.close()
        pose_stop_event.set()
        pose_thread.join(timeout=1.0)
        return

    print("[MAIN] Arming")
    cf_client.cf.platform.send_arming_request(True)
    time.sleep(1.0)

    try:
        print("[MAIN] Taking off...")
        current_position = get_current_position(qtm_client)
        cf_client.go_to(
            start_xyz=current_position,
            goal_xyz=(current_position[0], current_position[1], HOVER_Z),
            yaw_deg=0.0,
            duration=TAKEOFF_TIME,
            rate_hz=RATE_HZ,
        )

        current_position = wait_until_near_waypoint(
            qtm_client,
            (current_position[0], current_position[1], HOVER_Z),
            threshold=WAYPOINT_REACH_THRESHOLD,
            timeout=TAKEOFF_TIME + 2.0,
        )

        cf_client.hold_position(
            x=current_position[0],
            y=current_position[1],
            z=HOVER_Z,
            yaw_deg=0.0,
            duration=1.0,
            rate_hz=RATE_HZ,
        )

        success, position_history = navigate_with_q_learning_3d(
            cf_client,
            qtm_client,
            grid_env,
            controller,
            get_current_position(qtm_client),
        )

        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        if SAVE_TRAJECTORY:
            trajectory_file = data_dir / f"trajectory_3d_{timestamp_str}.csv"
            save_position_history(position_history, trajectory_file)
            print(f"[MAIN] Saved 3D trajectory to {trajectory_file}")
            print(f"[MAIN] Navigation success: {success}")

        q_table_for_viz = Q_TABLE_FILE
        if SAVE_Q_TABLE and Q_TABLE_FILE is None:
            q_table_file = data_dir / f"q_table_3d_{timestamp_str}.npy"
            save_q_table(q_table, q_table_file)
            q_table_for_viz = q_table_file

        if SAVE_TRAJECTORY:
            if q_table_for_viz is not None:
                print(
                    "[MAIN] To visualize, run: "
                    f"python visualize_3d_flight.py {trajectory_file} --q-table {q_table_for_viz}"
                )
            else:
                print(
                    "[MAIN] To visualize, run: "
                    f"python visualize_3d_flight.py {trajectory_file}"
                )

        print("[MAIN] Landing...")
        current_position = get_current_position(qtm_client)
        cf_client.go_to(
            start_xyz=current_position,
            goal_xyz=(current_position[0], current_position[1], FINAL_Z),
            yaw_deg=0.0,
            duration=LAND_TIME,
            rate_hz=RATE_HZ,
        )
        time.sleep(0.5)

    except KeyboardInterrupt:
        print("[MAIN] Interrupted by user")

    finally:
        cf_client.stop()
        time.sleep(0.2)
        cf_client.disconnect()
        qtm_client.close()
        pose_stop_event.set()
        pose_thread.join(timeout=1.0)
        print("[MAIN] Done")


if __name__ == "__main__":
    main()
