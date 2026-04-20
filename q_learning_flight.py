# q_learning_flight.py
#
# Q-Learning based Crazyflie flight controller
# - Uses Qualisys mocap position as feedback signal
# - Implements greedy policy based on Q-table to navigate
# - Waypoints are grid cell centers
# - Once near a waypoint, uses Q-table to select next waypoint

import logging
import time
import queue
import numpy as np
from threading import Thread, Event

import cflib.crtp

from cf_mocap import CrazyflieClient, QualisysClient, pose_sender
from q_learning_controller import GridEnvironment, Action, QLearningController

logging.basicConfig(level=logging.ERROR)

# =====================================
# USER SETTINGS
# =====================================
URI = "radio://0/80/2M/E7E7E7E7E7"
QTM_IP = "128.174.245.64"
MARKER_DECK_NAME = "Crazyflie"
MARKER_DECK_IDS = [21, 22, 23, 24]

# Grid environment parameters
GRID_X_MIN, GRID_X_MAX = 0.0, 2.0
GRID_Y_MIN, GRID_Y_MAX = 0.0, 2.0
GRID_SIZE = 0.5
Z_CONSTANT = 0.8  # Constant altitude for hovering

# Flight parameters
HOVER_Z = 1.0
TAKEOFF_TIME = 3.0
LAND_TIME = 2.0
FINAL_Z = 0.10
RATE_HZ = 20

# Navigation parameters
WAYPOINT_REACH_THRESHOLD = 0.15  # Distance threshold to consider waypoint reached (meters)
TIME_AT_WAYPOINT = 1.0  # Time to hover at each waypoint (seconds)
MAX_NAVIGATION_STEPS = 50  # Max steps before giving up

# Q-table file (optional, will create dummy if not found)
Q_TABLE_FILE = None  # Set to a file path if you have a pre-trained Q-table


def create_dummy_q_table(grid_env):
    """Create a dummy Q-table biased towards moving right and up."""
    num_states = grid_env.num_states
    num_actions = len(Action)
    
    # Initialize with small random values
    q_table = np.random.randn(num_states, num_actions) * 0.1
    
    # Bias towards moving right and up (towards goal assumed at top-right)
    for state in range(num_states):
        grid_x, grid_y = grid_env.state_to_grid(state)
        if grid_x < grid_env.nx - 1:
            q_table[state, Action.RIGHT] = 1.0
        if grid_y < grid_env.ny - 1:
            q_table[state, Action.UP] = 1.0
    
    return q_table


def load_or_create_q_table(grid_env, q_table_file=None):
    """Load Q-table from file or create dummy if not found."""
    if q_table_file is not None:
        try:
            q_table = np.load(q_table_file)
            print(f"[FLIGHT] Loaded Q-table from {q_table_file}")
            return q_table
        except FileNotFoundError:
            print(f"[FLIGHT] Q-table file not found: {q_table_file}, creating dummy Q-table")
    else:
        print("[FLIGHT] No Q-table file specified, creating dummy Q-table")
    
    return create_dummy_q_table(grid_env)


def get_distance_to_point(current_pos, target_pos):
    """Calculate Euclidean distance between two 3D points."""
    current = np.array(current_pos[:3])
    target = np.array(target_pos[:3])
    return np.linalg.norm(current - target)


def predict_navigation_path(grid_env, controller, start_grid, target_grid, max_steps=50):
    """
    Predict the navigation path using greedy Q-policy without moving the drone.
    
    Args:
        grid_env: GridEnvironment instance
        controller: QLearningController instance
        start_grid: Starting (grid_x, grid_y) tuple
        target_grid: Target (grid_x, grid_y) tuple
        max_steps: Maximum number of steps to predict
    
    Returns:
        List of (grid_x, grid_y) waypoints and list of Action names
    """
    waypoints = [start_grid]
    actions = []
    
    current_grid = start_grid
    
    for step in range(max_steps):
        current_state = grid_env.grid_to_state(current_grid[0], current_grid[1])
        
        # Get best action from Q-table
        action = controller.get_best_action(current_state)
        actions.append(Action(action).name)
        
        # Compute next grid position
        next_grid_x, next_grid_y = current_grid
        if action == Action.UP:
            next_grid_y += 1
        elif action == Action.DOWN:
            next_grid_y -= 1
        elif action == Action.RIGHT:
            next_grid_x += 1
        elif action == Action.LEFT:
            next_grid_x -= 1
        
        # Clamp to valid range
        next_grid_x = np.clip(next_grid_x, 0, grid_env.nx - 1)
        next_grid_y = np.clip(next_grid_y, 0, grid_env.ny - 1)
        
        current_grid = (next_grid_x, next_grid_y)
        waypoints.append(current_grid)
        
        # Check if reached target
        if current_grid == target_grid:
            break
    
    return waypoints, actions


def print_navigation_plan(grid_env, waypoints, actions):
    """Print predicted waypoints and action sequence."""
    print("\n" + "="*60)
    print("PREDICTED NAVIGATION PLAN")
    print("="*60)
    print(f"\nWaypoint Sequence ({len(waypoints)} total):")
    for i, (gx, gy) in enumerate(waypoints):
        x, y = grid_env.grid_to_continuous(gx, gy)
        print(f"  {i:2d}: Grid ({gx}, {gy}) -> Continuous ({x:.2f}, {y:.2f})")
    
    print(f"\nAction Sequence ({len(actions)} total):")
    for i, action in enumerate(actions):
        print(f"  {i:2d}: {action}")
    
    print("="*60 + "\n")


def wait_for_user_confirmation():
    """Wait for user to press 'Y' to confirm before proceeding."""
    while True:
        response = input("[MAIN] Press 'Y' to confirm and start the drone (or 'N' to abort): ").strip().upper()
        if response == 'Y':
            print("[MAIN] Confirmed. Starting drone...")
            return True
        elif response == 'N':
            print("[MAIN] Aborted by user.")
            return False
        else:
            print("[MAIN] Invalid input. Please press 'Y' to confirm or 'N' to abort.")


def get_current_position(qtm_client, timeout=2.0):
    """Get the latest mocap position as an (x, y, z) tuple."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            pose = qtm_client.pose_queue.get(timeout=0.05)
            return pose[:3]  # Extract (x, y, z) from (x, y, z, qx, qy, qz, qw)
        except queue.Empty:
            time.sleep(0.05)

    raise TimeoutError("Timed out waiting for a valid mocap pose.")


def wait_until_near_waypoint(qtm_client, target_xyz, threshold, timeout=5.0):
    """Wait until the live mocap position is near the waypoint target."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        current_position = get_current_position(qtm_client, timeout=0.2)
        distance = get_distance_to_point(current_position, target_xyz)
        if distance <= threshold:
            return current_position
        time.sleep(0.05)

    return get_current_position(qtm_client, timeout=0.2)


def navigate_with_q_learning(cf_client, qtm_client, grid_env, controller, current_position):
    """
    Navigate from current position to target using Q-learning.
    
    Args:
        cf_client: CrazyflieClient instance
        grid_env: GridEnvironment instance
        controller: QLearningController instance
        current_position: Current (x, y, z) position tuple
    
    Returns:
        True if reached target, False otherwise
    """
    print("[FLIGHT] Starting Q-learning navigation")
    print(f"[FLIGHT] Start position: ({current_position[0]:.2f}, {current_position[1]:.2f}, {current_position[2]:.2f})")
    
    # Determine starting grid from current position
    start_grid_x, start_grid_y = grid_env.continuous_to_grid(current_position[0], current_position[1])
    
    # Define target grid (arbitrary: top-right corner, or you can parameterize this)
    target_grid_x = grid_env.nx - 1
    target_grid_y = grid_env.ny - 1
    
    print(f"[FLIGHT] Start grid: ({start_grid_x}, {start_grid_y})")
    print(f"[FLIGHT] Target grid: ({target_grid_x}, {target_grid_y})")
    
    # Set start and target in controller
    controller.set_start_target((start_grid_x, start_grid_y), (target_grid_x, target_grid_y))
    
    # Navigate step by step
    for step in range(MAX_NAVIGATION_STEPS):
        # Get current grid position from live mocap feedback
        current_position = get_current_position(qtm_client)
        current_grid_x, current_grid_y = grid_env.continuous_to_grid(current_position[0], current_position[1])
        current_state = grid_env.grid_to_state(current_grid_x, current_grid_y)
        
        print(f"\n[FLIGHT] Step {step}: Current grid ({current_grid_x}, {current_grid_y}), State {current_state}")
        
        # Get best action from Q-table (greedy policy)
        action = controller.get_best_action(current_state)
        print(f"[FLIGHT] Best action: {Action(action).name}")
        
        # Determine next grid position based on action
        next_grid_x, next_grid_y = current_grid_x, current_grid_y
        if action == Action.UP:
            next_grid_y += 1
        elif action == Action.DOWN:
            next_grid_y -= 1
        elif action == Action.RIGHT:
            next_grid_x += 1
        elif action == Action.LEFT:
            next_grid_x -= 1
        
        # Clamp to valid range
        next_grid_x = np.clip(next_grid_x, 0, grid_env.nx - 1)
        next_grid_y = np.clip(next_grid_y, 0, grid_env.ny - 1)
        
        # Get continuous coordinates of next waypoint (grid center)
        next_x, next_y = grid_env.grid_to_continuous(next_grid_x, next_grid_y)
        next_z = grid_env.z_constant
        
        print(f"[FLIGHT] Moving to waypoint ({next_grid_x}, {next_grid_y}) at ({next_x:.2f}, {next_y:.2f}, {next_z:.2f})")
        
        # Move to next waypoint
        start_xyz = (current_position[0], current_position[1], current_position[2])
        goal_xyz = (next_x, next_y, next_z)
        
        # Calculate travel time to reach waypoint (estimate based on distance and speed)
        distance = np.linalg.norm(np.array(goal_xyz) - np.array(start_xyz))
        # Assuming cruise speed of ~0.5 m/s, but add extra time for safety
        travel_time = max(1.0, distance / 0.5 + 0.5)
        
        cf_client.go_to(start_xyz, goal_xyz, yaw_deg=0.0, duration=travel_time, rate_hz=RATE_HZ)

        # Wait until the live pose is near the target waypoint before proceeding.
        reached_position = wait_until_near_waypoint(
            qtm_client,
            goal_xyz,
            threshold=WAYPOINT_REACH_THRESHOLD,
            timeout=travel_time + TIME_AT_WAYPOINT,
        )

        # Hover at waypoint using the planned setpoint, then update from live pose.
        cf_client.hold_position(next_x, next_y, next_z, yaw_deg=0.0, duration=TIME_AT_WAYPOINT, rate_hz=RATE_HZ)

        # Update current position from mocap for the next iteration.
        current_position = reached_position

        # Check if reached target grid
        current_grid_x, current_grid_y = grid_env.continuous_to_grid(current_position[0], current_position[1])
        if current_grid_x == target_grid_x and current_grid_y == target_grid_y:
            print(f"\n[FLIGHT] Reached target grid in {step + 1} steps!")
            return True
    
    print(f"\n[FLIGHT] Failed to reach target within {MAX_NAVIGATION_STEPS} steps")
    return False


def main():
    """Main execution: connect to Crazyflie and QTM, then perform Q-learning navigation."""
    cflib.crtp.init_drivers()

    # Initialize Crazyflie client
    cf_client = CrazyflieClient(URI, marker_deck_ids=MARKER_DECK_IDS)
    cf_client.wait_until_ready(timeout=10.0)

    # Start pose streaming thread
    pose_queue = queue.Queue(maxsize=1)
    pose_stop_event = Event()
    pose_thread = Thread(
        target=pose_sender,
        args=(cf_client, pose_queue, pose_stop_event),
        daemon=True,
    )
    pose_thread.start()

    # Start Qualisys client
    qtm_client = QualisysClient(QTM_IP, MARKER_DECK_NAME, pose_queue)

    print("[MAIN] Waiting for mocap stream...")
    if not qtm_client.pose_streaming.wait(timeout=10.0):
        raise TimeoutError("No Qualisys pose stream received.")

    print("[MAIN] Letting extpose stream for estimator warm-up...")
    time.sleep(2.0)

    cf_client.reset_estimator()
    time.sleep(2.0)

    # Create grid environment and Q-table controller
    print("[MAIN] Initializing Q-learning controller...")
    grid_env = GridEnvironment(
        x_min=GRID_X_MIN,
        x_max=GRID_X_MAX,
        y_min=GRID_Y_MIN,
        y_max=GRID_Y_MAX,
        grid_size=GRID_SIZE,
        z_constant=Z_CONSTANT,
    )
    
    q_table = load_or_create_q_table(grid_env, Q_TABLE_FILE)
    controller = QLearningController(grid_env, q_table)
    
    print("\n[MAIN] Q-table:")
    print(q_table)
    
    # Predict navigation path and get start/target grids from the live mocap pose.
    initial_position = get_current_position(qtm_client)
    start_grid_x, start_grid_y = grid_env.continuous_to_grid(initial_position[0], initial_position[1])
    target_grid_x = grid_env.nx - 1
    target_grid_y = grid_env.ny - 1
    start_grid = (start_grid_x, start_grid_y)
    target_grid = (target_grid_x, target_grid_y)
    
    # Predict path
    waypoints, actions = predict_navigation_path(grid_env, controller, start_grid, target_grid, MAX_NAVIGATION_STEPS)
    print_navigation_plan(grid_env, waypoints, actions)
    
    # Wait for user confirmation
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
        # Takeoff to hover altitude
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

        # Hover briefly at takeoff location
        print("[MAIN] Hovering after takeoff...")
        cf_client.hold_position(
            x=current_position[0],
            y=current_position[1],
            z=HOVER_Z,
            yaw_deg=0.0,
            duration=1.0,
            rate_hz=RATE_HZ,
        )

        # Start Q-learning navigation from current position
        current_position = get_current_position(qtm_client)
        navigate_with_q_learning(cf_client, qtm_client, grid_env, controller, current_position)

        # Land
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
        # Cleanup
        cf_client.stop()
        time.sleep(0.2)
        cf_client.disconnect()

        qtm_client.close()

        pose_stop_event.set()
        pose_thread.join(timeout=1.0)

        print("[MAIN] Done")


if __name__ == "__main__":
    main()
