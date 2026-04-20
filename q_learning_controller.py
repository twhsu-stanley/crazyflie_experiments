# q_learning_controller.py
#
# Q-Learning based Crazyflie controller
# - Discretizes 3D space into 2D square grids
# - Uses pre-learned Q-table to navigate from start to target grid
# - Integrates with CrazyflieClient to send position setpoints

import numpy as np
from enum import IntEnum


class Action(IntEnum):
    """Discrete actions in the grid environment."""
    UP = 0      # +Y direction
    DOWN = 1    # -Y direction
    RIGHT = 2   # +X direction
    LEFT = 3    # -X direction


class GridEnvironment:
    """Handles discretization of continuous 3D space into 2D grids."""
    
    def __init__(self, x_min, x_max, y_min, y_max, grid_size, z_constant=0.0):
        """
        Initialize the grid environment.
        
        Args:
            x_min, x_max: Range of X coordinates (meters)
            y_min, y_max: Range of Y coordinates (meters)
            grid_size: Size of each square grid cell (meters)
            z_constant: Constant Z altitude for the drone (meters)
        """
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.grid_size = grid_size
        self.z_constant = z_constant
        
        # Calculate number of grid cells
        self.nx = int(np.ceil((x_max - x_min) / grid_size))
        self.ny = int(np.ceil((y_max - y_min) / grid_size))
        self.num_states = self.nx * self.ny
        
        print(f"[GRID] Environment: {self.nx}x{self.ny} grid ({self.num_states} states)")
        print(f"[GRID] X range: [{x_min}, {x_max}], Y range: [{y_min}, {y_max}]")
        print(f"[GRID] Grid size: {grid_size}m, Z constant: {z_constant}m")
    
    def continuous_to_grid(self, x, y):
        """
        Convert continuous (x, y) coordinates to grid indices.
        
        Args:
            x, y: Continuous coordinates (meters)
            
        Returns:
            grid_x, grid_y: Grid indices (integers)
        """
        grid_x = int(np.clip((x - self.x_min) / self.grid_size, 0, self.nx - 1))
        grid_y = int(np.clip((y - self.y_min) / self.grid_size, 0, self.ny - 1))
        return grid_x, grid_y
    
    def grid_to_continuous(self, grid_x, grid_y):
        """
        Convert grid indices to continuous coordinates (center of grid cell).
        
        Args:
            grid_x, grid_y: Grid indices (integers)
            
        Returns:
            x, y: Continuous coordinates (meters) at grid cell center
        """
        x = self.x_min + (grid_x + 0.5) * self.grid_size
        y = self.y_min + (grid_y + 0.5) * self.grid_size
        return x, y
    
    def grid_to_state(self, grid_x, grid_y):
        """Convert 2D grid indices to 1D state index."""
        return grid_y * self.nx + grid_x
    
    def state_to_grid(self, state):
        """Convert 1D state index to 2D grid indices."""
        grid_x = state % self.nx
        grid_y = state // self.nx
        return grid_x, grid_y
    
    def is_valid_grid(self, grid_x, grid_y):
        """Check if grid indices are within valid range."""
        return 0 <= grid_x < self.nx and 0 <= grid_y < self.ny


class QLearningController:
    """Navigates drone using pre-learned Q-table."""
    
    def __init__(self, grid_env, q_table):
        """
        Initialize the Q-Learning controller.
        
        Args:
            grid_env: GridEnvironment instance
            q_table: Pre-learned Q-table (numpy array of shape [num_states, 4])
        """
        self.grid_env = grid_env
        self.q_table = q_table
        
        # Validate Q-table dimensions
        expected_states = grid_env.num_states
        expected_actions = len(Action)
        if q_table.shape != (expected_states, expected_actions):
            raise ValueError(
                f"Q-table shape {q_table.shape} does not match "
                f"expected ({expected_states}, {expected_actions})"
            )
        
        self.current_state = None
        self.target_state = None
        self.path = []
    
    def set_start_target(self, start_grid, target_grid):
        """
        Set the starting and target grid positions.
        
        Args:
            start_grid: Tuple (grid_x, grid_y) for starting position
            target_grid: Tuple (grid_x, grid_y) for target position
        """
        start_x, start_y = start_grid
        target_x, target_y = target_grid
        
        if not self.grid_env.is_valid_grid(start_x, start_y):
            raise ValueError(f"Invalid start grid: ({start_x}, {start_y})")
        if not self.grid_env.is_valid_grid(target_x, target_y):
            raise ValueError(f"Invalid target grid: ({target_x}, {target_y})")
        
        self.current_state = self.grid_env.grid_to_state(start_x, start_y)
        self.target_state = self.grid_env.grid_to_state(target_x, target_y)
        self.path = [self.current_state]
        
        print(f"[QL] Start: grid {start_grid} (state {self.current_state})")
        print(f"[QL] Target: grid {target_grid} (state {self.target_state})")
    
    def get_best_action(self, state=None):
        """
        Get the best action using Q-table (greedy policy).
        
        Args:
            state: State to evaluate (uses current state if None)
            
        Returns:
            action: Best action (Action enum value)
        """
        if state is None:
            state = self.current_state
        
        q_values = self.q_table[state]
        action = np.argmax(q_values)
        return Action(action)
    
    def step(self, epsilon=0.0):
        """
        Take one step in the grid using Q-learning policy.
        
        Args:
            epsilon: Exploration rate for epsilon-greedy policy
            
        Returns:
            True if reached target, False otherwise
        """
        if self.current_state is None:
            raise RuntimeError("Start/target not set. Call set_start_target() first.")
        
        # Epsilon-greedy action selection
        if np.random.random() < epsilon:
            action = Action(np.random.randint(0, len(Action)))
        else:
            action = self.get_best_action()
        
        # Execute action
        grid_x, grid_y = self.grid_env.state_to_grid(self.current_state)
        
        if action == Action.UP:
            grid_y += 1
        elif action == Action.DOWN:
            grid_y -= 1
        elif action == Action.RIGHT:
            grid_x += 1
        elif action == Action.LEFT:
            grid_x -= 1
        
        # Clamp to valid range
        grid_x = np.clip(grid_x, 0, self.grid_env.nx - 1)
        grid_y = np.clip(grid_y, 0, self.grid_env.ny - 1)
        
        self.current_state = self.grid_env.grid_to_state(grid_x, grid_y)
        self.path.append(self.current_state)
        
        return self.current_state == self.target_state
    
    def get_target_setpoint(self):
        """
        Get the target position setpoint for the current state.
        
        Returns:
            Tuple (x, y, z) continuous coordinates for drone position
        """
        if self.current_state is None:
            raise RuntimeError("Start/target not set. Call set_start_target() first.")
        
        grid_x, grid_y = self.grid_env.state_to_grid(self.current_state)
        x, y = self.grid_env.grid_to_continuous(grid_x, grid_y)
        z = self.grid_env.z_constant
        
        return x, y, z
    
    def navigate_to_target(self, cf_client, max_steps=1000, duration_per_step=2.0, rate_hz=20):
        """
        Navigate from start to target using Q-table and Crazyflie client.
        
        Args:
            cf_client: CrazyflieClient instance
            max_steps: Maximum number of steps before giving up
            duration_per_step: Time to spend at each grid cell (seconds)
            rate_hz: Control rate (Hz)
        """
        if self.current_state is None:
            raise RuntimeError("Start/target not set. Call set_start_target() first.")
        
        print(f"[QL] Starting navigation (max {max_steps} steps)...")
        
        for step in range(max_steps):
            # Get current grid position
            grid_x, grid_y = self.grid_env.state_to_grid(self.current_state)
            x, y = self.grid_env.grid_to_continuous(grid_x, grid_y)
            z = self.grid_env.z_constant
            
            print(f"[QL] Step {step}: Grid ({grid_x}, {grid_y}) -> Position ({x:.2f}, {y:.2f}, {z:.2f})")
            
            # Send position setpoint and hold
            cf_client.hold_position(x, y, z, yaw_deg=0.0, duration=duration_per_step, rate_hz=rate_hz)
            
            # Take next step
            if self.step():
                print(f"[QL] Reached target in {step + 1} steps!")
                return True
        
        print(f"[QL] Failed to reach target within {max_steps} steps")
        return False
    
    def get_path_as_grids(self):
        """Return the path taken as grid coordinates."""
        grids = [self.grid_env.state_to_grid(state) for state in self.path]
        return grids
    
    def get_path_as_continuous(self):
        """Return the path taken as continuous coordinates."""
        coords = [self.grid_env.grid_to_continuous(*self.grid_env.state_to_grid(state)) 
                  for state in self.path]
        return coords


# Example usage and testing
if __name__ == "__main__":
    # Example parameters
    x_min, x_max = 0.0, 2.0
    y_min, y_max = 0.0, 2.0
    grid_size = 0.5
    z_constant = 0.5
    
    # Create grid environment
    grid_env = GridEnvironment(x_min, x_max, y_min, y_max, grid_size, z_constant)
    
    # Create a dummy Q-table (in practice, this would be loaded from training)
    num_states = grid_env.num_states
    num_actions = len(Action)
    
    # Simple Q-table: bias towards moving right and up
    q_table = np.random.randn(num_states, num_actions) * 0.1
    for state in range(num_states):
        grid_x, grid_y = grid_env.state_to_grid(state)
        if grid_x < grid_env.nx - 1:
            q_table[state, Action.RIGHT] = 1.0
        if grid_y < grid_env.ny - 1:
            q_table[state, Action.UP] = 1.0
    
    # Create controller
    controller = QLearningController(grid_env, q_table)
    
    # Print Q-table
    print("\n[DEMO] Q-Table (states x actions [UP, DOWN, RIGHT, LEFT]):")
    print(q_table)
    
    # Set start and target
    start_grid = (0, 0)
    target_grid = (3, 3)
    controller.set_start_target(start_grid, target_grid)
    
    # Simulate navigation (without actual drone)
    print("\n[DEMO] Simulating navigation path:")
    for i in range(20):
        reached = controller.step()
        grid_x, grid_y = grid_env.state_to_grid(controller.current_state)
        print(f"  Step {i}: ({grid_x}, {grid_y})", end="")
        if reached:
            print(" [TARGET REACHED]")
            break
        print()
    
    # Print final path
    print(f"\n[DEMO] Path as grids: {controller.get_path_as_grids()}")
    coords = controller.get_path_as_continuous()
    print(f"[DEMO] Path as continuous coordinates:")
    for i, (x, y, z) in enumerate([(c[0], c[1], z_constant) for c in coords]):
        print(f"  Step {i}: ({x:.2f}, {y:.2f}, {z:.2f})")
