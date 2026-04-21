# q_learning_controller.py
#
# Q-Learning based Crazyflie controller
# - Discretizes 3D space into 3D square grids
# - Uses pre-learned Q-table to navigate from start to target grid
# - Integrates with CrazyflieClient to send position setpoints

import numpy as np
from enum import IntEnum


class Action3D(IntEnum):
    """Discrete actions in the 3D grid environment."""
    UP_Y = 0        # +Y direction
    DOWN_Y = 1      # -Y direction
    RIGHT_X = 2     # +X direction
    LEFT_X = 3      # -X direction
    UP_Z = 4        # +Z direction
    DOWN_Z = 5      # -Z direction


class GridEnvironment3D:
    """Handles discretization of continuous 3D space into cubic grids."""

    def __init__(self, x_min, x_max, y_min, y_max, z_min, z_max, grid_size):
        """
        Initialize the 3D grid environment.

        Args:
            x_min, x_max: Range of X coordinates (meters)
            y_min, y_max: Range of Y coordinates (meters)
            z_min, z_max: Range of Z coordinates (meters)
            grid_size: Size of each cubic grid cell (meters)
        """
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.z_min = z_min
        self.z_max = z_max
        self.grid_size = grid_size

        self.nx = int(np.ceil((x_max - x_min) / grid_size))
        self.ny = int(np.ceil((y_max - y_min) / grid_size))
        self.nz = int(np.ceil((z_max - z_min) / grid_size))
        self.num_states = self.nx * self.ny * self.nz

        print(
            f"[GRID3D] Environment: {self.nx}x{self.ny}x{self.nz} grid "
            f"({self.num_states} states)"
        )
        print(
            f"[GRID3D] X range: [{x_min}, {x_max}], Y range: [{y_min}, {y_max}], "
            f"Z range: [{z_min}, {z_max}]"
        )
        print(f"[GRID3D] Grid size: {grid_size}m")

    def continuous_to_grid(self, x, y, z):
        """Convert continuous (x, y, z) coordinates to grid indices."""
        grid_x = int(np.clip((x - self.x_min) / self.grid_size, 0, self.nx - 1))
        grid_y = int(np.clip((y - self.y_min) / self.grid_size, 0, self.ny - 1))
        grid_z = int(np.clip((z - self.z_min) / self.grid_size, 0, self.nz - 1))
        return grid_x, grid_y, grid_z

    def grid_to_continuous(self, grid_x, grid_y, grid_z):
        """Convert 3D grid indices to continuous coordinates at the cell center."""
        x = self.x_min + (grid_x + 0.5) * self.grid_size
        y = self.y_min + (grid_y + 0.5) * self.grid_size
        z = self.z_min + (grid_z + 0.5) * self.grid_size
        return x, y, z

    def grid_to_state(self, grid_x, grid_y, grid_z):
        """Convert 3D grid indices to a 1D state index."""
        return grid_z * (self.nx * self.ny) + grid_y * self.nx + grid_x

    def state_to_grid(self, state):
        """Convert a 1D state index to 3D grid indices."""
        xy_plane_size = self.nx * self.ny
        grid_z = state // xy_plane_size
        plane_state = state % xy_plane_size
        grid_x = plane_state % self.nx
        grid_y = plane_state // self.nx
        return grid_x, grid_y, grid_z

    def is_valid_grid(self, grid_x, grid_y, grid_z):
        """Check if 3D grid indices are within valid range."""
        return (
            0 <= grid_x < self.nx and
            0 <= grid_y < self.ny and
            0 <= grid_z < self.nz
        )


class QLearningController3D:
    """Navigates drone using a pre-learned Q-table in a 3D grid."""

    def __init__(self, grid_env, q_table):
        """
        Initialize the 3D Q-Learning controller.

        Args:
            grid_env: GridEnvironment3D instance
            q_table: Pre-learned Q-table (numpy array of shape [num_states, 6])
        """
        self.grid_env = grid_env
        self.q_table = q_table

        expected_states = grid_env.num_states
        expected_actions = len(Action3D)
        if q_table.shape != (expected_states, expected_actions):
            raise ValueError(
                f"Q-table shape {q_table.shape} does not match "
                f"expected ({expected_states}, {expected_actions})"
            )

        self.current_state = None
        self.target_state = None
        self.path = []

    def set_start_target(self, start_grid, target_grid):
        """Set the starting and target 3D grid positions."""
        start_x, start_y, start_z = start_grid
        target_x, target_y, target_z = target_grid

        if not self.grid_env.is_valid_grid(start_x, start_y, start_z):
            raise ValueError(f"Invalid start grid: ({start_x}, {start_y}, {start_z})")
        if not self.grid_env.is_valid_grid(target_x, target_y, target_z):
            raise ValueError(f"Invalid target grid: ({target_x}, {target_y}, {target_z})")

        self.current_state = self.grid_env.grid_to_state(start_x, start_y, start_z)
        self.target_state = self.grid_env.grid_to_state(target_x, target_y, target_z)
        self.path = [self.current_state]

        print(f"[QL3D] Start: grid {start_grid} (state {self.current_state})")
        print(f"[QL3D] Target: grid {target_grid} (state {self.target_state})")

    def get_best_action(self, state=None):
        """Get the greedy 3D action for the provided state."""
        if state is None:
            state = self.current_state

        q_values = self.q_table[state]
        action = np.argmax(q_values)
        return Action3D(action)

    def step(self, epsilon=0.0):
        """Take one 3D step using the Q-learning policy."""
        if self.current_state is None:
            raise RuntimeError("Start/target not set. Call set_start_target() first.")

        if np.random.random() < epsilon:
            action = Action3D(np.random.randint(0, len(Action3D)))
        else:
            action = self.get_best_action()

        grid_x, grid_y, grid_z = self.grid_env.state_to_grid(self.current_state)

        if action == Action3D.UP_Y:
            grid_y += 1
        elif action == Action3D.DOWN_Y:
            grid_y -= 1
        elif action == Action3D.RIGHT_X:
            grid_x += 1
        elif action == Action3D.LEFT_X:
            grid_x -= 1
        elif action == Action3D.UP_Z:
            grid_z += 1
        elif action == Action3D.DOWN_Z:
            grid_z -= 1

        grid_x = np.clip(grid_x, 0, self.grid_env.nx - 1)
        grid_y = np.clip(grid_y, 0, self.grid_env.ny - 1)
        grid_z = np.clip(grid_z, 0, self.grid_env.nz - 1)

        self.current_state = self.grid_env.grid_to_state(grid_x, grid_y, grid_z)
        self.path.append(self.current_state)

        return self.current_state == self.target_state

    def get_target_setpoint(self):
        """Get the continuous target setpoint for the current state."""
        if self.current_state is None:
            raise RuntimeError("Start/target not set. Call set_start_target() first.")

        grid_x, grid_y, grid_z = self.grid_env.state_to_grid(self.current_state)
        return self.grid_env.grid_to_continuous(grid_x, grid_y, grid_z)

    def get_path_as_grids(self):
        """Return the 3D path taken as grid coordinates."""
        return [self.grid_env.state_to_grid(state) for state in self.path]

    def get_path_as_continuous(self):
        """Return the 3D path taken as continuous coordinates."""
        return [
            self.grid_env.grid_to_continuous(*self.grid_env.state_to_grid(state))
            for state in self.path
        ]
