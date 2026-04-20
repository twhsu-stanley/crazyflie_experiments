# visualize_flight.py
#
# Post-flight visualization script
# - Reads trajectory CSV files
# - Loads Q-table and grid parameters
# - Visualizes 2D value function with trajectory overlay

import numpy as np
import matplotlib.pyplot as plt
import csv
import argparse
import sys

from q_learning_controller import GridEnvironment, QLearningController


def load_position_history(csv_filename):
    """
    Load position history from CSV file.
    
    Args:
        csv_filename: CSV filename with columns [timestamp, x, y, z]
    
    Returns:
        List of (timestamp, x, y, z) tuples
    """
    position_history = []
    try:
        with open(csv_filename, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                timestamp = float(row['timestamp'])
                x = float(row['x'])
                y = float(row['y'])
                z = float(row['z'])
                position_history.append((timestamp, x, y, z))
        print(f"[VIZ] Loaded {len(position_history)} positions from {csv_filename}")
    except FileNotFoundError:
        print(f"[VIZ] ERROR: File not found: {csv_filename}")
        sys.exit(1)
    except Exception as e:
        print(f"[VIZ] ERROR: Failed to load trajectory: {e}")
        sys.exit(1)
    
    return position_history


def load_q_table(q_table_filename):
    """
    Load Q-table from NumPy file.
    
    Args:
        q_table_filename: Path to .npy file
    
    Returns:
        Q-table numpy array
    """
    try:
        q_table = np.load(q_table_filename)
        print(f"[VIZ] Loaded Q-table from {q_table_filename}, shape: {q_table.shape}")
        return q_table
    except FileNotFoundError:
        print(f"[VIZ] ERROR: Q-table file not found: {q_table_filename}")
        sys.exit(1)
    except Exception as e:
        print(f"[VIZ] ERROR: Failed to load Q-table: {e}")
        sys.exit(1)


def visualize_value_function(grid_env, q_table, position_history, output_filename="value_function_viz.png"):
    """
    Visualize the 2D value function (max Q-values per grid cell) with trajectory overlay.
    
    Args:
        grid_env: GridEnvironment instance
        q_table: Q-table numpy array (num_states x num_actions)
        position_history: List of (timestamp, x, y, z) tuples
        output_filename: Output filename for the visualization
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Extract value function (max Q-value per state)
    value_function = np.max(q_table, axis=1)
    value_grid = value_function.reshape(grid_env.ny, grid_env.nx)
    
    # Display value function as heatmap
    im = ax.imshow(value_grid, cmap='viridis', origin='lower', 
                   extent=[grid_env.x_min, grid_env.x_max, grid_env.y_min, grid_env.y_max])
    
    # Add grid lines
    for i in np.arange(grid_env.x_min, grid_env.x_max + grid_env.grid_size, grid_env.grid_size):
        ax.axvline(x=i, color='white', linewidth=0.5, alpha=0.3)
    for j in np.arange(grid_env.y_min, grid_env.y_max + grid_env.grid_size, grid_env.grid_size):
        ax.axhline(y=j, color='white', linewidth=0.5, alpha=0.3)
    
    # Plot actual trajectory
    if position_history:
        timestamps, xs, ys, zs = zip(*[(t, x, y, z) for t, x, y, z in position_history])
        ax.plot(xs, ys, 'r-', linewidth=2, label='Actual trajectory', alpha=0.7)
        ax.plot(xs[0], ys[0], 'go', markersize=10, label='Start', zorder=5)
        ax.plot(xs[-1], ys[-1], 'r*', markersize=15, label='End', zorder=5)
    
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_title('2D Value Function (Max Q-values) with Trajectory')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Max Q-value')
    ax.legend(loc='upper left')
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(output_filename, dpi=150)
    print(f"[VIZ] Visualization saved to {output_filename}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize Q-learning flight trajectory with value function",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python visualize_flight.py trajectory_20260420_153022.csv
  python visualize_flight.py trajectory_20260420_153022.csv --q-table q_table.npy --output my_viz.png
        """
    )
    
    parser.add_argument('trajectory', help='CSV file with position history (from flight)')
    parser.add_argument('--q-table', default=None, help='Q-table .npy file (optional, uses dummy if not provided)')
    parser.add_argument('--grid-x-min', type=float, default=0.0, help='Grid X minimum')
    parser.add_argument('--grid-x-max', type=float, default=2.0, help='Grid X maximum')
    parser.add_argument('--grid-y-min', type=float, default=0.0, help='Grid Y minimum')
    parser.add_argument('--grid-y-max', type=float, default=2.0, help='Grid Y maximum')
    parser.add_argument('--grid-size', type=float, default=0.5, help='Grid cell size')
    parser.add_argument('--z-constant', type=float, default=0.8, help='Constant Z altitude')
    parser.add_argument('--output', default=None, help='Output filename (default: value_function_viz.png)')
    
    args = parser.parse_args()
    
    # Create grid environment
    grid_env = GridEnvironment(
        x_min=args.grid_x_min,
        x_max=args.grid_x_max,
        y_min=args.grid_y_min,
        y_max=args.grid_y_max,
        grid_size=args.grid_size,
        z_constant=args.z_constant,
    )
    
    # Load trajectory
    position_history = load_position_history(args.trajectory)
    
    # Load or create Q-table
    if args.q_table is not None:
        q_table = load_q_table(args.q_table)
    else:
        # Create dummy Q-table biased towards top-right
        print("[VIZ] No Q-table provided, using dummy Q-table")
        num_states = grid_env.num_states
        num_actions = 4  # UP, DOWN, RIGHT, LEFT
        q_table = np.random.randn(num_states, num_actions) * 0.1
        
        for state in range(num_states):
            grid_x, grid_y = grid_env.state_to_grid(state)
            if grid_x < grid_env.nx - 1:
                q_table[state, 2] = 1.0  # RIGHT
            if grid_y < grid_env.ny - 1:
                q_table[state, 0] = 1.0  # UP
    
    # Determine output filename
    if args.output is None:
        base = args.trajectory.replace('.csv', '')
        output_filename = f"{base}_valuefunction.png"
    else:
        output_filename = args.output
    
    # Visualize
    visualize_value_function(grid_env, q_table, position_history, output_filename)
    print("[VIZ] Done!")


if __name__ == "__main__":
    main()
