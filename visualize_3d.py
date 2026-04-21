# visualize_3d_flight.py
#
# Post-flight visualization script for 3D Q-learning flights
# - Reads trajectory CSV files
# - Loads a 3D trajectory CSV
# - Visualizes the 3D flight trajectory

import argparse
import csv
import sys

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def load_position_history(csv_filename):
    """Load position history from a CSV file."""
    position_history = []
    try:
        with open(csv_filename, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                position_history.append(
                    (
                        float(row["timestamp"]),
                        float(row["x"]),
                        float(row["y"]),
                        float(row["z"]),
                    )
                )
        print(f"[VIZ3D] Loaded {len(position_history)} positions from {csv_filename}")
    except FileNotFoundError:
        print(f"[VIZ3D] ERROR: File not found: {csv_filename}")
        sys.exit(1)
    except Exception as exc:
        print(f"[VIZ3D] ERROR: Failed to load trajectory: {exc}")
        sys.exit(1)

    return position_history


def visualize_trajectory_3d(position_history, output_filename):
    """Visualize only the 3D flight trajectory."""
    fig = plt.figure(figsize=(10, 8))
    ax_3d = fig.add_subplot(111, projection="3d")

    if position_history:
        _, traj_x, traj_y, traj_z = zip(*position_history)
        ax_3d.plot(traj_x, traj_y, traj_z, "r-", linewidth=2.5, label="Actual trajectory")
        ax_3d.scatter(traj_x[0], traj_y[0], traj_z[0], c="lime", s=80, label="Start")
        ax_3d.scatter(traj_x[-1], traj_y[-1], traj_z[-1], c="red", s=100, marker="*", label="End")

    ax_3d.set_xlabel("X (meters)")
    ax_3d.set_ylabel("Y (meters)")
    ax_3d.set_zlabel("Z (meters)")
    ax_3d.set_title("3D Flight Trajectory")
    ax_3d.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(output_filename, dpi=150)
    print(f"[VIZ3D] Visualization saved to {output_filename}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize 3D Q-learning flight trajectory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python visualize_3d_flight.py data/trajectory_3d_20260420_180100.csv
        """,
    )

    parser.add_argument("trajectory", help="CSV file with 3D position history")
    parser.add_argument("--output", default=None, help="Output visualization filename")
    args = parser.parse_args()

    position_history = load_position_history(args.trajectory)

    if args.output is None:
        output_filename = args.trajectory.replace(".csv", "_trajectory3d.png")
    else:
        output_filename = args.output

    visualize_trajectory_3d(position_history, output_filename)
    print("[VIZ3D] Done!")


if __name__ == "__main__":
    main()
