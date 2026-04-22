# basic_hover_qualisys.py
#
# Simple Crazyflie + Qualisys hover script
# - Connects to Crazyflie over Crazyradio
# - Connects to QTM over network
# - Streams mocap pose into Crazyflie estimator via extpose
# - Uses built-in position controller for takeoff / hover / landing

import logging
import time
import queue
from threading import Thread, Event

import cflib.crtp

from cf_mocap import CrazyflieClient, QualisysClient, pose_sender

logging.basicConfig(level=logging.ERROR)

# =====================================
# USER SETTINGS
# =====================================
URI = "radio://0/80/2M/E7E7E7E7E7"
QTM_IP = "128.174.245.64"
# MARKER_DECK_NAME = "marker_deck_20"
MARKER_DECK_NAME = "Crazyflie"
MARKER_DECK_IDS = [1, 2, 3, 4]

HOVER_Z = 1.00
TAKEOFF_TIME = 3.0
HOVER_TIME = 5.0
LAND_TIME = 2.0
FINAL_Z = 0.10
RATE_HZ = 20


def main():
    """Main execution: connect to Crazyflie and QTM, then perform hover sequence."""
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

    print("[MAIN] Arming")
    cf_client.cf.platform.send_arming_request(True)
    time.sleep(1.0)

    try:
        # Takeoff
        cf_client.go_to(
            start_xyz=(0.0, 0.0, 0.0),
            goal_xyz=(0.0, 0.0, HOVER_Z),
            yaw_deg=0.0,
            duration=TAKEOFF_TIME,
            rate_hz=RATE_HZ,
        )

        # Hover
        cf_client.hold_position(
            x=0.0,
            y=0.0,
            z=HOVER_Z,
            yaw_deg=0.0,
            duration=HOVER_TIME,
            rate_hz=RATE_HZ,
        )

        # Land
        cf_client.go_to(
            start_xyz=(0.0, 0.0, HOVER_Z),
            goal_xyz=(0.0, 0.0, FINAL_Z),
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
