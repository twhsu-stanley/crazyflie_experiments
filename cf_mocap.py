# cf_mocap.py
#
# Crazyflie + Qualisys integration components
# - MocapTransform: Coordinate transformation from QTM to Crazyflie
# - CrazyflieClient: Crazyflie connection and control interface
# - QualisysClient: QTM mocap streaming interface
# - pose_sender: Thread function to stream mocap poses to Crazyflie

import logging
import time
import asyncio
import queue
import xml.etree.cElementTree as ET
from threading import Thread, Event

import numpy as np
import cflib.crtp
from cflib.crazyflie import Crazyflie
import qtm_rt as qtm
from scipy.spatial.transform import Rotation


class MocapTransform:
    """Transforms mocap coordinates from QTM frame to Crazyflie frame."""
    def __init__(self, d1=0.0136, d2=0.0109):
        self.R_inA_ofB = np.eye(3)
        self.p_inA_ofB = np.array([0.0, 0.0, -d1])
        self.R_inA_ofW = np.eye(3)
        self.p_inA_ofW = np.array([0.0, 0.0, -d1 - d2])

        self.initialized = False
        self.R_inW_ofQ = None
        self.p_inW_ofQ = None

    def update_and_transform(self, mocap_position, mocap_euler):
        if not np.all(np.isfinite(mocap_position)) or not np.all(np.isfinite(mocap_euler)):
            return np.full(3, np.nan), np.full(4, np.nan)

        R_inQ_ofA = Rotation.from_euler("ZYX", mocap_euler).as_matrix()
        p_inQ_ofA = np.array(mocap_position)

        if not self.initialized:
            R_inQ_ofW = R_inQ_ofA @ self.R_inA_ofW
            p_inQ_ofW = p_inQ_ofA + R_inQ_ofA @ self.p_inA_ofW

            self.R_inW_ofQ = R_inQ_ofW.T
            self.p_inW_ofQ = -R_inQ_ofW.T @ p_inQ_ofW
            self.initialized = True

        R_inW_ofB = Rotation.from_matrix(self.R_inW_ofQ @ R_inQ_ofA @ self.R_inA_ofB)
        p_inW_ofB = self.p_inW_ofQ + self.R_inW_ofQ @ (
            p_inQ_ofA + R_inQ_ofA @ self.p_inA_ofB
        )
        q_cf = R_inW_ofB.as_quat()

        return p_inW_ofB, q_cf


class CrazyflieClient:
    """Manages Crazyflie connection and control commands."""
    def __init__(self, uri, marker_deck_ids=None):
        self.uri = uri
        self.marker_deck_ids = marker_deck_ids
        self.cf = Crazyflie(rw_cache="./cache")
        self.ready = Event()

        self.cf.connected.add_callback(self._connected)
        self.cf.fully_connected.add_callback(self._fully_connected)
        self.cf.connection_failed.add_callback(self._connection_failed)
        self.cf.connection_lost.add_callback(self._connection_lost)
        self.cf.disconnected.add_callback(self._disconnected)

        print(f"[CF] Connecting to {uri}")
        self.cf.open_link(uri)

    def _connected(self, uri):
        print(f"[CF] Connected to {uri}")

    def _fully_connected(self, uri):
        print(f"[CF] Fully connected to {uri}")

        if self.marker_deck_ids is not None:
            print(f"[CF] Configuring active marker deck IDs: {self.marker_deck_ids}")
            self.cf.param.set_value("activeMarker.mode", 3)
            self.cf.param.set_value("activeMarker.front", self.marker_deck_ids[0])
            self.cf.param.set_value("activeMarker.right", self.marker_deck_ids[1])
            self.cf.param.set_value("activeMarker.back", self.marker_deck_ids[2])
            self.cf.param.set_value("activeMarker.left", self.marker_deck_ids[3])

        self.cf.param.set_value("stabilizer.estimator", 2)
        self.cf.param.set_value("stabilizer.controller", 1)

        self.ready.set()

    def _connection_failed(self, uri, msg):
        print(f"[CF] Connection failed to {uri}: {msg}")

    def _connection_lost(self, uri, msg):
        print(f"[CF] Connection lost to {uri}: {msg}")

    def _disconnected(self, uri):
        print(f"[CF] Disconnected from {uri}")

    def wait_until_ready(self, timeout=10.0):
        if not self.ready.wait(timeout=timeout):
            raise TimeoutError("Crazyflie did not become ready in time.")

    def reset_estimator(self):
        print("[CF] Resetting Kalman estimator")
        self.cf.param.set_value("kalman.resetEstimation", 1)
        time.sleep(0.1)
        self.cf.param.set_value("kalman.resetEstimation", 0)

    def hold_position(self, x, y, z, yaw_deg, duration, rate_hz=20):
        dt = 1.0 / rate_hz
        t0 = time.time()
        while time.time() - t0 < duration:
            self.cf.commander.send_position_setpoint(x, y, z, yaw_deg)
            time.sleep(dt)

    def go_to(self, start_xyz, goal_xyz, yaw_deg, duration, rate_hz=20):
        dt = 1.0 / rate_hz
        n = max(1, int(duration * rate_hz))
        start_xyz = np.array(start_xyz, dtype=float)
        goal_xyz = np.array(goal_xyz, dtype=float)

        for k in range(n):
            alpha = (k + 1) / n
            xyz = (1 - alpha) * start_xyz + alpha * goal_xyz
            self.cf.commander.send_position_setpoint(
                float(xyz[0]), float(xyz[1]), float(xyz[2]), yaw_deg
            )
            time.sleep(dt)

    def stop(self):
        print("[CF] Sending stop setpoint")
        self.cf.commander.send_stop_setpoint()
        self.cf.commander.send_notify_setpoint_stop()

    def disconnect(self):
        print("[CF] Closing link")
        self.cf.close_link()


class QualisysClient(Thread):
    """Manages QTM mocap streaming in a background thread."""
    def __init__(self, ip_address, marker_deck_name, pose_queue):
        super().__init__(daemon=True)
        self.ip_address = ip_address
        self.marker_deck_name = marker_deck_name
        self.pose_queue = pose_queue

        self.connection = None
        self.qtm_6d_labels = []
        self.stop_event = Event()
        self.pose_streaming = Event()
        self.mocap_transform = MocapTransform()

        self.start()

    def close(self):
        self.stop_event.set()
        self.join()

    def run(self):
        asyncio.run(self._life_cycle())

    async def _life_cycle(self):
        await self._connect()
        while not self.stop_event.is_set():
            await asyncio.sleep(0.2)
        await self._close()

    async def _connect(self):
        print(f"[QTM] Connecting to {self.ip_address}")
        self.connection = await qtm.connect(self.ip_address, version="1.24")
        if self.connection is None:
            raise RuntimeError("Could not connect to QTM")

        params = await self.connection.get_parameters(parameters=["6d"])
        xml = ET.fromstring(params)
        self.qtm_6d_labels = [label.text.strip() for label in xml.findall("*/Body/Name")]

        print(f"[QTM] Bodies: {self.qtm_6d_labels}")

        await self.connection.stream_frames(
            components=["6d"],
            on_packet=self._on_packet,
        )

    def _on_packet(self, packet):
        header, bodies = packet.get_6d()
        if bodies is None:
            return

        if self.marker_deck_name not in self.qtm_6d_labels:
            return

        idx = self.qtm_6d_labels.index(self.marker_deck_name)
        position, orientation = bodies[idx]

        x, y, z = np.array(position) / 1e3
        R = Rotation.from_matrix(np.reshape(orientation.matrix, (3, 3), order="F"))
        yaw, pitch, roll = R.as_euler("ZYX", degrees=False)

        pos_cf, quat_cf = self.mocap_transform.update_and_transform(
            [x, y, z],
            [yaw, pitch, roll],
        )

        if np.all(np.isfinite(pos_cf)) and np.all(np.isfinite(quat_cf)):
            self.pose_streaming.set()
            pose = (
                float(pos_cf[0]),
                float(pos_cf[1]),
                float(pos_cf[2]),
                float(quat_cf[0]),
                float(quat_cf[1]),
                float(quat_cf[2]),
                float(quat_cf[3]),
            )

            try:
                if self.pose_queue.full():
                    self.pose_queue.get_nowait()
            except queue.Empty:
                pass

            try:
                self.pose_queue.put_nowait(pose)
            except queue.Full:
                pass

    async def _close(self):
        print("[QTM] Stopping stream")
        if self.connection is not None:
            await self.connection.stream_frames_stop()
            self.connection.disconnect()


def pose_sender(cf_client, pose_queue, stop_event):
    """Thread function that streams mocap poses to the Crazyflie."""
    print("[POSE] Pose sender started")
    while not stop_event.is_set():
        try:
            pose = pose_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        x, y, z, qx, qy, qz, qw = pose
        cf_client.cf.extpos.send_extpose(x, y, z, qx, qy, qz, qw)

    print("[POSE] Pose sender stopped")
