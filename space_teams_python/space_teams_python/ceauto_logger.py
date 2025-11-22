#!/usr/bin/env python3

import math
import time
import csv
from typing import List, Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from geometry_msgs.msg import Point, Quaternion
from sensor_msgs.msg import Image
from space_teams_definitions.srv import Float
from space_teams_python.transformations import *

import cv2


class RoverDepthLoggerNode(Node):
    CONTROL_PERIOD = 0.1
    WAYPOINT_TOLERANCE = 5.0
    SPEED_LIMIT_KPH = 15.0
    INITIAL_MOVE_TIME = 10.0

    def __init__(self) -> None:
        super().__init__("ce_depth_logger")

        self.depth_qos = qos_profile_sensor_data

        self.fx = float(self.declare_parameter("fx", 600.0).value)
        self.fy = float(self.declare_parameter("fy", 600.0).value)
        self.cx = float(self.declare_parameter("cx", 320.0).value)
        self.cy = float(self.declare_parameter("cy", 240.0).value)

        self.steer_client = self.create_client(Float, "Steer")
        self.accel_client = self.create_client(Float, "Accelerator")
        self.reverse_client = self.create_client(Float, "Reverse")
        self.brake_client = self.create_client(Float, "Brake")

        self.loc_local: Optional[Point] = None
        self.vel_local: Optional[Point] = None
        self.rot_local: Optional[Quaternion] = None

        self.depth_msg: Optional[Image] = None

        self.waypoints_local: List[np.ndarray] = self._build_waypoint_list()
        self.current_wp_index: int = 0
        self.target_local: Optional[np.ndarray] = None

        self.navigation_active: bool = False
        self.initial_move_end: Optional[float] = None
        self.initial_move_done: bool = False

        self.band_start = 0.50
        self.band_end = 1.00
        self.band_step = 0.01
        if self.band_end <= self.band_start:
            self.band_fracs: List[float] = []
        else:
            n_bands = int(math.floor((self.band_end - self.band_start) / self.band_step)) + 1
            self.band_fracs = [self.band_start + i * self.band_step for i in range(n_bands)]

        self.csv_path = "depth_bands.csv"
        self.csv_file = open(self.csv_path, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        header = ["timestamp"] + [f"h_{f:.2f}" for f in self.band_fracs]
        self.csv_writer.writerow(header)
        self.csv_file.flush()

        self.create_subscription(Point, "LocationLocalFrame", self._loc_local_cb, 10)
        self.create_subscription(Point, "VelocityLocalFrame", self._vel_local_cb, 10)
        self.create_subscription(Quaternion, "RotationLocalFrame", self._rot_local_cb, 10)

        self.create_subscription(
            Image,
            "/camera/depth/image_raw",
            self._depth_cb,
            qos_profile=self.depth_qos,
        )

        self.timer = self.create_timer(self.CONTROL_PERIOD, self._control_loop)
        self.log_timer = self.create_timer(1.0, self._log_depth_bands)

        self.get_logger().info("RoverDepthLoggerNode initialised.")

    def destroy_node(self) -> None:
        try:
            self.csv_file.flush()
            self.csv_file.close()
        except Exception:
            pass
        super().destroy_node()

    def _loc_local_cb(self, msg: Point) -> None:
        self.loc_local = msg

    def _vel_local_cb(self, msg: Point) -> None:
        self.vel_local = msg

    def _rot_local_cb(self, msg: Quaternion) -> None:
        self.rot_local = msg

    def _depth_cb(self, msg: Image) -> None:
        self.depth_msg = msg

    @staticmethod
    def _clamp(v: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, v))

    def _send_steer(self, value: float) -> None:
        req = Float.Request()
        req.data = self._clamp(value, -1.0, 1.0)
        self.steer_client.call_async(req)

    def _send_accel(self, value: float) -> None:
        req = Float.Request()
        req.data = self._clamp(value, 0.0, 1.0)
        self.accel_client.call_async(req)

    def _send_reverse(self, value: float) -> None:
        req = Float.Request()
        req.data = self._clamp(value, 0.0, 1.0)
        self.reverse_client.call_async(req)

    def _send_brake(self, value: float) -> None:
        req = Float.Request()
        req.data = self._clamp(value, 0.0, 1.0)
        self.brake_client.call_async(req)

    def _build_waypoint_list(self) -> List[np.ndarray]:
        pts = [
            (-54.31019727, 191.84449903, -19.54598818),
            (111.24089259, 427.56166121, -54.81398767),
            (-349.10709106, 558.01869306, -68.71836618),
            (1281.36380015, 1647.50529027, -39.35361376),
            (654.62948546, 1186.61595725, -48.4778713),
            (-606.74433428, 332.44253661, -20.41775233),
            (1349.86835614, 1047.23075279, -46.89420337),
            (231.41034119, -858.69285702, -63.3150879),
            (45.56236659, 921.05755228, -65.76412603),
            (1960.32237043, 1423.88737415, -89.97019481),
            (1098.14343253, 1987.40560248, -45.45757708),
            (10.15805303, -752.47151722, -68.15878792),
            (1532.81368707, 1255.13690297, -48.47378546),
            (-561.74721182, 28.52558036, -29.92751284),
            (1958.28017108, 1381.24222162, -76.75680176),
            (-1025.65838348, 274.39353778, -76.31593519),
            (410.36797363, -956.93367913, -84.31272572),
            (247.67056987, 579.07900331, -75.04176954),
            (345.53461945, 1330.35839896, -73.5301525),
            (1073.3882324, 1613.84763245, -50.72357905),
        ]
        return [np.array(p, dtype=float) for p in pts]

    def _decode_depth_image(self, msg: Image) -> Optional[np.ndarray]:
        h = msg.height
        w = msg.width
        if h == 0 or w == 0:
            return None
        enc = msg.encoding.lower()
        try:
            if "32fc" in enc:
                depth = np.frombuffer(msg.data, dtype=np.float32).reshape(h, w)
            elif "16uc" in enc or "mono16" in enc:
                depth_u16 = np.frombuffer(msg.data, dtype=np.uint16).reshape(h, w)
                depth = depth_u16.astype(np.float32) * 0.001
            else:
                self.get_logger().warn(f"Unsupported depth encoding: {msg.encoding}")
                return None
        except Exception as e:
            self.get_logger().warn(f"depth decode error: {e}")
            return None
        depth = np.where(np.isfinite(depth), depth, np.inf)
        return depth

    def _log_depth_bands(self) -> None:
        if self.depth_msg is None:
            return
        depth = self._decode_depth_image(self.depth_msg)
        if depth is None:
            return
        h, w = depth.shape

        if not self.band_fracs:
            return

        u_start = int(w * (0.5 - 0.33 / 2.0))
        u_end = int(w * (0.5 + 0.33 / 2.0))
        u_start = max(0, min(u_start, w))
        u_end = max(u_start, min(u_end, w))

        band_values: List[float] = []
        for frac in self.band_fracs:
            v0 = int(frac * h)
            v1 = int((frac + self.band_step) * h)
            v0 = max(0, min(v0, h - 1))
            v1 = max(v0 + 1, min(v1, h))
            region = depth[v0:v1, u_start:u_end]
            if region.size == 0:
                band_values.append(float("nan"))
                continue
            mask = np.isfinite(region) & (region > 0.1) & (region < np.inf)
            if not np.any(mask):
                band_values.append(float("nan"))
                continue
            min_depth = float(np.min(region[mask]))
            band_values.append(min_depth)

        row = [time.time()] + band_values
        self.csv_writer.writerow(row)
        self.csv_file.flush()

    def _control_loop(self) -> None:
        if self.loc_local is None or self.rot_local is None or self.vel_local is None:
            return
        if not self.navigation_active:
            self._start_navigation()
            return
        self._do_navigation_step()

    def _start_navigation(self) -> None:
        if not self.waypoints_local:
            self.get_logger().info("No waypoints configured - staying idle.")
            return
        self.navigation_active = True
        self.current_wp_index = 0
        self.target_local = self.waypoints_local[0]
        self.initial_move_end = time.time() + self.INITIAL_MOVE_TIME
        self.initial_move_done = False
        loc = self.loc_local
        if loc is not None:
            self.get_logger().info(
                f"Starting navigation to first waypoint from ({loc.x:.1f}, {loc.y:.1f}) "
                f"towards ({self.target_local[0]:.1f}, {self.target_local[1]:.1f})."
            )
        self._send_accel(0.2)

    def _do_navigation_step(self) -> None:
        assert self.target_local is not None
        loc = self.loc_local
        vel = self.vel_local
        rot = self.rot_local
        if loc is None or vel is None or rot is None:
            return
        now = time.time()
        if not self.initial_move_done and self.initial_move_end is not None:
            if now < self.initial_move_end:
                return
            self._send_accel(0.0)
            self.initial_move_done = True

        p = np.array([float(loc.x), float(loc.y), float(loc.z)], dtype=float)
        v = np.array([float(vel.x), float(vel.y), float(vel.z)], dtype=float)
        q = Quat(float(rot.w), float(rot.x), float(rot.y), float(rot.z))
        m = q.to_matrix()
        forward = normalize(np.array([m[0, 0], m[1, 0], 0.0]))
        to_target = self.target_local - p
        distance = float(np.linalg.norm(to_target))
        tgt_dir = normalize(np.array([to_target[0], to_target[1], 0.0]))
        dot_val = float(np.clip(np.dot(forward, tgt_dir), -1.0, 1.0))
        heading_error = math.acos(dot_val)
        cross_z = np.cross(tgt_dir, forward)[2]
        if cross_z <= 0.0:
            heading_error *= -1.0
        if distance < self.WAYPOINT_TOLERANCE:
            self._handle_first_waypoint_reached(p)
            return
        speed_diff_kph = mps_to_kph(
            kph_to_mps(self.SPEED_LIMIT_KPH) - np.linalg.norm(v)
        )
        accel_factor = remap_clamp(0.0, self.SPEED_LIMIT_KPH, 0.0, 1.0, speed_diff_kph)
        brake_factor = 1.0 - remap_clamp(
            -self.SPEED_LIMIT_KPH, 0.0, 0.0, 1.0, speed_diff_kph
        )
        deadband = math.radians(3.0)
        steer_cmd = remap_clamp(
            -0.25 * math.pi, 0.25 * math.pi, -1.0, 1.0, heading_error
        )
        if abs(heading_error) < deadband:
            steer_cmd = 0.0
        steer_gain = 1.0
        final_steer = -steer_gain * steer_cmd
        accel_gain = 2.0
        accel_cmd = accel_gain * remap_clamp(
            0.0, 1.0, accel_factor, accel_factor * 0.5, abs(steer_cmd)
        )
        brake_gain = 1.0
        brake_cmd = brake_gain * brake_factor
        self._send_steer(final_steer)
        self._send_accel(accel_cmd)
        self._send_reverse(brake_cmd)
        self._send_brake(0.0)

    def _handle_first_waypoint_reached(self, pos_vec: np.ndarray) -> None:
        self._send_accel(0.0)
        self._send_steer(0.0)
        self._send_reverse(0.0)
        self._send_brake(1.0)
        self.get_logger().info(
            f"Reached first waypoint at ({pos_vec[0]:.1f}, {pos_vec[1]:.1f}). Navigation stopped."
        )
        self.navigation_active = False


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RoverDepthLoggerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
