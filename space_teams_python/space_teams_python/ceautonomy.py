#!/usr/bin/env python3
#
# ceautonomy.py
#
# Autonomous waypoint navigator with depth-based obstacle avoidance
# and stuck/collision recovery for the SpaceTeamsROS rover.

import math
import time
from typing import List, Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from geometry_msgs.msg import Point, Quaternion
from sensor_msgs.msg import Image, CameraInfo, LaserScan
from space_teams_definitions.srv import String, Float

from space_teams_python.transformations import *  # noqa: F401,F403


class RoverAutoNode(Node):
    """Autonomous controller for the rover."""

    CONTROL_PERIOD = 0.1           # [s] main loop period
    WAYPOINT_TOLERANCE = 4.0       # [m] distance to consider waypoint reached
    SPEED_LIMIT_KPH = 15.0         # [km/h]

    # Depth / obstacle parameters
    MAX_OBS_RANGE = 30.0           # [m]
    MIN_OBS_RANGE = 0.2            # [m]
    OBSTACLE_CAUTION_RANGE = 10.0  # [m]
    OBSTACLE_EMERGENCY_RANGE = 4.0 # [m]
    OBSTACLE_CLEAR_RANGE = 12.0    # [m]

    # Stuck / recovery
    STUCK_SPEED_THRESH = 0.2       # [m/s]
    STUCK_TIME = 6.0               # [s] low speed window
    STUCK_PROGRESS_TIME = 5.0      # [s] no progress window
    STUCK_PROGRESS_DELTA = 0.3     # [m] required progress
    RECOVER_DURATION = 6.0         # [s] typical reverse duration
    RECOVER_MAX_TIME = 10.0        # [s] max in RECOVER

    # Steering / accel tuning
    AVOID_STEER_MAG = 0.6
    AVOID_SLOWDOWN_ACCEL = 0.3

    def __init__(self) -> None:
        super().__init__("ce_autonomy")

        self.depth_qos = qos_profile_sensor_data

        # service clients
        self.logger_client = self.create_client(String, "log_message")
        self.steer_client = self.create_client(Float, "Steer")
        self.accel_client = self.create_client(Float, "Accelerator")
        self.reverse_client = self.create_client(Float, "Reverse")
        self.brake_client = self.create_client(Float, "Brake")
        self.core_sample_client = self.create_client(Float, "CoreSample")
        self.exposure_client = self.create_client(Float, "ChangeExposure")

        # pose and velocity (local and Mars frames)
        self.loc_local: Optional[Point] = None
        self.vel_local: Optional[Point] = None
        self.rot_local: Optional[Quaternion] = None

        self.loc_mars: Optional[Point] = None
        self.vel_mars: Optional[Point] = None
        self.rot_mars: Optional[Quaternion] = None

        self.core_sampling_state = "Driving"

        # latest depth image
        self.depth_msg: Optional[Image] = None
        self.depth_info: Optional[CameraInfo] = None

        # latest obstacle distances (from depth image)
        self.front_dist: Optional[float] = None
        self.left_dist: Optional[float] = None
        self.right_dist: Optional[float] = None

        # navigation state
        self.waypoints_local: List[np.ndarray] = []
        self.visit_order: List[int] = []
        self.current_wp_index: Optional[int] = None
        self.navigation_active: bool = False

        # mode: "NAV", "AVOID", or "RECOVER"
        self.nav_mode: str = "NAV"
        self.recover_start_time: Optional[float] = None

        # Stuck detection
        self.last_moving_time: float = time.time()
        self.last_cmd_forward: bool = False
        self.last_progress_time: float = time.time()
        self.last_target_distance: Optional[float] = None

        # subscriptions
        self.create_subscription(Point, "LocationLocalFrame", self._loc_local_cb, 10)
        self.create_subscription(Point, "VelocityLocalFrame", self._vel_local_cb, 10)
        self.create_subscription(Quaternion, "RotationLocalFrame", self._rot_local_cb, 10)

        self.create_subscription(Point, "LocationMarsFrame", self._loc_mars_cb, 10)
        self.create_subscription(Point, "VelocityMarsFrame", self._vel_mars_cb, 10)
        self.create_subscription(Quaternion, "RotationMarsFrame", self._rot_mars_cb, 10)

        self.create_subscription(Point, "CoreSamplingComplete", self._core_done_cb, 1)

        self.create_subscription(Point, "WaypointsLocal", self._waypoint_cb, 10)

        # depth image and camera info
        self.create_subscription(
            Image,
            "/camera/depth/image_raw",
            self._depth_cb,
            qos_profile=self.depth_qos,
        )
        self.create_subscription(
            CameraInfo,
            "/camera/depth/camera_info",
            self._depth_info_cb,
            qos_profile=self.depth_qos,
        )

        # optional: LaserScan input if you also have depth_to_scan
        self.create_subscription(
            LaserScan,
            "/depth_scan",
            self._scan_cb,
            qos_profile_sensor_data,
        )

        self.timer = self.create_timer(self.CONTROL_PERIOD, self._control_loop)

        self.get_logger().info(
            "ce_autonomy initialised "
            "(runtime waypoints + depth-based avoidance + recovery)."
        )

    # --------------------------------------------------------------------- utils

    def _log(self, msg: str) -> None:
        self.get_logger().info(msg)
        req = String.Request()
        req.data = msg
        self.logger_client.call_async(req)

    # topic callbacks
    def _loc_local_cb(self, msg: Point) -> None:
        self.loc_local = msg

    def _vel_local_cb(self, msg: Point) -> None:
        self.vel_local = msg

    def _rot_local_cb(self, msg: Quaternion) -> None:
        self.rot_local = msg

    def _loc_mars_cb(self, msg: Point) -> None:
        self.loc_mars = msg

    def _vel_mars_cb(self, msg: Point) -> None:
        self.vel_mars = msg

    def _rot_mars_cb(self, msg: Quaternion) -> None:
        self.rot_mars = msg

    def _core_done_cb(self, _msg: Point) -> None:
        self.core_sampling_state = "Driving"

    def _depth_cb(self, msg: Image) -> None:
        self.depth_msg = msg

    def _depth_info_cb(self, msg: CameraInfo) -> None:
        self.depth_info = msg

    def _scan_cb(self, msg: LaserScan) -> None:
        # If you prefer /depth_scan over raw depth, you can use this
        # to override front/left/right directly. For now, we just ignore it
        # if using the depth image path.
        pass

    def _waypoint_cb(self, msg: Point) -> None:
        wp = np.array([float(msg.x), float(msg.y), float(msg.z)], dtype=float)
        self.waypoints_local.append(wp)
        if not self.navigation_active:
            # reset visit order if we were idle and receive new waypoints
            self.visit_order = []
        self._log(
            f"Received waypoint #{len(self.waypoints_local)}: "
            f"({wp[0]:.1f}, {wp[1]:.1f}, {wp[2]:.1f})."
        )

    # low-level command helpers
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

    # ---------------------------------------------------------------- nav helpers

    def _current_speed(self) -> float:
        if self.vel_local is None:
            return 0.0
        return math.sqrt(
            self.vel_local.x ** 2 + self.vel_local.y ** 2 + self.vel_local.z ** 2
        )

    def _local_position_vec(self) -> Optional[np.ndarray]:
        if self.loc_local is None:
            return None
        return np.array(
            [float(self.loc_local.x), float(self.loc_local.y), float(self.loc_local.z)]
        )

    def _distance_to_target(self, target: np.ndarray) -> Optional[float]:
        pos = self._local_position_vec()
        if pos is None:
            return None
        return float(np.linalg.norm(target - pos))

    def _nearest_neighbor_order(self) -> List[int]:
        if not self.waypoints_local:
            return []

        remaining = list(range(len(self.waypoints_local)))
        order: List[int] = []

        pos = self._local_position_vec()
        if pos is None:
            current_index = remaining.pop(0)
            order.append(current_index)
        else:
            dists = [
                (i, float(np.linalg.norm(self.waypoints_local[i] - pos)))
                for i in remaining
            ]
            dists.sort(key=lambda x: x[1])
            current_index = dists[0][0]
            remaining.remove(current_index)
            order.append(current_index)

        while remaining:
            last_wp = self.waypoints_local[current_index]
            dists = [
                (i, float(np.linalg.norm(self.waypoints_local[i] - last_wp)))
                for i in remaining
            ]
            dists.sort(key=lambda x: x[1])
            current_index = dists[0][0]
            remaining.remove(current_index)
            order.append(current_index)

        return order

    def _ensure_visit_order(self) -> None:
        if not self.waypoints_local:
            self.visit_order = []
            self.current_wp_index = None
            return

        if not self.visit_order:
            self.visit_order = self._nearest_neighbor_order()
            self.current_wp_index = 0
            self._log(f"Computed visit order: {self.visit_order}")

    def _current_target_wp(self) -> Optional[np.ndarray]:
        if not self.waypoints_local:
            return None
        if self.current_wp_index is None:
            return None
        if self.current_wp_index < 0 or self.current_wp_index >= len(self.visit_order):
            return None
        idx = self.visit_order[self.current_wp_index]
        return self.waypoints_local[idx]

    def _advance_waypoint_if_reached(self) -> None:
        target = self._current_target_wp()
        if target is None:
            return

        dist = self._distance_to_target(target)
        if dist is None:
            return

        if dist <= self.WAYPOINT_TOLERANCE:
            self._log(
                f"Reached waypoint index {self.visit_order[self.current_wp_index]} "
                f"(dist={dist:.1f} m)."
            )
            self.current_wp_index += 1
            if self.current_wp_index >= len(self.visit_order):
                self._log("All waypoints visited.")
                self.navigation_active = False
                self.current_wp_index = None

    # -------------------------------------------------------------- obstacle map
    # Use depth image: split into left / center / right sectors in image space

    def _update_obstacle_sectors(self) -> None:
        """
        Use the depth image to estimate left / front / right
        obstacle distances from min ranges in each sector.
        """
        self.front_dist = None
        self.left_dist = None
        self.right_dist = None

        if self.depth_msg is None:
            return

        msg = self.depth_msg
        if msg.height == 0 or msg.width == 0:
            return

        # decode depth to meters
        try:
            if msg.encoding in ("32FC1", "32FC"):
                depth = np.frombuffer(msg.data, dtype=np.float32).reshape(
                    (msg.height, msg.width)
                )
            elif msg.encoding in ("16UC1", "mono16"):
                raw = np.frombuffer(msg.data, dtype=np.uint16).reshape(
                    (msg.height, msg.width)
                )
                depth = raw.astype(np.float32) * 0.001  # mm -> m
            else:
                self.get_logger().warn(f"Unsupported depth encoding: {msg.encoding}")
                return
        except Exception:
            return

        depth = np.where(np.isfinite(depth), depth, np.inf)

        valid_mask = (
            (depth >= self.MIN_OBS_RANGE) & (depth <= self.MAX_OBS_RANGE)
        )
        if not np.any(valid_mask):
            return

        depth_valid = depth.copy()
        depth_valid[~valid_mask] = np.inf

        h, w = depth_valid.shape
        third = max(1, w // 3)

        left_region = depth_valid[:, :third]
        center_region = depth_valid[:, third:2 * third]
        right_region = depth_valid[:, 2 * third:]

        def region_min(arr: np.ndarray) -> Optional[float]:
            if arr.size == 0:
                return None
            m = float(np.min(arr))
            if not math.isfinite(m) or m < self.MIN_OBS_RANGE or m > self.MAX_OBS_RANGE:
                return None
            return m

        self.left_dist = region_min(left_region)
        self.front_dist = region_min(center_region)
        self.right_dist = region_min(right_region)

    # --------------------------------------------------------------- main loop

    def _control_loop(self) -> None:
        if self.loc_local is None or self.rot_local is None:
            self.get_logger().debug("Waiting for initial pose...")
            return

        # Ensure we actually pick and advance through waypoints
        self._ensure_visit_order()

        now = time.time()
        self._update_obstacle_sectors()
        self._update_nav_mode(now)

        if self.core_sampling_state == "Driving":
            self._do_driving(now)
        else:
            self._do_core_sampling(now)

    # -------------------------------------------------------------- nav modes

    def _update_nav_mode(self, now: float) -> None:
        front = self.front_dist

        # Stuck detection: low speed or no progress
        speed = self._current_speed()
        if speed > self.STUCK_SPEED_THRESH:
            self.last_moving_time = now

        target = self._current_target_wp()
        if target is not None:
            dist = self._distance_to_target(target)
            if dist is not None:
                if self.last_target_distance is None:
                    self.last_target_distance = dist
                    self.last_progress_time = now
                else:
                    if dist < self.last_target_distance - self.STUCK_PROGRESS_DELTA:
                        self.last_progress_time = now
                    self.last_target_distance = dist

        stuck = False
        if self.last_cmd_forward:
            no_speed = now - self.last_moving_time > self.STUCK_TIME
            no_progress = now - self.last_progress_time > self.STUCK_PROGRESS_TIME
            if no_speed or no_progress:
                stuck = True

        if self.nav_mode != "RECOVER" and stuck:
            self._log("Detected STUCK condition -> entering RECOVER mode.")
            self.nav_mode = "RECOVER"
            self.recover_start_time = now
            return

        if (
            self.nav_mode != "RECOVER"
            and front is not None
            and front < self.OBSTACLE_EMERGENCY_RANGE
        ):
            self._log(
                f"Obstacle very close ahead (front={front:.1f} m) -> RECOVER mode."
            )
            self.nav_mode = "RECOVER"
            self.recover_start_time = now
            return

        if self.nav_mode == "RECOVER":
            if self.recover_start_time is None:
                self.recover_start_time = now
            elapsed = now - self.recover_start_time
            if elapsed > self.RECOVER_MAX_TIME:
                self._log(
                    "Recovery exceeded maximum time; switching back to NAV mode."
                )
                self.nav_mode = "NAV"
            return

        if front is not None and front < self.OBSTACLE_CAUTION_RANGE:
            if self.nav_mode != "AVOID":
                self._log(
                    f"Obstacle ahead at {front:.1f} m -> entering AVOID mode."
                )
            self.nav_mode = "AVOID"
        else:
            if self.nav_mode == "AVOID":
                if front is None or front > self.OBSTACLE_CLEAR_RANGE:
                    self._log("Path clear -> returning to NAV mode.")
                    self.nav_mode = "NAV"

    # ---------------------------------------------------------- core sampling

    def _do_core_sampling(self, now: float) -> None:
        self._send_accel(0.0)
        self._send_reverse(0.0)
        self._send_brake(1.0)
        self._send_steer(0.0)

    # -------------------------------------------------------------- driving

    def _do_driving(self, now: float) -> None:
        target = self._current_target_wp()

        if target is None:
            self.navigation_active = False
            self._send_accel(0.0)
            self._send_reverse(0.0)
            self._send_brake(1.0)
            self._send_steer(0.0)
            return

        self.navigation_active = True

        pos = self._local_position_vec()
        if pos is None:
            return

        dir_vec = target - pos
        target_yaw = math.atan2(dir_vec[1], dir_vec[0])

        if self.rot_local is None:
            return
        qw = self.rot_local.w
        qx = self.rot_local.x
        qy = self.rot_local.y
        qz = self.rot_local.z

        yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))

        heading_error = target_yaw - yaw
        while heading_error > math.pi:
            heading_error -= 2.0 * math.pi
        while heading_error < -math.pi:
            heading_error += 2.0 * math.pi

        steer_cmd = self._clamp(heading_error / math.radians(45.0), -1.0, 1.0)

        if self.nav_mode == "NAV":
            accel_cmd = 1.0
            reverse_cmd = 0.0
            brake_cmd = 0.0

            if self.front_dist is not None:
                if self.front_dist < self.OBSTACLE_CAUTION_RANGE:
                    accel_cmd = max(
                        0.2,
                        self.front_dist / self.OBSTACLE_CAUTION_RANGE,
                    )

        elif self.nav_mode == "AVOID":
            left = self.left_dist if self.left_dist is not None else self.MAX_OBS_RANGE
            right = (
                self.right_dist if self.right_dist is not None else self.MAX_OBS_RANGE
            )

            if left is not None and right is not None:
                if left < right:
                    steer_bias = self.AVOID_STEER_MAG
                else:
                    steer_bias = -self.AVOID_STEER_MAG
                steer_cmd = self._clamp(steer_cmd + steer_bias, -1.0, 1.0)

            accel_cmd = self.AVOID_SLOWDOWN_ACCEL
            reverse_cmd = 0.0
            brake_cmd = 0.0

        else:  # RECOVER
            left = self.left_dist if self.left_dist is not None else self.MAX_OBS_RANGE
            right = (
                self.right_dist if self.right_dist is not None else self.MAX_OBS_RANGE
            )

            if left < right:
                steer_cmd = -self.AVOID_STEER_MAG
            else:
                steer_cmd = self.AVOID_STEER_MAG

            accel_cmd = 0.0
            reverse_cmd = 1.0
            brake_cmd = 0.0

            if self.recover_start_time is not None:
                elapsed = now - self.recover_start_time
                if elapsed > self.RECOVER_DURATION:
                    self._log("Recovery reverse duration complete -> NAV mode.")
                    self.nav_mode = "NAV"
                    self.recover_start_time = None

        self._send_steer(steer_cmd)
        self._send_accel(accel_cmd)
        self._send_reverse(reverse_cmd)
        self._send_brake(brake_cmd)

        self.last_cmd_forward = accel_cmd > 0.1

        speed = self._current_speed()
        speed_limit_mps = self.SPEED_LIMIT_KPH / 3.6
        if speed > speed_limit_mps:
            self._send_accel(0.0)
            self._send_brake(0.5)

        self._advance_waypoint_if_reached()

    # ------------------------------------------------------------------ main

def main(args=None) -> None:
    rclpy.init(args=args)
    node = RoverAutoNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
