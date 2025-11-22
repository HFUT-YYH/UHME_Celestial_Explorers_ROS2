#!/usr/bin/env python3

import math
import time
from typing import List, Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from geometry_msgs.msg import Point, Quaternion
from sensor_msgs.msg import Image
from space_teams_definitions.srv import String, Float

from space_teams_python.transformations import *

import cv2


class RoverAutoNode(Node):
    CONTROL_PERIOD = 0.1
    WAYPOINT_TOLERANCE = 5.0
    SPEED_LIMIT_KPH = 16.0
    INITIAL_MOVE_TIME = 10.0
    OBSTACLE_STOP_RANGE = 10.0
    OBSTACLE_FOV_FRACTION = 0.25

    OBSTACLE_CAUTION_RANGE = 50.0
    OBSTACLE_EMERGENCY_RANGE = 60.0
    OBSTACLE_CLEAR_RANGE = 100.0
    REVERSE_DURATION = 3.0
    AVOID_STEER_MAG = 1.0
    AVOID_SLOWDOWN_ACCEL = 0.4

    VOXEL_SIZE = 2.0
    MAX_BOXES = 6000
    MAX_DEPTH_FOR_BOXES = 3000.0
    DOWNSAMPLE_STEP = 5

    DEPTH_EXP_A = 32440924.9297
    DEPTH_EXP_B = -19.8397
    DEPTH_EXP_C = 216.4641

    DEPTH_SAFETY_SCALE = 0.75

    STUCK_TIME_WINDOW = 5.0
    STUCK_DIST_THRESHOLD = 1.0
    UNSTUCK_REVERSE_SPEED = 0.75

    AVOID_SPEED_THRESHOLD_KPH = 12.0
    AVOID_ACCEL_INT_GAIN = 0.1
    AVOID_ACCEL_INT_MAX = 0.5

    def __init__(self) -> None:
        super().__init__("ce_autonomy")

        self.depth_qos = qos_profile_sensor_data

        self.fx = float(self.declare_parameter("fx", 600.0).value)
        self.fy = float(self.declare_parameter("fy", 600.0).value)
        self.cx = float(self.declare_parameter("cx", 320.0).value)
        self.cy = float(self.declare_parameter("cy", 240.0).value)

        self.logger_client = self.create_client(String, "log_message")
        self.steer_client = self.create_client(Float, "Steer")
        self.accel_client = self.create_client(Float, "Accelerator")
        self.reverse_client = self.create_client(Float, "Reverse")
        self.brake_client = self.create_client(Float, "Brake")
        self.core_sample_client = self.create_client(Float, "CoreSample")
        self.exposure_client = self.create_client(Float, "ChangeExposure")

        self.loc_local: Optional[Point] = None
        self.vel_local: Optional[Point] = None
        self.rot_local: Optional[Quaternion] = None

        self.loc_mars: Optional[Point] = None
        self.vel_mars: Optional[Point] = None
        self.rot_mars: Optional[Quaternion] = None

        self.core_sampling_state = "Driving"

        self.depth_msg: Optional[Image] = None

        self.front_dist: Optional[float] = None
        self.left_dist: Optional[float] = None
        self.right_dist: Optional[float] = None

        self.avoid_state: str = "NAV"
        self.reverse_start_time: Optional[float] = None

        self.waypoints_local: List[np.ndarray] = self._build_waypoint_cluster()
        self.current_wp_index: int = 0
        self.target_local: Optional[np.ndarray] = None

        self.navigation_active: bool = False
        self.initial_move_end: Optional[float] = None
        self.initial_move_done: bool = False

        self.nearest_voxel_dist: Optional[float] = None
        self.voxel_left_avg: Optional[float] = None
        self.voxel_right_avg: Optional[float] = None

        self.band_y_list: List[float] = []
        self.band_min_dist_list: List[Optional[float]] = []
        self.band_thr_list: List[float] = []

        self.stuck_check_start_time: Optional[float] = None
        self.stuck_check_start_pos_xy: Optional[np.ndarray] = None
        self.unstuck_active: bool = False
        self.unstuck_end_time: Optional[float] = None

        self.avoid_accel_int: float = 0.0

        self.create_subscription(Point, "LocationLocalFrame", self._loc_local_cb, 10)
        self.create_subscription(Point, "VelocityLocalFrame", self._vel_local_cb, 10)
        self.create_subscription(Quaternion, "RotationLocalFrame", self._rot_local_cb, 10)

        self.create_subscription(Point, "LocationMarsFrame", self._loc_mars_cb, 10)
        self.create_subscription(Point, "VelocityMarsFrame", self._vel_mars_cb, 10)
        self.create_subscription(Quaternion, "RotationMarsFrame", self._rot_mars_cb, 10)

        self.create_subscription(Point, "CoreSamplingComplete", self._core_done_cb, 1)

        self.create_subscription(
            Image,
            "/camera/depth/image_raw",
            self._depth_cb,
            qos_profile=self.depth_qos,
        )

        self.vis_pub = self.create_publisher(Image, "depth_voxels_image", 1)

        self.timer = self.create_timer(self.CONTROL_PERIOD, self._control_loop)

        self.get_logger().info("ceautonomy node initialised.")

    def _log(self, msg: str) -> None:
        self.get_logger().info(msg)
        req = String.Request()
        req.data = msg
        self.logger_client.call_async(req)

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

    def _depth_cb(self, msg: Image) -> None:
        self.depth_msg = msg
        self._build_and_publish_collision_image(msg)

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

    def _send_core_sample(self) -> None:
        self.core_sampling_state = "Sampling"
        req = Float.Request()
        req.data = 0.0
        self.core_sample_client.call_async(req)

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
    def _build_waypoint_cluster(self) -> List[np.ndarray]:
        pts = [
            (-54.31019727, 191.84449903, -19.54598818),
            (111.24089259, 427.56166121, -54.81398767),
            (247.67056987, 579.07900331, -75.04176954),
            (45.56236659, 921.05755228, -65.76412603),
            (345.53461945, 1330.35839896, -73.5301525),
            (654.62948546, 1186.61595725, -48.4778713),
            (1073.3882324, 1613.84763245, -50.72357905),
            (1281.36380015, 1647.50529027, -39.35361376),
            (1098.14343253, 1987.40560248, -45.45757708),
            (1532.81368707, 1255.13690297, -48.47378546),
            (1349.86835614, 1047.23075279, -46.89420337),
            (1958.28017108, 1381.24222162, -76.75680176),
            (1960.32237043, 1423.88737415, -89.97019481),
            (-349.10709106, 558.01869306, -68.71836618),
            (-606.74433428, 332.44253661, -20.41775233),
            (-561.74721182, 28.52558036, -29.92751284),
            (-1025.65838348, 274.39353778, -76.31593519),
            (10.15805303, -752.47151722, -68.15878792),
            (231.41034119, -858.69285702, -63.3150879),
            (410.36797363, -956.93367913, -84.31272572),
        ]
        return [np.array(p, dtype=float) for p in pts]
    
    def _front_obstacle_distance(self) -> Optional[float]:
        return None

    def _expected_depth_from_h(self, h_frac: float) -> float:
        return (
            self.DEPTH_EXP_A * math.exp(self.DEPTH_EXP_B * h_frac)
            + self.DEPTH_EXP_C
        )

    def _build_and_publish_collision_image(self, msg: Image) -> None:
        depth = self._decode_depth_image(msg)
        if depth is None:
            return
        h, w = depth.shape
        step = self.DOWNSAMPLE_STEP
        voxel_size = self.VOXEL_SIZE
        voxels = {}
        fx, fy, cx, cy = self.fx, self.fy, self.cx, self.cy

        v_start = int(h * 0.50)
        v_end = int(h * 1.00)

        half_width_frac = 0.3   
        u_start = int(w * (0.5 - half_width_frac))
        u_end   = int(w * (0.5 + half_width_frac))


        v_start = max(0, min(v_start, h))
        v_end = max(v_start, min(v_end, h))
        u_start = max(0, min(u_start, w))
        u_end = max(u_start, min(u_end, w))

        for v in range(v_start, v_end, step):
            row = depth[v, :]
            for u in range(u_start, u_end, step):
                z = float(row[u])
                if not math.isfinite(z):
                    continue
                if z <= 0.1 or z > self.MAX_DEPTH_FOR_BOXES:
                    continue
                x = (u - cx) * z / fx
                y = (v - cy) * z / fy
                ix = int(math.floor(x / voxel_size))
                iy = int(math.floor(y / voxel_size))
                iz = int(math.floor(z / voxel_size))
                key = (ix, iy, iz)
                dist = math.sqrt(x * x + y * y + z * z)
                if key not in voxels or dist < voxels[key][3]:
                    voxels[key] = (x, y, z, dist)

        if voxels:
            voxel_list = list(voxels.values())
            voxel_list.sort(key=lambda t: t[3])
            voxel_list = voxel_list[: self.MAX_BOXES]
        else:
            voxel_list = []

        if voxel_list:
            self.nearest_voxel_dist = voxel_list[0][3]
        else:
            self.nearest_voxel_dist = None

        band_start = 0.50
        band_end = 1.00
        band_step = 0.01
        if band_end <= band_start:
            n_bands = 0
            band_fracs = []
            band_mins = []
        else:
            n_bands = int(math.floor((band_end - band_start) / band_step)) + 1
            band_fracs = [band_start + i * band_step for i in range(n_bands)]
            band_mins = [math.inf] * n_bands

        left_sum = 0.0
        right_sum = 0.0
        left_count = 0
        right_count = 0
        valid_mask = np.isfinite(depth)
        if valid_mask.any():
            max_vis_depth = float(np.percentile(depth[valid_mask], 95))
        else:
            max_vis_depth = self.MAX_DEPTH_FOR_BOXES
        if max_vis_depth <= 0:
            max_vis_depth = 1.0
        depth_vis = np.clip(depth, 0.0, max_vis_depth)
        depth_vis = (depth_vis / max_vis_depth * 255.0).astype(np.uint8)
        depth_vis = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2BGR)

        for (x, y, z, dist) in voxel_list:
            if z <= 0.0:
                continue
            u = fx * x / z + cx
            v = fy * y / z + cy
            u = int(round(u))
            v = int(round(v))
            if 0 <= u < w:
                if u < w // 2:
                    left_sum += dist
                    left_count += 1
                else:
                    right_sum += dist
                    right_count += 1
            if 0 <= u < w and 0 <= v < h and n_bands > 0:
                y_frac = v / float(h)
                if band_start <= y_frac <= band_end:
                    idx = int((y_frac - band_start) / band_step)
                    if idx < 0:
                        idx = 0
                    elif idx >= n_bands:
                        idx = n_bands - 1
                    if dist < band_mins[idx]:
                        band_mins[idx] = dist
            if u < 0 or u >= w or v < 0 or v >= h:
                continue
            du = abs(fx * (voxel_size / 2.0) / z)
            dv = abs(fy * (voxel_size / 2.0) / z)
            du = max(1, int(round(du)))
            dv = max(1, int(round(dv)))
            u1 = max(0, u - du)
            u2 = min(w - 1, u + du)
            v1 = max(0, v - dv)
            v2 = min(h - 1, v + dv)
            cv2.rectangle(depth_vis, (u1, v1), (u2, v2), (0, 0, 255), 1)

        if left_count > 0:
            self.voxel_left_avg = left_sum / float(left_count)
        else:
            self.voxel_left_avg = None
        if right_count > 0:
            self.voxel_right_avg = right_sum / float(right_count)
        else:
            self.voxel_right_avg = None

        self.band_y_list = []
        self.band_min_dist_list = []
        self.band_thr_list = []

        for frac, d in zip(band_fracs, band_mins):
            if math.isfinite(d):
                band_dist = d
            else:
                band_dist = None
            expected_depth = self._expected_depth_from_h(frac)
            thr = expected_depth * self.DEPTH_SAFETY_SCALE
            self.band_y_list.append(frac)
            self.band_min_dist_list.append(band_dist)
            self.band_thr_list.append(thr)

        text = f"voxels: {len(voxel_list)}"
        cv2.putText(
            depth_vis,
            text,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        if self.nearest_voxel_dist is not None:
            text2 = f"min_dist: {self.nearest_voxel_dist:.1f} m"
        else:
            text2 = "min_dist: n/a"
        cv2.putText(
            depth_vis,
            text2,
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        text3 = f"SafetyScale: {self.DEPTH_SAFETY_SCALE:.2f}"
        cv2.putText(
            depth_vis,
            text3,
            (10, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        bands_to_show = [0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
        y0 = 105
        dy = 20
        for i, target_frac in enumerate(bands_to_show):
            if not self.band_y_list:
                break
            idx = min(
                range(len(self.band_y_list)),
                key=lambda j: abs(self.band_y_list[j] - target_frac),
            )
            d = self.band_min_dist_list[idx]
            frac = self.band_y_list[idx]
            if d is None:
                line = f"h={frac:.2f}: n/a"
            else:
                expected = self._expected_depth_from_h(frac)
                if expected > 1e-6:
                    ratio = 100.0 * d / expected
                    line = f"h={frac:.2f}: {d:.1f} m ({ratio:.0f}%)"
                else:
                    line = f"h={frac:.2f}: {d:.1f} m (exp n/a)"
            cv2.putText(
                depth_vis,
                line,
                (10, y0 + i * dy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        img_msg = Image()
        img_msg.header = msg.header
        img_msg.height = h
        img_msg.width = w
        img_msg.encoding = "bgr8"
        img_msg.is_bigendian = 0
        img_msg.step = 3 * w
        img_msg.data = depth_vis.tobytes()
        self.vis_pub.publish(img_msg)

    def _update_stuck_detector(self, now: float) -> None:
        if not self.initial_move_done:
            return
        if self.loc_local is None:
            return
        cur_xy = np.array([float(self.loc_local.x), float(self.loc_local.y)], dtype=float)
        if self.stuck_check_start_time is None or self.stuck_check_start_pos_xy is None:
            self.stuck_check_start_time = now
            self.stuck_check_start_pos_xy = cur_xy
            return
        dt = now - self.stuck_check_start_time
        if dt < self.STUCK_TIME_WINDOW:
            return
        dist = float(np.linalg.norm(cur_xy - self.stuck_check_start_pos_xy))
        if dist < self.STUCK_DIST_THRESHOLD:
            self.get_logger().info(
                f"Stuck detected (Δt={dt:.1f}s, Δd={dist:.2f}m) - reversing for {self.REVERSE_DURATION:.1f}s."
            )
            self.unstuck_active = True
            self.unstuck_end_time = now + self.REVERSE_DURATION
        self.stuck_check_start_time = now
        self.stuck_check_start_pos_xy = cur_xy

    def _control_loop(self) -> None:
        if self.loc_local is None or self.rot_local is None:
            self.get_logger().debug("Waiting for initial pose...")
            return
        now = time.time()

        if self.unstuck_active:
            if self.unstuck_end_time is not None and now < self.unstuck_end_time:
                self._send_steer(0.0)
                self._send_accel(0.0)
                self._send_brake(0.0)
                self._send_reverse(self.UNSTUCK_REVERSE_SPEED)
                return
            else:
                self.unstuck_active = False
                self._send_reverse(0.0)

        if not self.navigation_active:
            self._start_navigation()
            return

        if not self.initial_move_done and self.initial_move_end is not None:
            if now < self.initial_move_end:
                return
            self._send_accel(0.0)
            self.initial_move_done = True
            self.stuck_check_start_time = now
            if self.loc_local is not None:
                self.stuck_check_start_pos_xy = np.array(
                    [float(self.loc_local.x), float(self.loc_local.y)], dtype=float
                )

        self._update_stuck_detector(now)

        collision = False
        for d, thr in zip(self.band_min_dist_list, self.band_thr_list):
            if d is None:
                continue
            if d < thr:
                collision = True
                break
        if collision:
            if self.avoid_state != "AVOID":
                self.get_logger().info("Gradient collision detected - avoidance mode.")
            self.avoid_state = "AVOID"
        else:
            if self.avoid_state != "NAV":
                self.get_logger().info("Gradient bands safe - normal navigation.")
                self.avoid_accel_int = 0.0
            self.avoid_state = "NAV"
        mode = "AVOID" if self.avoid_state == "AVOID" else "NAV"
        self._do_navigation_step(mode=mode)

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
                f"Starting navigation from ({loc.x:.1f}, {loc.y:.1f}) "
                f"towards ({self.target_local[0]:.1f}, {self.target_local[1]:.1f})."
            )
        self._send_accel(0.2)

    def _do_navigation_step(self, mode: str = "NAV") -> None:
        assert self.target_local is not None
        loc = self.loc_local
        vel = self.vel_local
        rot = self.rot_local
        if loc is None or vel is None or rot is None:
            return
        p = np.array([float(loc.x), float(loc.y), float(loc.z)], dtype=float)
        v = np.array([float(vel.x), float(vel.y), float(vel.z)], dtype=float)
        q = Quat(float(rot.w), float(rot.x), float(rot.y), float(rot.z))
        m = q.to_matrix()
        forward = normalize(np.array([m[0, 0], m[1, 0], 0.0]))
        to_target = self.target_local - p
        distance = float(np.linalg.norm(to_target))
        self.get_logger().info(f"Distance to waypoint {self.current_wp_index+1}: {distance:.2f} m")
        tgt_dir = normalize(np.array([to_target[0], to_target[1], 0.0]))
        dot_val = float(np.clip(np.dot(forward, tgt_dir), -1.0, 1.0))
        heading_error = math.acos(dot_val)
        cross_z = np.cross(tgt_dir, forward)[2]
        if cross_z <= 0.0:
            heading_error *= -1.0
        if distance < self.WAYPOINT_TOLERANCE:
            self._handle_waypoint_reached(p)
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
        cmd_steer = final_steer
        cmd_accel = accel_cmd
        cmd_reverse = brake_cmd
        cmd_brake = 0.0
        if mode == "AVOID":
            steer_dir = 0.0
            if self.voxel_left_avg is not None and self.voxel_right_avg is not None:
                steer_dir = -1.0 if self.voxel_left_avg > self.voxel_right_avg else 1.0
            cmd_steer = self._clamp(
                cmd_steer + steer_dir * self.AVOID_STEER_MAG, -1.0, 1.0
            )
            cmd_accel = min(cmd_accel, self.AVOID_SLOWDOWN_ACCEL)
            cur_speed_kph = mps_to_kph(np.linalg.norm(v))
            if cur_speed_kph < self.AVOID_SPEED_THRESHOLD_KPH:
                self.avoid_accel_int += self.AVOID_ACCEL_INT_GAIN * self.CONTROL_PERIOD
                self.avoid_accel_int = self._clamp(
                    self.avoid_accel_int, 0.0, self.AVOID_ACCEL_INT_MAX
                )
            else:
                self.avoid_accel_int = 0.0
            cmd_accel = self._clamp(
                cmd_accel + self.avoid_accel_int, 0.0, 1.0
            )
            cmd_reverse = 0.0
            cmd_brake = 0.0
        self._send_steer(cmd_steer)
        self._send_accel(cmd_accel)
        self._send_reverse(cmd_reverse)
        self._send_brake(cmd_brake)

    def _handle_waypoint_reached(self, pos_vec: np.ndarray) -> None:
        self._send_accel(0.0)
        self._send_steer(0.0)
        self._send_reverse(0.0)
        self._send_brake(1.0)
        self.get_logger().info(
            f"Reached waypoint {self.current_wp_index + 1}/{len(self.waypoints_local)} "
            f"at ({pos_vec[0]:.1f}, {pos_vec[1]:.1f}). Starting core sample."
        )
        self._send_core_sample()
        if self.current_wp_index >= len(self.waypoints_local) - 1:
            self.navigation_active = False
            self.get_logger().info("All waypoints complete - navigation finished.")
            return
        self.current_wp_index += 1
        self.target_local = self.waypoints_local[self.current_wp_index]
        nxt = self.target_local
        self.get_logger().info(
            f"Next waypoint: {self.current_wp_index + 1}/{len(self.waypoints_local)} "
            f"at ({nxt[0]:.1f}, {nxt[1]:.1f})."
        )
        self._send_brake(0.0)
        self._send_accel(0.2)


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
