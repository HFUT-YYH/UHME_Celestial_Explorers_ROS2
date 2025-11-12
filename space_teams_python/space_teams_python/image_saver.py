import os
import cv2
import json
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_srvs.srv import Trigger
import zmq
import time
import socket
import subprocess
from typing import Optional
from dataclasses import dataclass

@dataclass
class TopicConfig:
    name: str
    qos_history_depth: int = 5
    qos_reliability: int = rclpy.qos.ReliabilityPolicy.BEST_EFFORT
    qos_durability: int = rclpy.qos.DurabilityPolicy.VOLATILE

class ImageSubscriber(Node):
    def __init__(self, rgb_topic: Optional[TopicConfig] = None, depth_topic: Optional[TopicConfig] = None):
        super().__init__('image_client')
        self.bridge = CvBridge()
        self._rgb_topic_config = rgb_topic or TopicConfig(name='camera/image_raw')
        self._depth_topic_config = depth_topic or TopicConfig(name='camera/depth/image_raw')
        self.declare_parameter('save_dir', os.path.expanduser('~/rgbd_saves'))
        self.declare_parameter('depth_min', 1.0)
        self.declare_parameter('depth_max', 30000.0)
        self.save_dir = self.get_parameter('save_dir').value
        self.depth_min = float(self.get_parameter('depth_min').value)
        self.depth_max = float(self.get_parameter('depth_max').value)
        os.makedirs(self.save_dir, exist_ok=True)
        self.connected = False
        self.frames_count = {'RGB': 0, 'DEPTH': 0}
        self.last_time = {'RGB': time.time(), 'DEPTH': time.time()}
        self.my_publishers = {}
        self.latest_rgb = None
        self.latest_depth = None
        self.latest_depth_encoding = None
        self._setup_zmq_subscriber()
        self._setup_publishers()
        self.srv = self.create_service(Trigger, '/save_rgbd', self.handle_save)
        self.timer = self.create_timer(0.0001, self.timer_callback)

    def _setup_zmq_subscriber(self):
        self._zmq_context = zmq.Context()
        self._zmq_socket = self._zmq_context.socket(zmq.SUB)
        self._zmq_socket.setsockopt(zmq.CONFLATE, 1)
        self._zmq_socket.setsockopt(zmq.SUBSCRIBE, b"")
        ip = self._get_windows_ip()
        if not ip:
            self.get_logger().error("no windows ip")
            return
        try:
            self._zmq_socket.connect(f"tcp://{ip}:55556")
            self.connected = True
        except Exception as e:
            self.get_logger().error(str(e))

    def _get_windows_ip(self) -> str:
        try:
            out = subprocess.check_output(["/sbin/ip", "route"], text=True)
            for line in out.splitlines():
                if line.startswith("default "):
                    return line.split()[2]
        except Exception:
            pass
        try:
            return socket.gethostbyname("host.docker.internal")
        except Exception:
            return ""

    def _setup_publishers(self):
        self.my_publishers['RGB'] = self.create_publisher(Image, self._rgb_topic_config.name, 10)
        self.my_publishers['DEPTH'] = self.create_publisher(Image, self._depth_topic_config.name, 10)

    def _process_frame(self, frame_data: bytes):
        try:
            hs = [i for i in range(min(50, len(frame_data))) if frame_data[i:i+1] == b'#']
            if len(hs) < 4:
                return
            header_end = hs[3]
            parts = frame_data[:header_end].decode('ascii').split('#')
            ftype = parts[0]
            h, w, c = int(parts[1]), int(parts[2]), int(parts[3])
            img_data = frame_data[header_end + 1:]
            
            if ftype == "RGB":
                arr = np.frombuffer(img_data, dtype=np.uint8).reshape((h, w, c))
                
                msg = self.bridge.cv2_to_imgmsg(arr, encoding='rgb8')
                msg.header.stamp = self.get_clock().now().to_msg()
                self.my_publishers['RGB'].publish(msg)
                self.latest_rgb = arr
            elif ftype == "DEPTH":
                arr = np.frombuffer(img_data, dtype=np.float32).reshape((h, w, c))
                if c == 1:
                    arr = arr
                else:
                    arr = arr[:, :, 0]
                msg = self.bridge.cv2_to_imgmsg(arr, encoding='32FC1')
                msg.header.stamp = self.get_clock().now().to_msg()
                self.my_publishers['DEPTH'].publish(msg)
                self.latest_depth = arr
                self.latest_depth_encoding = '32FC1'
        except Exception as e:
            self.get_logger().error(f"process_frame: {e}")

    def timer_callback(self):
        if not self.connected:
            return
        try:
            for _ in range(100):
                try:
                    data = self._zmq_socket.recv(flags=zmq.NOBLOCK)
                    self._process_frame(data)
                except zmq.Again:
                    return
        except Exception as e:
            self.get_logger().error(f"timer: {e}")

    def handle_save(self, request, response):
        if self.latest_rgb is None or self.latest_depth is None:
            response.success = False
            response.message = 'no rgb or depth'
            return response
        now = self.get_clock().now().to_msg()
        stamp = f'{now.sec:010d}_{now.nanosec:09d}'
        rgb_path = os.path.join(self.save_dir, f'rgb_{stamp}.png')
        cv2.imwrite(rgb_path, cv2.cvtColor(self.latest_rgb, cv2.COLOR_RGB2BGR))
        depth = self.latest_depth.astype(np.float32)
        invalid = ~np.isfinite(depth) | (depth <= 0.0)
        depth_viz = depth.copy()
        np.clip(depth_viz, self.depth_min, self.depth_max, out=depth_viz)
        den = max(1e-6, (self.depth_max - self.depth_min))
        norm = (depth_viz - self.depth_min) / den
        norm[invalid] = 0.0
        norm_u8 = (norm * 255.0).astype(np.uint8)
        depth_color = cv2.applyColorMap(norm_u8, cv2.COLORMAP_JET)
        depth_color_path = os.path.join(self.save_dir, f'depth_color_{stamp}.png')
        cv2.imwrite(depth_color_path, depth_color)
        depth_npy_path = os.path.join(self.save_dir, f'depth_raw_{stamp}.npy')
        np.save(depth_npy_path, self.latest_depth)
        depth_mm = depth.copy()
        depth_mm[invalid] = 0.0
        depth_mm[~invalid] = depth_mm[~invalid] * 1000.0
        np.clip(depth_mm, 0.0, 65535.0, out=depth_mm)
        depth_u16 = depth_mm.astype(np.uint16)
        depth_png_path = os.path.join(self.save_dir, f'depth_raw_{stamp}.png')
        cv2.imwrite(depth_png_path, depth_u16)
        finite_mask = np.isfinite(depth) & (depth > 0.0)
        meta = {
            "stamp": stamp,
            "depth_encoding": str(self.latest_depth_encoding),
            "depth_dtype": str(self.latest_depth.dtype),
            "rgb_file": os.path.basename(rgb_path),
            "colorized_file": os.path.basename(depth_color_path),
            "npy_file": os.path.basename(depth_npy_path),
            "png_file_mm": os.path.basename(depth_png_path),
            "visual_clip_min_m": self.depth_min,
            "visual_clip_max_m": self.depth_max,
            "stats": {
                "shape": list(depth.shape),
                "count_total": int(depth.size),
                "count_finite_positive": int(np.count_nonzero(finite_mask)),
                "count_nan": int(np.count_nonzero(np.isnan(depth))),
                "count_inf": int(np.count_nonzero(~np.isfinite(depth) & ~np.isnan(depth))),
                "min_m": float(np.min(depth[finite_mask])) if np.any(finite_mask) else None,
                "max_m": float(np.max(depth[finite_mask])) if np.any(finite_mask) else None,
                "mean_m": float(np.mean(depth[finite_mask])) if np.any(finite_mask) else None,
                "std_m": float(np.std(depth[finite_mask])) if np.any(finite_mask) else None
            }
        }
        meta_path = os.path.join(self.save_dir, f'depth_meta_{stamp}.json')
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        response.success = True
        response.message = f"Saved {os.path.basename(rgb_path)}, {os.path.basename(depth_color_path)}, {os.path.basename(depth_npy_path)}, {os.path.basename(depth_png_path)}, {os.path.basename(meta_path)}"
        self.get_logger().info(response.message)
        return response

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = ImageSubscriber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
