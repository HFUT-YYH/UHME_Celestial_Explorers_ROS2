#!/usr/bin/env python3
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from nav_msgs.msg import OccupancyGrid
from cv_bridge import CvBridge

class Image_Monitor(Node):
    def __init__(self):
        super().__init__('image_monitor')
        self.rgb_topic = 'camera/image_raw'
        self.depth_topic = 'camera/depth/image_raw'
        self.rgb_subscription = self.create_subscription(
            Image,
            self.rgb_topic,
            self.rgb_callback,
            10
        )
        self.depth_subscription = self.create_subscription(
            Image,
            self.depth_topic,
            self.depth_callback,
            10
        )
        self.bridge = CvBridge()
        self.column_cost = None
        self.cost_bar_h = 40
        self.min_depth = 1.0
        self.max_depth = 1000.0
        self.grad_low = 0.01
        self.grad_high = 0.5
        self.cost_pub = self.create_publisher(
            OccupancyGrid,
            '/depth_column_costmap',
            10
        )
        self.cost_resolution = 0.05
        self.cost_origin_x = 0.0
        self.cost_origin_y = 0.0
        self.cost_frame_id = 'camera_link'

    def rgb_callback(self, msg: Image):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'RGB cv_bridge error: {e}')
            return

        h, w, _ = cv_image.shape
        if self.column_cost is not None:
            cost = self.column_cost.copy()
            n = cost.shape[0]
            if n != w:
                cost_img = cost.astype(np.float32)[None, :]
                cost_img_resized = cv2.resize(
                    cost_img,
                    (w, 1),
                    interpolation=cv2.INTER_NEAREST
                )[0]
                cost = cost_img_resized.astype(np.int8)
            valid = cost >= 0
            cost_norm = np.zeros_like(cost, dtype=np.uint8)
            cost_norm[valid] = np.clip(cost[valid], 0, 100).astype(np.uint8) * 255 // 100
            cost_line = np.tile(cost_norm[None, :], (self.cost_bar_h, 1))
            cost_color = cv2.applyColorMap(cost_line, cv2.COLORMAP_JET)
            unknown = ~valid
            if np.any(unknown):
                unknown_line = np.tile(unknown[None, :], (self.cost_bar_h, 1))
                cost_color[unknown_line] = (128, 128, 128)
            if self.cost_bar_h <= h:
                roi = cv_image[h - self.cost_bar_h:h, :, :]
                blended = cv2.addWeighted(roi, 0.5, cost_color, 0.5, 0.0)
                cv_image[h - self.cost_bar_h:h, :, :] = blended

        cv2.imshow('Camera Feed', cv_image)
        cv2.waitKey(1)

    def depth_callback(self, msg: Image):
        try:
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
        except Exception as e:
            self.get_logger().error(f'Depth cv_bridge error: {e}')
            return

        if depth_image.ndim != 2:
            self.get_logger().error('Depth image is not single channel')
            return

        d = depth_image.astype(np.float32)
        d[d <= 0.0] = np.nan
        d[d < self.min_depth] = np.nan
        d[d > self.max_depth] = np.nan

        if np.all(np.isnan(d)):
            self.column_cost = None
            return

        dz_dy = np.gradient(d, axis=0)
        g = np.abs(dz_dy)
        col_grad = np.nanmax(g, axis=0)
        nan_cols = np.isnan(col_grad)
        col_grad[nan_cols] = 0.0

        gh = self.grad_high
        gl = self.grad_low
        if gh <= gl:
            gh = gl + 1e-3
        norm = (col_grad - gl) / (gh - gl)
        norm[norm < 0.0] = 0.0
        norm[norm > 1.0] = 1.0
        cost = (norm * 100.0).astype(np.int8)
        cost[nan_cols] = -1
        self.column_cost = cost

        w = cost.shape[0]
        h = 1
        grid = OccupancyGrid()
        grid.header.stamp = msg.header.stamp
        grid.header.frame_id = self.cost_frame_id
        grid.info.resolution = self.cost_resolution
        grid.info.width = int(w)
        grid.info.height = int(h)
        grid.info.origin.position.x = self.cost_origin_x
        grid.info.origin.position.y = self.cost_origin_y
        grid.info.origin.position.z = 0.0
        grid.info.origin.orientation.w = 1.0
        grid.data = cost.reshape(-1).tolist()
        self.cost_pub.publish(grid)

        try:
            depth_vis = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03),
                cv2.COLORMAP_JET
            )
            cv2.imshow('Depth Camera', depth_vis)
            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().error(f'Depth visualize error: {e}')

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = Image_Monitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
