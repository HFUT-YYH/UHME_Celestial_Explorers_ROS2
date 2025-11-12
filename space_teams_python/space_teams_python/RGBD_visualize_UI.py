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


class Image_Monitor(Node):
    def __init__(self):
        super().__init__('image_monitor')
        self.rgb_topic   = 'camera/image_raw'
        self.depth_topic = 'camera/depth/image_raw'
        self.rgb_subscribption = self.create_subscription(Image,
                                                       self.rgb_topic,
                                                       self.rgb_callback,
                                                       10)
        self.depth_subscription = self.create_subscription(
                                    Image,
                                    '/camera/depth/image_raw',
                                    self.depth_callback,
                                    10
                                )
        
        self.bridge=CvBridge()

    def rgb_callback(self,msg):
     try:
         # Convert ROS Image message to OpenCV image
         cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
         # Process the image
         
         #self.process_image(cv_image)
         cv2.imshow('Camera Feed', cv_image)
         cv2.waitKey(1)
     except Exception as e:
         self.get_logger().error(f'Error processing image: {str(e)}')

    def depth_callback(self, msg):
        try:
            # Convert to numpy array (depth map)
            depth_image = self.bridge.imgmsg_to_cv2(msg)
            
            # You can access distance values directly from the image
            # For example, to get the distance at the center:
            height, width = depth_image.shape
            center_distance = depth_image[height//2, width//2]
            self.get_logger().info(f'Center distance: {center_distance} meters')
            
            # Visualize the depth map
            # Note: Need to normalize for visualization
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03), 
                cv2.COLORMAP_JET
            )
            cv2.imshow('Depth Camera', depth_colormap)
            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().error(f'Error processing depth image: {str(e)}')
    def destroy_node(self):
        """Clean up resources when the node is destroyed"""
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

