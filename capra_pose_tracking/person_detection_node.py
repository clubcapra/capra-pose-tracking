import hashlib
import os
import sys

import keras
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.publisher import Publisher
from std_msgs.msg import String
from geometry_msgs.msg import PointStamped

from pose_detection.tools.pose_detector import PoseDetector
from zed_interfaces.msg import ObjectsStamped

class TrackingNode(Node):

    def __init__(self):
        super().__init__('tracking_node')
        
        model_path = './pose_detection/trainings/training19/model/model19.keras'
        checksum_path = './pose_detection/trainings/training19/model/weights_checksum19.txt'
        # Needs to be executed from rove root workspace (... /rove)
        assert os.path.exists(model_path), "rove needs to be ran from the root of the project"
        
        mlp = keras.saving.load_model(model_path)
        checksum_after = self.model_weights_checksum(mlp)
        

        with open(checksum_path, 'r') as f:
          checksum_before = f.read().strip()

        assert checksum_before == checksum_after, "The weights have not been loaded correctly!"
        print("The weights have been loaded correctly!")
        
        self.poseDetector = PoseDetector(mlp)
        # zed subscription
        self.subscription = self.create_subscription(
            ObjectsStamped,
            '/zed2i/zed_node/body_trk/skeletons',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning
        
        # position/state publisher
        # self.publisherPoint = self.create_publisher(PointStamped, 'tracking/point', 10)
        # self.publisherState = self.create_publisher(PointStamped, 'tracking/state', 10)
    
    def model_weights_checksum(self, model):
        weights = model.get_weights()
        weights_concat = np.concatenate([w.flatten() for w in weights])
        return hashlib.md5(weights_concat).hexdigest()
        
    def listener_callback(self, msg:ObjectsStamped):
        self.poseDetector.detect(msg.objects)
        # self.get_logger().info('I heard: "%s"' % msg.objects)
        # self.publish_data(self.publisherPoint, msg)

    def publish_data(self, publisher: Publisher, data):
        publisher.publish(data)
        self.get_logger().info('Publishing: "%s"' % data)

def main(args=None):
    rclpy.init(args=args)

    trackingNode = TrackingNode()

    rclpy.spin(trackingNode)
    trackingNode.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()