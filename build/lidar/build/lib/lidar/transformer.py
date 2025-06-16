#!/usr/bin/env python3

import numpy as np
from rclpy import qos
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, PointCloud2, PointField
from sensor_msgs_py.point_cloud2 import create_cloud
from geometry_msgs.msg import PoseStamped
from transforms3d.euler import quat2euler



PC2FIELDS = [
    PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
    PointField(name="cluster", offset=16, datatype=PointField.FLOAT32, count=1),
]


class Transformer(Node):
    def __init__(self):
        super().__init__("transformer")
        lidar_qos = qos.QoSProfile(depth=10, reliability=qos.QoSReliabilityPolicy.BEST_EFFORT)
        
        #---Publisher---#
        self.pub = self.create_publisher(PointCloud2, "points", 10)

        #---Subscriber---#
        self.sub = self.create_subscription(LaserScan, "scan", self.callback, lidar_qos)
        self.pose_sub = self.create_subscription(PoseStamped, "/pose_enco", self.pose_callback, 10) # subscribe à odométrie
        
        # Stocker la première pose pour l'initialisation
        self.initial_pose = None  
        self.last_pose = None

    def pose_callback(self, msg):
        """Initialise le repère de base comme repère global

        :param points: Array of [[x0, y0], [x1, y1], ...]
        """
        self.last_pose = msg.pose
        self.last_pose = msg.pose

    def callback(self, msg: LaserScan):
        xy = []
        intensities = []
        if self.last_pose is None:
            return 

        for i, theta in enumerate(np.arange(msg.angle_min, msg.angle_max, msg.angle_increment)):
            # Remove points too close
            if msg.ranges[i]>0.10:
                intensities.append(msg.ranges[i])
    
                # Polar to Cartesian transformation
                xyi = (msg.ranges[i]*np.cos(theta), msg.ranges[i]*np.sin(theta))
                xy.append(xyi)
        

        ### Passage au repère global ###
        # position et orientation de la dernière pose
        x = self.last_pose.position.x
        y = self.last_pose.position.y
        q = [self.last_pose.orientation.w, self.last_pose.orientation.x, self.last_pose.orientation.y, self.last_pose.orientation.z]  

        # Conversion des quaternions en angle de Euleur
        _,_,th = quat2euler(q)  

        # Matrice de transformation
        T = np.array([
            [np.cos(th), -np.sin(th), x],
            [np.sin(th),  np.cos(th), y],
            [0, 0, 1]
        ])

        # Passage en coordonnées homogènes
        xy = np.array(xy)
        ones = np.ones((len(xy), 1))
        xy_hom = np.hstack((xy, ones))
        xy_globals_hom = (T @ xy_hom.T).T  # Transpose
        xy_globals = xy_globals_hom[:, :2]  

        zeros = np.zeros((len(xy), 1))
        intensities = np.reshape(intensities, (len(intensities), 1))
        points = np.hstack((xy_globals, zeros, intensities, zeros))
        
        self.pub.publish(create_cloud(msg.header, PC2FIELDS, points))


def main(args=None):
    import rclpy
    print("Starting Transformer Node")
    rclpy.init(args=args)
    node = Transformer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
