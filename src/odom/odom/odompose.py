#!/usr/bin/env python3

from builtin_interfaces.msg import Time
from geometry_msgs.msg import PoseStamped
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, MagneticField
from turtlebot3_msgs.msg import SensorState
from transforms3d.euler import euler2quat
from nav_msgs.msg import Path # pour pouvoir tracer les poses


class Odom2Pose(Node):
    # Constants
    ENCODER_RESOLUTION = 4096
    WHEEL_RADIUS = 0.033
    WHEEL_SEPARATION = 0.160
    MAG_OFFSET = np.pi / 2.0 - 0.07

    def __init__(self):
        super().__init__("odom_to_pose")

        # Variables
        self.x_odom, self.y_odom, self.O_odom = 0.0, 0.0, 0.0
        self.prev_left_encoder = 0.0
        self.prev_right_encoder = 0.0
        self.v = 0.0

        # Publishers
        self.pub_enco = self.create_publisher(PoseStamped, "/pose_enco", 10)
        self.pub_path = self.create_publisher(Path, "/pose_path", 10)

        # Subscribers

        self.sub_enco = self.create_subscription(
            SensorState, "/sensor_state", self.callback_enco, 10
        )

        #tracer les poses successives
        
        self.path = Path() #permet de réinitialiser à chaque lancement
        self.path.header.frame_id = "odom"

    @staticmethod
    def coordinates_to_message(x: float, y: float, O: float, t: Time) -> PoseStamped:
        msg = PoseStamped()
        msg.header.stamp = t
        msg.header.frame_id = "odom"
        msg.pose.position.x = x
        msg.pose.position.y = y
        [
            msg.pose.orientation.w,
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
        ] = euler2quat(0.0, 0.0, O)
        return msg

    def dt_from_stamp(self, stamp: Time, field: str) -> float:
        t = stamp.sec + stamp.nanosec / 1e9
        dt = t - getattr(self, field) if hasattr(self, field) else 0.0
        setattr(self, field, t)
        return dt

    def callback_enco(self, sensor_state: SensorState):
        # Compute the differential in encoder count
        dq_left = sensor_state.left_encoder - self.prev_left_encoder
        dq_right = sensor_state.right_encoder - self.prev_right_encoder
        self.prev_left_encoder = sensor_state.left_encoder
        self.prev_right_encoder = sensor_state.right_encoder

        dt = self.dt_from_stamp(sensor_state.header.stamp, "prev_enco_t")
        if dt <= 0:
            return

        # TODO: Compute the linear and angular velocity (self.v and w)
        angle_left = 2 * np.pi * dq_left / self.ENCODER_RESOLUTION
        angle_right = 2 * np.pi * dq_right / self.ENCODER_RESOLUTION
        dist_left = angle_left * self.WHEEL_RADIUS
        dist_right = angle_right * self.WHEEL_RADIUS

        self.v = (dist_left + dist_right) / (2 * dt)
        self.w = (dist_right - dist_left) / (self.WHEEL_SEPARATION * dt)

        # TODO: Update x_odom, y_odom and O_odom accordingly
        self.x_odom = self.x_odom + self.v * dt * np.cos(self.O_odom)
        self.y_odom = self.y_odom + self.v * dt * np.sin(self.O_odom)
        self.O_odom = self.O_odom + self.w * dt 
        print(f"v: {self.v}, w: {self.w}, x: {self.x_odom}, y: {self.y_odom}, O: {self.O_odom}")
        
        self.pub_enco.publish(
            Odom2Pose.coordinates_to_message(
                self.x_odom, self.y_odom, self.O_odom, sensor_state.header.stamp
            )
        )

        # Ajoute la pose à la trajectoire
        self.path.header.stamp = sensor_state.header.stamp
        self.path.poses.append(Odom2Pose.coordinates_to_message(
                self.x_odom, self.y_odom, self.O_odom, sensor_state.header.stamp
            ))
        self.pub_path.publish(self.path)
        print(f"Nb poses dans path: {len(self.path.poses)}")
    

        

def main(args=None):
    try:
        rclpy.init(args=args)
        node = Odom2Pose()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
