#! /usr/bin/env python3

import numpy as np
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py.point_cloud2 import read_points_numpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path # pour pouvoir tracer les poses





class ICPPoseEstimator(Node):
    def __init__(self):
        super().__init__("icp_pose_estimator")
        self.get_logger().info("ICP Pose Estimator node started")


        self.min_points = 5
        self.ref_points = None
        self.ref_T = np.eye(4)
  
        # Liste de tous les points obtenus par l'ICP
        self.cumulated_icp_points = [] 

        #---Subscriber---#
        self.create_subscription(PoseStamped, "/pose_enco", self.enco_callback, 10)
        self.create_subscription(PointCloud2, "/points", self.cloud_callback, 10)
        

        self.last_enco_orientation = None  # Pour stocker la dernière orientation reçue

        #---Publisher---#
        self.pub_pose_icp = self.create_publisher(PoseStamped, "/cluster_pose", 10)
        self.path_icp = self.create_publisher(Path, "/path_icp", 10) # 

        # tracer les poses successives
        self.path = Path() # permet de réinitialiser à chaque lancement
        
        # Forcer le frame_id
        self.path.header.frame_id = "odom"
    
    def enco_callback(self, msg: PoseStamped):
        """Permet d'établir la première orientation mais est ensuite adapté à travers les itérations ICP
        """  
        # Stocke le quaternion de la dernière pose enco reçue
        q = msg.pose.orientation
        rot_matrix = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        self.ref_T[:3, :3] = rot_matrix
        self.last_enco_orientation = (q.x, q.y, q.z, q.w)
  


    def cloud_callback(self, msg: PointCloud2):
        self.get_logger().info("callback started")

        xyz = read_points_numpy(msg, ["x", "y", "z"])

        # Too small: remove reference and wait for a new one
        if xyz.shape[0] <= self.min_points:
            self.ref_points = None
            

        # Replace the reference if necessary
        if self.ref_points is None:
            # Reference needs more than 'min_points' xyz
            if xyz.shape[0] > self.min_points:
                pose_vect3 = ICPPoseEstimator.estimate_pose_from_points(xyz)
                # self.ref_T = np.eye(4)
                self.ref_T[:3, 3] = pose_vect3[:,0]
                self.ref_points = xyz
            else:
                self.get_logger().info("Pas assez de points")
                return
            
   

        # Get T between ref and xyz
        T_current = ICPPoseEstimator.apply_icp(self.ref_points, xyz)
        # Previous T -> current T
        self.ref_T = self.ref_T @ T_current
        self.ref_points = xyz

        self.cumulated_icp_points.append(np.array(self.ref_T[:3, 3], dtype=float).flatten())
        # Publish pose
        pose = PoseStamped()
        pose.header.stamp = msg.header.stamp
        pose.header.frame_id = "odom"
        pos, ori = pose.pose.position, pose.pose.orientation
        pos.x, pos.y, pos.z = self.ref_T[:3, 3]
        ori.x, ori.y, ori.z, ori.w = R.from_matrix(self.ref_T[:3, :3]).as_quat()
        
        # Publication des poses individuelles
        self.pub_pose_icp.publish(pose)

        # Publication du chemin en entier
        self.path.header.stamp = msg.header.stamp
        points_stamped = self.cumulated_points_to_poses()
        self.path.poses = points_stamped 
        self.path_icp.publish(self.path)

    def cumulated_points_to_poses(self)-> PoseStamped:
        """Transformation des points en message pour rviz2.
        :return: Message PoseStamped
        """
        poses = []
        for i, point in enumerate(self.cumulated_icp_points):
            pose = PoseStamped()
            pose.header.frame_id = self.path.header.frame_id
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.pose.position.x = float(point[0])
            pose.pose.position.y = float(point[1])
            pose.pose.position.z = 0.0

            rotation_matrix = self.ref_T[:3, :3]
            quat = R.from_matrix(rotation_matrix).as_quat()
            pose.pose.orientation.x = quat[0]
            pose.pose.orientation.y = quat[1]
            pose.pose.orientation.z = quat[2]
            pose.pose.orientation.w = quat[3]
            poses.append(pose)
        return poses

    @staticmethod

    @staticmethod
    def estimate_pose_from_points(xyz: np.ndarray) -> np.ndarray:
        """Return the cluster's center as a (3*1) matrix."""
        # Return the cluster's center as a (3*1) matrix
        return np.mean(xyz, axis=0).reshape(3, 1)

    @staticmethod
    def apply_icp(S: np.ndarray, D: np.ndarray, max_iter: int = 20, tol: float = 1e-2, max_distance: float = 0.5):
        """Apply ICP from S to D.
        :param S: source pointcloud
        :param D: destination pointcloud
        :param max_iter: How many iterations to do at most
        :param tol: Error tolereance between 2 consecutive errors before stopping the process
        :return: (4*4) transform matrix
        """
        T = np.eye(4)
        error_prev = float("-inf")
        error_current = float("inf")
        ST = S.copy()  # Work on a copy
        i = 0
        
        # Procédure d'ICP
        while i < max_iter and abs(error_current - error_prev) > tol:

            voisin_ST_dans_D  = ICPPoseEstimator.find_nearest_neighbors_euclidian(ST, D)
            
            Tlocal = ICPPoseEstimator.best_fit_transform(ST, voisin_ST_dans_D)
            ST = ICPPoseEstimator.apply_transform(ST, Tlocal)
            T = T @ Tlocal
            
            error_prev = error_current
            error_current = np.mean(np.linalg.norm(voisin_ST_dans_D, axis=1))
            i = i+1
            
        return T

    @staticmethod
    def apply_transform(xyz: np.ndarray, T: np.ndarray) -> np.ndarray:
        """Apply transform T on points.
        :param points: (N*3) matrix of x, y and z coordinates
        :param T: (4*4) transformation matrix
        :return: (N*3) matrix of transformed x, y and z coordinates
        """
        xyz_homogeneous = np.hstack((xyz, np.ones((xyz.shape[0], 1))))

        return (T @ xyz_homogeneous.T).T[:, :3]

    @staticmethod
    def find_nearest_neighbors_euclidian(S: np.ndarray, D: np.ndarray) -> np.ndarray:
        """Find for each point of S, the closest point in D
        :param S: source pointcloud
        :param D: destination pointcloud
        :return: list of points
        """
        matched = []

        # Ajoute le point de plus proche de p_s dans D à la liste matched
        for s_p in S:
            d_p = np.linalg.norm(D - s_p, axis=1)
            d_pmin = D[np.argmin(d_p)]
            matched.append(d_pmin)
        return np.array(matched)

    @staticmethod
    def best_fit_transform(S, D):
        """Finds the best transform (T) between S and D.
        :param S: source pointcloud
        :param D: destination pointcloud
        :return: (4*4) transform matrix between source and destination
        """
        # Mass center
        centroid_S = np.mean(S, axis=0)
        centroid_D = np.mean(D, axis=0)

        # Center pointclouds in zero
        SS = S - centroid_S
        DD = D - centroid_D
        H = SS.T @ DD

        # If the covariance matrix H has NaN or Inf, something is wrong
        if not np.all(np.isfinite(H)):
            return np.eye(4)

        try:
            U, _, VT = np.linalg.svd(H)
        except np.linalg.LinAlgError:
            # SVD did not converge
            return np.eye(4)

        R_mat = VT.T @ U.T

        # Reflecion detected
        if np.linalg.det(R_mat) < 0:
            VT[2, :] *= -1
            R_mat = VT.T @ U.T

        t = centroid_D - R_mat @ centroid_S

        T = np.eye(4)
        T[:3, :3] = R_mat
        T[:3, 3] = t
        return T


def main(args=None):
    import rclpy

    rclpy.init(args=args)
    node = ICPPoseEstimator()
    try:
        rclpy.spin(node)
        
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
