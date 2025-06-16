#!/usr/bin/env python3

import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py.point_cloud2 import create_cloud, read_points_numpy
from sensor_msgs.msg import PointField


class Clusterer(Node):
    def __init__(self):
        super().__init__("clusterer")
        
        #---Publisher---#
        self.pub = self.create_publisher(PointCloud2, "clusters", 10)

        #---Subscriber---#
        self.sub = self.create_subscription(PointCloud2, "points", self.callback, 10)
        
        #---Initialisation des points cumulés---#
        self.allpoints = []
        
        self.get_logger().info("Clusterer node started")
        


    def callback(self, msg: PointCloud2):
  
        xy = read_points_numpy(msg, ["x", "y"])

        # Apply your custom cluster function
        clusters = self.cluster(xy)

        # Rebuild the complete point-cloud with the computed cluster information
        clusters = np.reshape(clusters, (len(clusters), 1))
        intensity = read_points_numpy(msg, ["intensity"])
        clustered_points = np.hstack((xy, np.zeros((len(xy), 1)), intensity, clusters))

        # Prepare the `counts` array that describes for each point how big their clusters are.
        # Ex: [len(c0), len(c1), ...]
        cluster_ids, cluster_counts = np.unique(clusters, return_counts=True)
        counts = cluster_counts[cluster_ids[clusters]]

        # Tri des points
        filtered_points = []
        for i in range(len(clustered_points)):
            if (counts[i] > 5) and (clustered_points[i, 4] !=0):
                filtered_points.append(clustered_points[i])                    

        # Ajout à une liste globale pour affichage de la structure du labyrinthe
        self.allpoints.append(filtered_points)      

        # Forcer le frame_id
        msg.header.frame_id = "odom" 

        # Renvoie la dernière création du nuage de points complet
        self.pub.publish(create_cloud(msg.header, msg.fields, np.vstack(self.allpoints))) 
 

    def cluster(self, points: np.ndarray) -> np.ndarray:
        """Associates a cluster id with each point.

        :param points: Array of [[x0, y0], [x1, y1], ...]
        :return: Array of [c0, c1, ...]
        """
        C = np.zeros(points.shape[0], dtype=int)
        # Parameter values
        k = 3
        D = 0.1
       

        # Clustering algorithm
        for i in range(k,len(points)):
            dist = [0 for _ in range(k-1)]
            for j in range(1,k):
                dist[j-1] = np.linalg.norm(points[i]-points[i-j])
            dmin = min(dist)
            jmin = np.argmin(dist) +1
            
            if dmin<D:
                if C[i-jmin] ==0:
                    C[i-jmin] = max(C)+1   
                C[i] = C[i-jmin]				
        print("Clusters found:", np.unique(C))
        return C


def main(args=None):
    import rclpy
    print("Starting clusterer node...")

    rclpy.init(args=args)
    node = Clusterer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
