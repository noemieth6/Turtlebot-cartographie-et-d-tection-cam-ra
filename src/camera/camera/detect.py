#! /usr/bin/env python3
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String # Pour publier la direction détectée
from sensor_msgs.msg import LaserScan # Pour récupérer les données du LIDAR


from rclpy.qos import QoSProfile, ReliabilityPolicy # Pour la QoS (qualité de service) de la souscription au LIDAR


class ArrowDetector(Node):

    def __init__(self):
        """Initialise le noeud ArrowDetector"""

        super().__init__("arrow_detector")

        # Initialise le pont entre ROS Image et OpenCV
        self.bridge = CvBridge()

        # Publisher pour l'image annotée
        self.pub = self.create_publisher(Image, "detections", 1)

        # Publisher pour la direction détectée (gauche/droite)
        self.pub_dir = self.create_publisher(String, "arrow_direction", 1)

        # Plage HSV pour le rouge (format openCV)
        self.red_lower = np.array([160, 100, 100])
        self.red_upper = np.array([179, 255, 255])

        # Plage HSV pour le bleu
        self.blue_lower = np.array([95, 146, 63])
        self.blue_upper = np.array([110, 255, 150])


        self.sub = self.create_subscription(Image, "image_rect", self.callback, 1)

        # Initialisation variable pour détection mur devant
        self.obstacle_near = False
        
        # Création d'un QoS pour la souscription au LIDAR
        # Utilisation de BEST_EFFORT pour la fiabilité, car on veut des données en temps réel
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT) 
        
        # Souscription au LIDAR avec QoS
        self.create_subscription(LaserScan, "/scan", self.lidar_callback, qos)


    def lidar_callback(self, msg):
        """Callback pour traiter les données du LIDAR : si un obstacle est détecté devant, on ne fait pas de détection d'image.
        msg : LaserScan, message du LIDAR contenant les distances mesurées"""
        
        angle_min = msg.angle_min  # angle de départ du scan en radians
        angle_increment = msg.angle_increment # incrément angulaire entre chaque mesure en radians
        ranges = msg.ranges # distances mesurées par le LIDAR

        # On récupère les indices des mesures LIDAR qui sont situées dans la zone frontale (plus ou moins 10° soit 0.17 rad) et qui sont valides
        front_indices = []
        for i, distance in enumerate(ranges): # On parcourt les mesures du LIDAR
            angle = angle_min + i * angle_increment  # Calcul de l'angle correspondant à la mesure
            if -0.17 <= angle <= 0.17 and not np.isnan(distance):  # On garde si l'angle est dans la zone frontale et la mesure valide
                front_indices.append(i) # On ajoute à la liste des indices valides

        # On crée une liste des distances correspondant aux indices sélectionnés dans la zone frontale
        front_distances = []
        for i in front_indices:
            front_distances.append(ranges[i])

        # Si aucune distance valide n'est trouvée, on ne fait pas de détection d'image
        if not front_distances:
            self.get_logger().info("Pas de mesure valide à l'avant.")
            self.obstacle_near = False
            return

        min_distance = min(front_distances) # On récupère la distance minimale dans la zone frontale
        self.obstacle_near = min_distance < 0.4 # Si un obstacle est détecté à moins de 0.4 m, on fait la détection d'image



    def callback(self, msg: Image):
        """Callback pour traiter les images reçues
        msg : Image, message ROS contenant l'image à traiter"""

        if not self.obstacle_near:
            self.get_logger().info("Aucun obstacle détecté")
            # On publie un message vide pour indiquer qu'il n'y a pas de détection
            self.pub_dir.publish(String(data="Attente nouvelle flèche...")) # Publie un message indiquant qu'il n'y a pas de direction détectée
            return  # Ne pas faire de détection

        try:
            # Conversion du message ROS Image en image OpenCV
            img = self.bridge.imgmsg_to_cv2(msg, "bgr8")

        except CvBridgeError as e:
            self.get_logger().warn(f"ROS->OpenCV {e}")
            return

        img_out = self.detect(img) # Détection des flèches dans l'image

        # Convert OpenCV -> ROS
        # Publication de l'image annotée
        try:
            format = "bgr8" if img_out.ndim == 3 else "mono8"
            msg_out = self.bridge.cv2_to_imgmsg(img_out, format)

        except CvBridgeError as e:
            self.get_logger().warn(f"ROS->OpenCV {e}")
            return
        
        self.pub.publish(msg_out) # Publie l'image annotée



    def detect(self, img: np.ndarray) -> np.ndarray:
        """
        Détecte les flèches rouges et bleues dans l'image.
        :param img: Image OpenCV dans laquelle détecter les flèches
        :return: Image OpenCV avec les flèches détectées dessinées dessus
        """

        # Conversion en espace de couleur HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Masque pour le rouge 
        mask_red = cv2.inRange(hsv, self.red_lower, self.red_upper)


        # Masque pour le bleu
        mask_blue = cv2.inRange(hsv, self.blue_lower, self.blue_upper)


        # Recherche des contours pour le rouge et le bleu
        contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Initialisation de la direction détectée
        direction_msg = String()
        detected_direction = None

        # On dessine les contours des zones rouges et bleues détectées 
        if contours_red:
            cv2.drawContours(img, contours_red, -1, (0, 0, 255), 3)  # Rouge

        if contours_blue:
            cv2.drawContours(img, contours_blue, -1, (255, 0, 0), 3)  # Bleu


        # Calcul de la plus grande zone rouge et bleue détectée
        max_red = max([cv2.contourArea(c) for c in contours_red], default=0)

        max_blue = max([cv2.contourArea(c) for c in contours_blue], default=0)

        # Détermination de la direction en fonction de la plus grande zone de couleur détectée
        if max_red > 20000 and max_red>max_blue:
            detected_direction = "droite"

        elif max_blue > 20000 and max_blue>max_red:
            detected_direction = "gauche"

        
        # Publication de la direction détectée
        if detected_direction:
            direction_msg.data = detected_direction
            self.pub_dir.publish(direction_msg) # Publie la direction détectée sur arrow_direction
            self.get_logger().info(f"Flèche détectée : {detected_direction}")

        return img


def main(args=None):
    import rclpy
    rclpy.init(args=args)
    node = ArrowDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass



if __name__ == "__main__":



    main()

