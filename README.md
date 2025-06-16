# SY31 Labyrinthe

Projet ROS 2 pour la cartographie d’un labyrinthe par un robot manuel, utilisant l’odométrie (encodeurs, gyromètre, LiDAR) et la détection de flèches colorées (caméra) pour indiquer à l’opérateur quand tourner à gauche (rouge) ou à droite (bleu).

## Installation

Compiler et sourcer le workspace :
    ```sh
    cd ~/sy31-labyrinthe
    colcon build --symlink-install
    source install/setup.bash
    ```

## Lancement du projet
Pour lancer tous les nœuds nécessaires, lancer le fichier de lancement ROS2 **en étant dans le dossier** sy31-labyrinthe/src/launch :

```sh
ros2 launch launcher.xml
```

## Visualisation

- La direction détectée (gauche/droite) est publiée sur le topic `/arrow_direction` et des loggers s'affichent aussi dans le terminal du launch.
    - Pour la voir, ouvrir un autre terminal et taper :
      ```sh
      ros2 topic echo /arrow_direction
      ```

Pour visualiser la trajectoire et la cartographie, lancez RViz2, allez dans Fichier et ajoutez la configuration :
`rviz_config.rviz`.
Pour l'ICP (pas complètement fonctionnel), ajoutez `rviz_config_ICP.rviz`.


## Structure du projet
- `camera/camera/detect.py` : nœud principal de détection de flèches et obstacles
- `camera/camera/decompressor.py` : décompression du flux image
- `lidar/lidar/transformer.py` : transformation du nuage de points LiDAR
- `lidar/lidar/clusterer.py` : clustering des points LiDAR
- `lidar/lidar/icp_pose_estimator.py` : estimation de la pose par ICP
- `odom/odom/odompose.py` : odométrie à partir des encodeurs et gyromètre

## Auteurs
Agathe RABASSE et Noémie THOMASSON dans le cadre de l'UV SY31 à l'UTC


