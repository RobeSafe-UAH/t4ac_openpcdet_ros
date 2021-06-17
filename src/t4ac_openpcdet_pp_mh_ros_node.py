#!/usr/bin/python3.8

"""
Created on Tue May 25 13:00:39 2021

@author: Javier de la Peña

Code to 

Communications are based on ROS (Robot Operating Sytem)

Inputs: LiDAR pointcloud [x, y ,z, intensity, sweep]
Outputs: Most relevant obstacles of the environment in the form of 3D bounding boxes with their associated velocity
"""

# General use imports
import os
import time
import sys
import copy
import json
import argparse
import glob
from pathlib import Path

# ROS imports
import rospy
from sensor_msgs.msg import PointCloud2, PointField, Image, CameraInfo
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64, Float32, Header, Bool
from message_filters import TimeSynchronizer, Subscriber, ApproximateTimeSynchronizer
from visualization_msgs.msg import MarkerArray
from t4ac_msgs.msg import Bounding_Box_3D, Bounding_Box_3D_list

# Math and geometry imports
import torch
import math
import numpy as np
from pyquaternion import Quaternion

# Auxiliar functions/classes imports
from modules.auxiliar_functions_multihead import get_bounding_box_3d, marker_bb, marker_arrow, filter_predictions, relative2absolute_velocity
from modules.processor_ros_nuscenes import Processor_ROS_nuScenes

# Config PointCloud processor

CONFIG_PATH = rospy.get_param("/t4ac/perception/detection/lidar/t4ac_openpcdet_ros/t4ac_openpcdet_ros_node/config_path")
MODEL_PATH = rospy.get_param("t4ac/perception/detection/lidar/t4ac_openpcdet_ros/t4ac_openpcdet_ros_node/model_path")

class OpenPCDet_ROS():
    """
    """
    def __init__(self):

        # PointCloud processor

        self.processor = Processor_ROS_nuScenes(CONFIG_PATH, MODEL_PATH)
        self.processor.initilize_network()

        # ROS Subscribers

        input_odometry_topic = rospy.get_param("/t4ac/perception/detection/lidar/t4ac_openpcdet_ros/t4ac_openpcdet_ros_node/sub_input_odometry")
        self.sub_input_odometry = rospy.Subscriber(input_odometry_topic, Odometry, self.ros_odometry_callback, queue_size=1)

        input_pointcloud_topic = rospy.get_param("/t4ac/perception/detection/lidar/t4ac_openpcdet_ros/t4ac_openpcdet_ros_node/sub_input_pointcloud")
        self.sub_input_pointcloud = rospy.Subscriber(input_pointcloud_topic, PointCloud2, self.ros_lidar_callback, queue_size=1, buff_size=2**24)

        # ROS Publishers

        # pub_pcl2 = rospy.Publisher("/carla/ego_vehicle/pcl2_used", PointCloud2, queue_size=20)
        lidar_3D_obstacles_markers_topic = rospy.get_param("/t4ac/perception/detection/lidar/t4ac_openpcdet_ros/t4ac_openpcdet_ros_node/pub_3D_lidar_obstacles_markers")
        self.pub_lidar_3D_obstacles_markers = rospy.Publisher(lidar_3D_obstacles_markers_topic, MarkerArray, queue_size=10)

        lidar_3D_obstacles_topic = rospy.get_param("/t4ac/perception/detection/lidar/t4ac_openpcdet_ros/t4ac_openpcdet_ros_node/pub_3D_lidar_obstacles")
        self.pub_lidar_3D_obstacles = rospy.Publisher(lidar_3D_obstacles_topic, Bounding_Box_3D_list, queue_size=10)

        self.laser_frame = rospy.get_param('/t4ac/frames/laser')
        self.header = None

    # Aux Functions

    def publish_obstacles(self, boxes, scores, labels):
        """
        """

        bounding_box_3d_list = Bounding_Box_3D_list()
        bounding_box_3d_list.header.stamp = self.header.stamp
        bounding_box_3d_list.header.frame_id = self.header.frame_id

        marker_array = MarkerArray()
        i = 0

        for box, score, label in zip(boxes, scores, labels):
            box_marker = marker_bb(self.header, box, score, label, i)
            marker_array.markers.append(box_marker)
            i += 1

            arrow_marker = marker_arrow(self.header, box, score, label, i)
            marker_array.markers.append(arrow_marker)
            i += 1

            bounding_box_3d = get_bounding_box_3d(box,score,label)
            bounding_box_3d_list.bounding_box_3d_list.append(bounding_box_3d)

        self.pub_lidar_3D_obstacles.publish(bounding_box_3d_list)
        self.pub_lidar_3D_obstacles_markers.publish(marker_array)

    # ROS callbacks

    def ros_odometry_callback(self,odom_msg):
        """
        """
        self.processor.odometry = odom_msg

    def ros_lidar_callback(self,point_cloud_msg):
        """
        """

        self.header = point_cloud_msg.header

        self.processor.new_pcl(point_cloud_msg)
        pred_dicts = self.processor.inference()

        pred_boxes, pred_scores, pred_labels = filter_predictions(pred_dicts, True)
        
        if self.processor.odometry != None and len(pred_boxes) != 0:
            pred_boxes = relative2absolute_velocity(pred_boxes, self.processor.odometry)

        # print(pred_boxes)
        # print(pred_scores)
        # print(pred_labels)

        self.publish_obstacles(pred_boxes, pred_scores, pred_labels)

def main():
    # Node name

    node_name = rospy.get_param("/t4ac/perception/detection/lidar/t4ac_openpcdet_ros/t4ac_openpcdet_ros_node/node_name")
    rospy.init_node(node_name, anonymous=True)
    
    OpenPCDet_ROS()

    try:
        rospy.spin()
    except KeyboardInterruput:
        rospy.loginfo("Shutting down OpenPCDet ROS module")

if __name__ == '__main__':
    print("[+] PCDet ros_node has started.")    
    main()







    

    

    


    

    

    
