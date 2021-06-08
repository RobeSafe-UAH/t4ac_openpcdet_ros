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
from t4ac_msgs.msg import BEV_detection, BEV_detections_list

# Math and geometry imports
import torch
import math
import numpy as np
from pyquaternion import Quaternion

# Auxiliar functions/classes imports
from modules.auxiliar_functions_multihead import marker_bb, marker_arrow, filter_predictions, relative2absolute_velocity
from modules.processor_ros_nuscenes import Processor_ROS_nuScenes

def display_rviz(msg, boxes, scores, labels):

    marker_array = MarkerArray()
    i = 0

    for box, score, label in zip(boxes, scores, labels):
        box_marker = marker_bb(msg, box, score, label, i)
        marker_array.markers.append(box_marker)
        i += 1

        arrow_marker = marker_arrow(msg, box, score, label, i)
        marker_array.markers.append(arrow_marker)
        i += 1

    pub_rviz.publish(marker_array)

def ros_odometry_callback(msg):
    
    processor.odometry = msg

def ros_lidar_callback(msg):

    processor.new_pcl(msg)
    pred_dicts = processor.inference()

    pred_boxes, pred_scores, pred_labels = filter_predictions(pred_dicts, True)
    
    if processor.odometry != None and len(pred_boxes) != 0:
        pred_boxes = relative2absolute_velocity(pred_boxes, processor.odometry)

    # print(pred_boxes)
    # print(pred_scores)
    # print(pred_labels)

    display_rviz(msg, pred_boxes, pred_scores, pred_labels)

if __name__ == "__main__":

    last_msg_odometry = None

    config_path = rospy.get_param("/t4ac/perception/detection/lidar/t4ac_openpcdet_ros/t4ac_openpcdet_ros_node/config_path")
    model_path = rospy.get_param("t4ac/perception/detection/lidar/t4ac_openpcdet_ros/t4ac_openpcdet_ros_node/model_path")

    processor = Processor_ROS_nuScenes(config_path, model_path)
    processor.initilize_network()
    
    node_name = rospy.get_param("/t4ac/perception/detection/lidar/t4ac_openpcdet_ros/t4ac_openpcdet_ros_node/node_name")
    rospy.init_node(node_name, anonymous=True)
    
    # ROS publishers

    # BEV_lidar_obstacles_topic = rospy.get_param("/t4ac/perception/detection/lidar/t4ac_openpcdet_ros/t4ac_openpcdet_ros_node/pub_BEV_lidar_obstacles")
    # pub_detected_obstacles = rospy.Publisher(BEV_lidar_obstacles_topic, BEV_detections_list, queue_size=20)

    lidar_3D_obstacles_markers_topic = rospy.get_param("/t4ac/perception/detection/lidar/t4ac_openpcdet_ros/t4ac_openpcdet_ros_node/pub_3D_lidar_obstacles_markers")
    pub_rviz = rospy.Publisher(lidar_3D_obstacles_markers_topic, MarkerArray, queue_size=3)

    # pub_pcl2 = rospy.Publisher("/carla/ego_vehicle/pcl2_used", PointCloud2, queue_size=20)

    # ROS subscriber

    input_odometry_topic = rospy.get_param("/t4ac/perception/detection/lidar/t4ac_openpcdet_ros/t4ac_openpcdet_ros_node/sub_input_odometry")
    sub_input_odometry = rospy.Subscriber(input_odometry_topic, Odometry, ros_odometry_callback, queue_size=1)

    input_pointcloud_topic = rospy.get_param("/t4ac/perception/detection/lidar/t4ac_openpcdet_ros/t4ac_openpcdet_ros_node/sub_input_pointcloud")
    sub_input_pointcloud = rospy.Subscriber(input_pointcloud_topic, PointCloud2, ros_lidar_callback, queue_size=1, buff_size=2**24)

    print("[+] PCDet ros_node has started.")    
    rospy.spin()
