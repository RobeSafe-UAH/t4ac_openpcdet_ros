#! /usr/bin/env python3.8
# N.B. Modify here your python interpreter

"""
Created on Thu Aug  6 11:27:43 2020

@author: Javier del Egido Sierra and Carlos Gómez-Huélamo

Code to 

Communications are based on ROS (Robot Operating Sytem)

Inputs: LiDAR pointcloud
Outputs: Most relevant obstacles of the environment in the form of 3D bounding boxes

Note that each obstacle shows an unique ID in addition to its semantic information (person, car, ...), 
in order to make easier the decision making processes.

Executed via Python3.8 (python3.8 inference.py)
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
import ros_numpy
from sensor_msgs.msg import PointCloud2, PointField, Image, CameraInfo
from std_msgs.msg import Float64, Float32, Header, Bool
from message_filters import TimeSynchronizer, Subscriber, ApproximateTimeSynchronizer
from sensor_msgs.msg import PointCloud2, PointField
from visualization_msgs.msg import MarkerArray, Marker
from t4ac_msgs.msg import BEV_detection, BEV_detections_list

# OpenPCDet imports
from pcdet.config import cfg, cfg_from_yaml_file

# Math and geometry imports
import math
import numpy as np
from pyquaternion import Quaternion

# Auxiliar functions/classes imports
from modules.auxiliar_functions import *
from modules.processor_ros import Processor_ROS
from modules.demodataset import DemoDataset

# Global variables
calib_file = rospy.get_param("/t4ac/perception/detection/t4ac_openpcdet_ros/t4ac_openpcdet_pp_ros_node/calib_file")

move_lidar_center = 20 
threshold = 0.5

inference_time_list = []

display_rviz = True
bev_camera = True

def xyz_array_to_pointcloud2(points_sum, stamp=None, frame_id=None):
    """
    Create a sensor_msgs.PointCloud2 from an array of points.
    """
    msg = PointCloud2()
    if stamp:
        msg.header.stamp = stamp
    if frame_id:
        msg.header.frame_id = frame_id
    msg.height = 1
    msg.width = points_sum.shape[0]
    msg.fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1)
        # PointField('i', 12, PointField.FLOAT32, 1)
        ]
    msg.is_bigendian = False
    msg.point_step = 12
    msg.row_step = points_sum.shape[0]
    msg.is_dense = int(np.isfinite(points_sum).all())
    msg.data = np.asarray(points_sum, np.float32).tostring()
    return msg

def anno_to_bev_detections(dt_box_lidar, scores, types, msg):
    """
    """

    detected_3D_objects_marker_array = MarkerArray()

    bev_detections_list = BEV_detections_list()
    bev_detections_list.header.stamp = msg.header.stamp
    bev_detections_list.header.frame_id = msg.header.frame_id 

    point_cloud_range = cfg.DATA_CONFIG.POINT_CLOUD_RANGE
    bev_detections_list.front = point_cloud_range[3] - move_lidar_center
    bev_detections_list.back = point_cloud_range[0] - move_lidar_center
    bev_detections_list.left = point_cloud_range[1]
    bev_detections_list.right = point_cloud_range[4]

    if scores.size != 0:
        for i in range(scores.size):
            if scores[i] > threshold: 
                z = float(dt_box_lidar[i][2])
                l = float(dt_box_lidar[i][3])
                w = float(dt_box_lidar[i][4])
                yaw = float(dt_box_lidar[i][6])

                x_corners = [-l/2,-l/2,l/2, l/2]
                y_corners = [ w/2,-w/2,w/2,-w/2]
                z_corners = [0,0,0,0]

                if yaw > math.pi:
                    yaw -= math.pi

                if bev_camera: # BEV camera frame
                    x = -float(dt_box_lidar[i][1])
                    y = -(float(dt_box_lidar[i][0]) - move_lidar_center)
                    yaw_bev = yaw - math.pi/2
                else: # BEV LiDAR frame
                    x = float(dt_box_lidar[i][0]) - move_lidar_center
                    y = float(dt_box_lidar[i][1])
                    yaw_bev = yaw
  
                R = rotz(-yaw_bev)

                corners_3d = np.dot(R, np.vstack([x_corners,y_corners,z_corners]))[0:2]

                bev_detection = BEV_detection()
                bev_detection.type = str(int(types[i]))
                bev_detection.score = scores[i]

                bev_detection.x = float(dt_box_lidar[i][0]) - move_lidar_center
                bev_detection.y = float(dt_box_lidar[i][1])
                bev_detection.tl_br = [0,0,0,0] #2D bbox top-left, bottom-right  xy coordinates
                                                # Upper left     Upper right      # Lower left     # Lower right
                bev_detection.x_corners = [corners_3d[0,0], corners_3d[0,1], corners_3d[0,2], corners_3d[0,3]]
                bev_detection.y_corners = [corners_3d[1,0], corners_3d[1,1], corners_3d[1,2], corners_3d[1,3]]
                bev_detection.l = l
                bev_detection.w = w
                bev_detection.o = -yaw_bev

                bev_detections_list.bev_detections_list.append(bev_detection)

                if display_rviz:
                    detected_3D_object_marker = Marker()
                    detected_3D_object_marker.header.stamp = msg.header.stamp
                    detected_3D_object_marker.header.frame_id = msg.header.frame_id
                    detected_3D_object_marker.type = Marker.CUBE
                    detected_3D_object_marker.id = i
                    detected_3D_object_marker.lifetime = rospy.Duration.from_sec(1)
                    detected_3D_object_marker.pose.position.x = float(dt_box_lidar[i][0]) - move_lidar_center
                    detected_3D_object_marker.pose.position.y = float(dt_box_lidar[i][1])
                    detected_3D_object_marker.pose.position.z = z
                    q = yaw2quaternion(yaw)
                    detected_3D_object_marker.pose.orientation.x = q[1] 
                    detected_3D_object_marker.pose.orientation.y = q[2]
                    detected_3D_object_marker.pose.orientation.z = q[3]
                    detected_3D_object_marker.pose.orientation.w = q[0]
                    detected_3D_object_marker.scale.x = l
                    detected_3D_object_marker.scale.y = w
                    detected_3D_object_marker.scale.z = 3
                    detected_3D_object_marker.color.r = 0
                    detected_3D_object_marker.color.g = 0
                    detected_3D_object_marker.color.b = 255
                    detected_3D_object_marker.color.a = 0.5
                    detected_3D_objects_marker_array.markers.append(detected_3D_object_marker)

    pub_detected_obstacles.publish(bev_detections_list)
    pub_rviz.publish(detected_3D_objects_marker_array)

    return 

def rslidar_callback(msg):
    """
    """

    frame = msg.header.seq

    if len(msg.data) > 1000:
        msg_cloud = ros_numpy.point_cloud2.pointcloud2_to_array(msg)
        np_p = get_xyz_points(msg_cloud, True)

        scores, dt_box_lidar, types, pred_dict = proc_1.run(np_p, calib, frame, move_lidar_center)
        anno_to_bev_detections(dt_box_lidar, scores, types, msg)
    else:
        bev_detections_list = BEV_detections_list()
        bev_detections_list.header.stamp = msg.header.stamp
        bev_detections_list.header.frame_id = msg.header.frame_id 

        bev_detections_list.bev_detections_list.append([])

        pub_detected_obstacles.publish(bev_detections_list)
 
if __name__ == "__main__":

    # config_path = os.path.join(cfg_root,"kitti_models/pointpillars.yaml")
    # model_path  = os.path.join(cfg_root,"kitti_models/pointpillars.pth")

    config_path = rospy.get_param("/t4ac/perception/detection/t4ac_openpcdet_ros/t4ac_openpcdet_pp_ros_node/config_path")
    model_path = rospy.get_param("t4ac/perception/detection/t4ac_openpcdet_ros/t4ac_openpcdet_pp_ros_node/model_path")

    proc_1 = Processor_ROS(config_path, model_path)
    print("Config path: ", config_path)
    print("Model path: ", model_path)
    proc_1.initialize()
    calib = proc_1.get_calib(calib_file)

    calib.P3 = calib.P2
    print("Calib.P2: ", calib.P2)
    print("Calib.P3: ", calib.P3)
    print("Calib.R0: ", calib.R0)
    print("Calib.T (Velo2Cam): ", calib.V2C)
    
    node_name = rospy.get_param("/t4ac/perception/detection/t4ac_openpcdet_ros/t4ac_openpcdet_pp_ros_node/node_name")
    rospy.init_node(node_name, anonymous=True)

    cfg_from_yaml_file(config_path, cfg)
    
    # ROS publishers

    BEV_lidar_obstacles_topic = rospy.get_param("/t4ac/perception/detection/t4ac_openpcdet_ros/t4ac_openpcdet_pp_ros_node/pub_BEV_lidar_obstacles")
    pub_detected_obstacles = rospy.Publisher(BEV_lidar_obstacles_topic, BEV_detections_list, queue_size=20)

    lidar_3D_obstacles_markers_topic = rospy.get_param("/t4ac/perception/detection/t4ac_openpcdet_ros/t4ac_openpcdet_pp_ros_node/pub_3D_lidar_obstacles_markers")
    pub_rviz = rospy.Publisher(lidar_3D_obstacles_markers_topic, MarkerArray, queue_size=20)

    # ROS subscriber

    input_pointcloud_topic = rospy.get_param("/t4ac/perception/detection/t4ac_openpcdet_ros/t4ac_openpcdet_pp_ros_node/sub_input_pointcloud")
    sub_input_pointcloud = rospy.Subscriber(input_pointcloud_topic, PointCloud2, rslidar_callback, queue_size=1, buff_size=2**24)

    print("[+] PCDet ros_node has started.")    
    rospy.spin()
